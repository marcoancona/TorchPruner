from __future__ import print_function
import argparse
from timeit import default_timer as timer
import torch
from shapley_pruning.prunable import ContinuousPruner
from torchsummary import summary, torchsummary

from experiments.utils import (
    now,
    log_dict,
    get_parameter_count_and_flops,
    get_layer_sizes,
    save_model_state,
    load_model_state,
)
import experiments.models.fmnist as fmnist
import experiments.models.cifar10 as cifar10


experiments = {"fmnist": fmnist, "cifar10": cifar10}

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str)
parser.add_argument("--load", type=str)
parser.add_argument("--save", action="store_true", default=False)

parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    metavar="N",
    help="number of epochs to train (default: 50)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: not set)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=60,
    metavar="N",
    help="how many batches to wait before logging training status",
)

parser.add_argument("--sparsity", help="0 = dynamic", type=float, default=1.0)
parser.add_argument(
    "--max-loss-gap", help="Max increment of loss at each pruning iteration", type=float
)
parser.add_argument("--pruning-steps", type=int, default=0)
parser.add_argument("--pruning-start", type=int, default=1)
parser.add_argument("--pruning-interval", type=int, default=1)
parser.add_argument("--pruning-logic", type=str)

args = parser.parse_args()
print(args)
experiment = experiments[args.experiment]
if experiments is None:
    raise RuntimeError(f"--model not valid, must be in {list(experiments.keys())}")
assert args.sparsity >= 0, "--sparsity must be > 0 (or 0 for dynamic sparsity)"

sparsity = f"s:{args.sparsity}" if args.sparsity > 0 else f"gap:{args.max_loss_gap}"
experiment_id = f"{args.experiment}__{args.pruning_logic}_{sparsity}_steps:{args.pruning_steps}_start:{args.pruning_start}_int:{args.pruning_interval}"
if args.load is not None:
    experiment_id += f"_load:{args.load}"
timestamp_id = now()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    samples = 0
    cumulative_loss = 0
    start = timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = experiment.loss(output, target)
        cumulative_loss += loss.item() * len(target)

        current_pred = output.argmax(dim=1, keepdim=True)
        correct += current_pred.eq(target.view_as(current_pred)).sum().item()
        samples += len(target)

        # Perform optimization step
        loss.backward()
        optimizer.step()

        if batch_idx % (args.log_interval * 1) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {:.3f}\t Time: {:.1f}s".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    correct / samples,
                    timer() - start,
                )
            )
            start = timer()

    return cumulative_loss / samples, correct / samples


def test(args, model, device, test_loader):
    model.eval()
    cum_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            cum_test_loss += experiment.loss(output, target).item() * len(target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    avg_test_loss = cum_test_loss / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n".format(
            avg_test_loss, correct, len(test_loader.dataset), 100.0 * test_accuracy
        )
    )
    return avg_test_loss, test_accuracy


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    (
        dataset,
        train_loader,
        validation_loader,
        test_loader,
    ) = experiment.get_dataset_and_loaders(use_cuda)

    # Print dataset info and infer input size
    batch, _ = next(iter(test_loader))
    input_size = batch[0].shape
    print(f"Loaded dataset normalized in [{batch.min().item()} ; {batch.max().item()}]")

    # Get model
    model, model_name = experiment.get_model_with_name()
    model = model.to(device)
    summary(model, input_size=input_size, device="cuda" if use_cuda else "cpu")

    pruner = ContinuousPruner(
        model,
        input_size=input_size,
        pruning_graph=model.get_pruning_graph(),
        device=device,
        data_loader=validation_loader,
        test_data_loader=test_loader,  # for debugging and plotting only
        loss=experiment.loss,
        verbose=1,
        experiment_id=experiment_id,
    )

    if args.load is not None:
        load_model_state(model, model_name, args.load)

    optimizer = experiment.get_optimizer_for_model(model, 0)
    initial_parameters, initial_flops = get_parameter_count_and_flops(
        model, input_size, device="cuda" if use_cuda else "cpu"
    )

    sparsity = args.sparsity
    pruning_steps = args.pruning_steps
    pruning_start = args.pruning_start
    pruning_interval = args.pruning_interval
    max_increment_loss = args.max_loss_gap

    if pruning_start == 0 and 0 < sparsity < 1.0:
        # Pruning before training if steps == 0
        # Effectively, this means training a smaller network from scratch
        opt_state = pruner.prune(
            sparsity,
            args.pruning_logic,
            optimizer,
            max_loss_increase_percent=max_increment_loss,
            epoch=0,
        )
        optimizer = experiment.get_optimizer_for_model(model, 0)
        optimizer.load_state_dict(opt_state)

    best_model_state = None
    # best_pruner_state = None
    best_test_loss = None

    for epoch in range(1, args.epochs + 1):

        # Train epoch
        start = timer()
        train_loss, train_acc = train(
            args, model, device, train_loader, optimizer, epoch
        )
        train_time = timer() - start

        # Compute test accuracy
        test_loss, test_acc = test(args, model, device, test_loader)
        test_loss_pp, test_acc_pp = test_loss, test_acc

        if best_model_state is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            # best_pruner_state = pruner.state_dict()
            best_model_state = model.state_dict()

        # Get n params before pruning
        # Otherwise test loss does not match the network
        n_params, flops = get_parameter_count_and_flops(
            model, input_size, device="cuda" if use_cuda else "cpu"
        )
        layer_size = get_layer_sizes(model)

        # Prune and rebuild optimizer
        start = timer()
        prune_time = 0.0

        if (
            pruning_start <= epoch < pruning_start + pruning_steps * pruning_interval
            and sparsity < 1.0
        ):
            if (epoch - 1) % pruning_interval == 0:
                opt_state = pruner.prune(
                    pow(sparsity, 1 / pruning_steps),
                    args.pruning_logic,
                    optimizer,
                    max_loss_increase_percent=max_increment_loss,
                    epoch=epoch,
                )
                optimizer = experiment.get_optimizer_for_model(model, epoch)
                optimizer.load_state_dict(opt_state)
                prune_time = timer() - start
                test_loss_pp, test_acc_pp = test(args, model, device, test_loader)

        log_dict(
            "log",
            {
                "timestamp": timestamp_id,
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "test_acc_pp": test_acc_pp,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_loss_pp": test_loss_pp,
                "n_params": n_params,
                "flops": flops,
                "n_params_full": initial_parameters,
                "layers": layer_size,
                "train_time": train_time,
                "prune_time": prune_time,
                "experiment": experiment_id,
            },
        )

    # Save best model
    if args.save is True:
        save_model_state(best_model_state, model_name, timestamp_id)


if __name__ == "__main__":
    main()
