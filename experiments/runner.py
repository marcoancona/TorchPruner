from __future__ import print_function
import argparse
from timeit import default_timer as timer
import torch
import numpy as np
from torchsummary import summary, torchsummary
from .train import train, test
from torchpruner import (
    Pruner,
    RandomAttributionMetric,
    TaylorAttributionMetric,
    ShapleyAttributionMetric,
    SensitivityAttributionMetric,
    WeightNormAttributionMetric,
    APoZAttributionMetric
)


from experiments.utils import (
    now,
    log_dict,
    get_parameter_count_and_flops,
    get_layer_sizes,
    save_model_state,
    load_model_state,
    Logger
)
import experiments.models.fmnist as fmnist
import experiments.models.mnist as mnist
import experiments.models.cifar10 as cifar10
import experiments.models.transfer as cub200

#
# parser.add_argument(
#     "--activations-test", action="store_true", default=False, help="Run activation test"
# )
# parser.add_argument(
#     "--max-loss-gap", help="Max increment of loss at each pruning iteration", type=float
# )

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--load", type=str)
parser.add_argument("--save", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=-1, help="-1 = no seed")

# Pruning settings
parser.add_argument("--pr", type=float, default=1.0, help="pruning ratio")
parser.add_argument("--pruning-steps", type=int, default=0)
parser.add_argument("--pruning-start", type=int, default=1)
parser.add_argument("--pruning-interval", type=int, default=1)
parser.add_argument("--attribution", type=str)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(f"Running {args}")

# Seed for reproducibility
if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)


# Load model and data
if args.model == "mnist-fc":
    model, name = mnist.get_model_with_name()
    model = model.to(device)
    loss = mnist.loss
    opt, scheduler = mnist.get_optimizer_for_model(model)
    train_loader, val_loader, test_loader = mnist.get_dataset_and_loaders(use_cuda=use_cuda, val_split=10000)

elif args.model == "cifar10-fc":
    model, name = cifar10.get_fc_model_with_name()
    model = model.to(device)
    loss = cifar10.loss
    opt, scheduler = cifar10.get_optimizer_for_model(model)
    train_loader, val_loader, test_loader = cifar10.get_dataset_and_loaders(use_cuda=use_cuda, val_split=10000)

elif args.model == "cifar10":
    model, name = cifar10.get_model_with_name()
    model = model.to(device)
    loss = cifar10.loss
    opt, scheduler = cifar10.get_optimizer_for_model(model)
    train_loader, val_loader, test_loader = cifar10.get_dataset_and_loaders(use_cuda=use_cuda, val_split=1000)

elif args.model == "cifar10-small-val":
    model, name = cifar10.get_model_with_name()
    model = model.to(device)
    opt, scheduler = cifar10.get_optimizer_for_model(model)
    loss = cifar10.loss
    _, val_loader, test_loader = cifar10.get_dataset_and_loaders(use_cuda=use_cuda, val_split=1000, val_from_test=True)
    # Split validation data further into a reduced train set and a validation set
    train_set, val_set = torch.utils.data.random_split(val_loader.dataset, [len(val_loader.dataset) - 500, 500])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False)

else:
    raise RuntimeError("model not valid")

print (f"Train data: {len(train_loader.dataset)} examples")
print (f"Val/Pruning data: {len(val_loader.dataset)} examples")
print (f"Test data: {len(test_loader.dataset)} examples")

# Init logger
input_size = next(iter(train_loader))[0].shape
logger = Logger(args, model, input_size, device)
summary(model, input_size=input_size, device=device.type)


# Load pre-trained weights if required
if args.load is not None:
    load_model_state(model, name, args.load)


if args.attribution == "random":
    attr = RandomAttributionMetric(model, val_loader, loss, device)
elif args.attribution == "weight":
    attr = WeightNormAttributionMetric(model, val_loader, loss, device)
elif args.attribution == "apoz":
    attr = APoZAttributionMetric(model, val_loader, loss, device)
elif args.attribution == "sensitivity":
    attr = SensitivityAttributionMetric(model, val_loader, loss, device)
elif args.attribution == "taylor":
    attr = TaylorAttributionMetric(model, val_loader, loss, device)
elif args.attribution == "taylor-signed":
    attr = TaylorAttributionMetric(model, val_loader, loss, device, signed=True)
elif args.attribution == "sv":
    attr = ShapleyAttributionMetric(model, val_loader, loss, device, sv_samples=5)
else:
    if args.pr < 1.0:
        raise RuntimeError("attribution not valid")

# pruner = ContinuousPruner(
#     model,
#     input_size=input_size,
#     pruning_graph=model.get_pruning_graph(),
#     device=device,
#     data_loader=validation_loader,
#     test_data_loader=test_loader,  # for debugging and plotting only
#     loss=experiment.loss,
#     verbose=1,
#     experiment_id=experiment_id,
# )

pruner = Pruner(model, device, input_size)

pr = args.pr
pruning_steps = args.pruning_steps
pruning_start = args.pruning_start
pruning_interval = args.pruning_interval


test_loss, test_acc = test(args, model, device, test_loader)
test_loss_pp, test_acc_pp = test_loss, test_acc
prune_time = 0.0
pruning_count = 0


if pruning_start == 0 and 0 <= pr < 1.0:
    # Pruning before training if steps == 0
    # Effectively, this means training a smaller network from scratch
    for i in range(1):
        prune_start = timer()
        pruner.prune(
            # pow(sparsity, 1/args.pruning_steps),
            sparsity, # to be used when prune_all_layers is False
            args.pruning_logic,
            optimizer,
            max_loss_increase_percent=max_increment_loss,
            epoch=pruning_count,
            prune_all_layers=False
        )
        pruning_count += 1
        prune_time = timer() - prune_start
        optimizer = experiment.get_optimizer_for_model(model, 0)
        test_loss_pp, test_acc_pp = test(args, model, device, test_loader)
        flops, n_params = get_parameter_count_and_flops(model, input_size, device="cuda" if use_cuda else "cpu")
        layer_size = get_layer_sizes(model)
        test_acc, test_loss = test_acc_pp, test_loss_pp

    log_dict(
        "log",
        {
            "timestamp": timestamp_id,
            "epoch": 0,
            "train_acc": 0,
            "test_acc": test_acc,
            "test_acc_pp": test_acc_pp,
            "train_loss": 0,
            "test_loss": test_loss,
            "test_loss_pp": test_loss_pp,
            "n_params": n_params,
            "flops": flops,
            "n_params_full": initial_parameters,
            "layers": layer_size,
            "train_time": 0.,
            "prune_time": prune_time,
            "experiment": experiment_id,
        },
    )

    best_model_state = None
    best_test_loss = None

    for epoch in range(1, args.epochs + 1):

        # Train epoch
        start = timer()

        fine_tune_loss, _ = test(args, model, device, train_loader)
        torch.save(model.state_dict(), "./tmp_model.pt")
        while True:
            print("Fine tune")
            train(args, model, device, validation_loader, optimizer, epoch)
            new_test_loss, new_test_acc = test(args, model, device, train_loader)
            if new_test_loss >= fine_tune_loss:
                print ("Restoring...")
                model.load_state_dict(torch.load("./tmp_model.pt"))
                break
            fine_tune_loss = new_test_loss
            torch.save(model.state_dict(), "./tmp_model.pt")

        # if test_acc < 1:
        #     train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
        # else:
        train_loss, train_acc = new_test_loss, new_test_acc
        train_time = timer() - start

        # Compute test accuracy
        test_loss, test_acc = test(args, model, device, test_loader)
        test_loss_pp, test_acc_pp = test_loss, test_acc

        if best_model_state is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()

        # Get n params before pruning
        # Otherwise test loss does not match the network
        flops, n_params = get_parameter_count_and_flops(model, input_size, device="cuda" if use_cuda else "cpu")
        layer_size = get_layer_sizes(model)

        # Prune and rebuild optimizer
        start = timer()
        prune_time = 0.0

        if pruning_count < pruning_steps and sparsity < 1.0:
            if (epoch-pruning_start) % pruning_interval == 0:
                opt_state = pruner.prune(
                    # pow(sparsity, 1/pruning_steps),
                    sparsity,  # to be used when prune_all_layers is False
                    args.pruning_logic,
                    optimizer,
                    max_loss_increase_percent=max_increment_loss,
                    epoch=pruning_count,
                    prune_all_layers=False
                )
                pruning_count += 1
                optimizer = experiment.get_optimizer_for_model(model, epoch, opt_state)
                prune_time = timer() - start
                test_loss_pp, test_acc_pp = test(args, model, device, test_loader)



        else:
            # Recreate optimizer anyway, because we need to adjust learning rate based on epoch
            opt_state = optimizer.state_dict()
            optimizer = experiment.get_optimizer_for_model(model, epoch, opt_state)

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
