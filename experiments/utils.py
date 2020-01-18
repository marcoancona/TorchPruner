from datetime import datetime
import numpy as np
import os
import csv
import torch
import pickle

current_dir = os.path.dirname(__file__)


def now():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _save_pruner_state(state, path):
    with open(path, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pruner_state(path):
    with open(f"{path}", 'rb') as handle:
        state = pickle.load(handle)
        print (state)
    return state


def save_model_state(state, model_name, timestamp, pruner_state = None):
    if not os.path.exists(f"{current_dir}/weights/"):
        os.makedirs(f"{current_dir}/weights/")
    torch.save(state, f"{current_dir}/weights/{model_name}_{timestamp}.pt")
    # Saving also without timestamp suffix makes it easier to just load the "last"
    torch.save(state, f"{current_dir}/weights/{model_name}.pt")
    if pruner_state is not None:
        _save_pruner_state(pruner_state, f"{current_dir}/weights/{model_name}_{timestamp}.pruner")
        _save_pruner_state(pruner_state, f"{current_dir}/weights/{model_name}.pruner")


def load_model_state(model, model_name, timestamp, pruner=None):
    print (f"Loading {model_name}_{timestamp}")
    load_path = model_name
    if isinstance(timestamp, str) and len(timestamp) > 0 and timestamp != "last":
        load_path += f"_{timestamp}"
    model.load_state_dict(torch.load(f"{current_dir}/weights/{load_path}.pt"))
    pruner.load_state_dict(_load_pruner_state(f"{current_dir}/weights/{load_path}.pruner"))


def log_dict(filename, dict):
    with open(f"{current_dir}/results/{filename}.csv", "a", newline="") as csvfile:
        fieldnames = list(dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(dict)


def log(
    experiment_name,
    timestamp_id,
    epoch,
    train_acc,
    test_acc,
    train_loss,
    test_loss,
    n_params,
    n_params_full,
    activations,
    train_time,
    prune_time,
):
    with open(f"{current_dir}/results/log.csv", "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "epoch",
            "train_acc",
            "test_acc",
            "train_loss",
            "test_loss",
            "n_params",
            "n_params_full",
            "activations",
            "train_time",
            "prune_time",
            "experiment",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp_id,
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "n_params": n_params,
                "n_params_full": n_params_full,
                "activations": activations,
                "train_time": train_time,
                "prune_time": prune_time,
                "experiment": experiment_name,
            }
        )


def get_layer_sizes(model):
    summary = []
    for m, _ in model.get_pruning_graph():
        summary.append(m.weight.shape[0])
    return str(summary).replace(", ", "-").replace("[", "").replace("]", "")


def get_parameter_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

