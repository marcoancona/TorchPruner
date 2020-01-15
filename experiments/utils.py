from datetime import datetime
import numpy as np
import os
import csv

current_dir = os.path.dirname(__file__)


def now():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


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

