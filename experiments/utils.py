from datetime import datetime
import numpy as np
import os
import csv
import torch
import pickle
from thop import profile
import matplotlib.pyplot as plt


current_dir = os.path.dirname(__file__)


def now():
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _save_pruner_state(state, path):
    with open(path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pruner_state(path):
    with open(f"{path}", "rb") as handle:
        state = pickle.load(handle)
    return state


def save_model_state(state, model_name, timestamp):
    if not os.path.exists(f"{current_dir}/weights/"):
        os.makedirs(f"{current_dir}/weights/")
    torch.save(state, f"{current_dir}/weights/{model_name}_{timestamp}.pt")
    # Saving also without timestamp suffix makes it easier to just load the "last"
    torch.save(state, f"{current_dir}/weights/{model_name}.pt")


def load_model_state(model, model_name, timestamp):
    print(f"Loading {model_name}_{timestamp}")
    load_path = model_name
    if isinstance(timestamp, str) and len(timestamp) > 0 and timestamp != "last":
        load_path += f"_{timestamp}"
    print(f"Loading {current_dir}/weights/{load_path}.pt")
    model.load_state_dict(torch.load(f"{current_dir}/weights/{load_path}.pt"))


def log_dict(filename, dict):
    with open(f"{current_dir}/results/{filename}.csv", "a", newline="") as csvfile:
        fieldnames = list(dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(dict)


def get_module_name(model, module):
    for module_name, m in model.named_modules():
        if module == m:
            return module_name
    return None


def get_layer_sizes(model):
    summary = []
    for m, _ in model.get_pruning_graph():
        summary.append(m.weight.shape[0])
    return str(summary).replace(", ", "-").replace("[", "").replace("]", "")


def get_parameter_count_and_flops(model, input_size, device):
    # Notice that, because of BatchNorm, the samples dim must be >= 2
    x = torch.randn((2,) + tuple(input_size))
    x = x.to(device)
    macs, params = profile(model, inputs=(x,))
    print(f"Model with {params} params and {2*macs} flops")
    return 2 * macs, params


class Logger:
    def __init__(self, args, model, model_input_size, device):
        self.now = now()
        self.args = args
        self.device = device
        self.input_size = model_input_size
        self.flops_original, self.n_params_original = get_parameter_count_and_flops(model, self.input_size, self.device)
        self.filename = f"{current_dir}/results/{args.log}.csv"

    def log(self, model, test_loss, test_acc, test_loss_pp, test_acc_pp, prune_time):
        flops, n_params = get_parameter_count_and_flops(model, self.input_size, self.device)
        layers = get_layer_sizes(model)
        with open(self.filename, "a", newline="") as csvfile:
            dict = {
                "timestamp": self.now,
                "epoch": 0,
                "train_acc": 0,
                "test_acc": test_acc,
                "test_acc_pp": test_acc_pp,
                "train_loss": 0,
                "test_loss": test_loss,
                "test_loss_pp": test_loss_pp,
                "n_params": n_params,
                "flops": flops,
                "n_params_full": self.n_params_original,
                "flops_full": self.n_params_original,
                "layers": layers,
                "train_time": 0.,
                "prune_time": prune_time,
                "experiment": self.args.log,
                "pr": self.args.pr
            }
            writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(dict)


COLORS = ["orange",
          "#4e79a7",
          "#59a14f",
          "#9c755f",
          "#666666",
          "#e15759",
          "#b07aa1",
          "#BEAD53",
          "grey"]

METHODS_MAPPING = {

    "SV": ("SV, $\mu+2\sigma$ aggr. (5 samples)", COLORS[5]),
    "Random": ("Random", COLORS[0]),
    "Sensitivity": ("Saliency", COLORS[2]),
    "Taylor": ("Taylor", COLORS[1]),
    "APoZ": ("APoZ", COLORS[8]),
    "Weight Norm": ("$||w||_1$", COLORS[6]),
    # "taylor": ("Taylor (no abs)", COLORS[3]),
    # "sv-loss#10": ("SV, $\mu$ aggr. (5 samples)", "black"),
}


def map_method_vis(method_name):
    return METHODS_MAPPING[method_name]




def format_plt(ax, title, xlabel, ylabel):
    plt.sca(ax)
    plt.box(False)
    plt.tick_params(color="#222222", labelcolor="#222222")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
    if title is not None:
        plt.title(title)

