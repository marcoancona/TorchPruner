from __future__ import print_function
from timeit import default_timer as timer
import torch

"""
This file contains standard training and test logic
for a PyTorch model
"""


def train(model, device, loss, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    samples = 0
    cumulative_loss = 0
    start = timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        curr_loss = loss(output, target)
        cumulative_loss += curr_loss.item() * len(target)

        current_pred = output.argmax(dim=1, keepdim=True)
        correct += current_pred.eq(target.view_as(current_pred)).sum().item()
        samples += len(target)

        # Perform optimization step
        curr_loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {:.3f}\t Time: {:.1f}s".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    curr_loss.item(),
                    correct / samples,
                    timer() - start,
                )
            )
            start = timer()

    return cumulative_loss / samples, correct / samples


def test(model, device, loss, test_loader, verbose=1):
    model.eval()
    cum_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            cum_test_loss += loss(output, target).item() * len(target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_accuracy = correct / len(test_loader.dataset)
        avg_test_loss = cum_test_loss / len(test_loader.dataset)

        if verbose > 0:
            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n".format(
                    avg_test_loss, correct, len(test_loader.dataset), 100.0 * test_accuracy
                )
            )
        return avg_test_loss, test_accuracy
