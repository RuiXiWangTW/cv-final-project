import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log(filepath):
    train_acc = []
    val_acc_by_resolution = defaultdict(list)
    epoch = -1

    with open(filepath, 'r') as f:
        for line in f:
            # Parse training line
            train_match = re.search(r'Train Epoch: (\d+), Loss: ([\d.]+), Acc: ([\d.]+)', line)
            if train_match:
                epoch = int(train_match.group(1))
                acc = float(train_match.group(3))
                train_acc.append(acc)

            # Parse validation line
            val_match = re.search(r'Val Epoch: (\d+), Resolution: (\d+), Loss: ([\d.]+), Acc: ([\d.]+)', line)
            if val_match:
                val_epoch = int(val_match.group(1))
                resolution = int(val_match.group(2))
                acc = float(val_match.group(4))
                val_acc_by_resolution[resolution].append(acc)

    return train_acc, val_acc_by_resolution

def plot_training_curves(log_files, labels, seen=False):
    plt.figure(figsize=(12, 6))

    for filepath, label in zip(log_files, labels):
        train_acc, val_acc_by_res = parse_log(filepath)
        epochs = list(range(len(train_acc)))

        # Plot training accuracy
        plt.plot(epochs, train_acc, label=f'{label} Train', linewidth=2, linestyle='--')

        # Plot validation accuracy per resolution
        for res, accs in val_acc_by_res.items():
            if not seen:
                if res == 192 or res == 256:
                    plt.plot(epochs, accs, label=f'{label} Val (Res {res})')
            else:
                if res == 64 or res == 128 or res == 160:
                    plt.plot(epochs, accs, label=f'{label} Val (Res {res})')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    title = 'Training & Validation Accuracy vs. Epoch'
    if seen:
        title += " (seen)"
    else:
        title += " (unseen)"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if seen:
        plt.savefig("seen_res_curve.png")
    else:
        plt.savefig("unseen_res_curve.png")
    # plt.show()

# Usage
log_files = ['normal.log', 'SPP.log']
labels = ['Normal', 'SPP']

plot_training_curves(log_files, labels, seen=False)
plot_training_curves(log_files, labels, seen=True)