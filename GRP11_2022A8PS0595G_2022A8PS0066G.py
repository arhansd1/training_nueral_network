"""
GRP11_2022A8PS0595G_2022A8PS0066G.py
Assignment 1 — Understanding Hidden Layer Representations (MNIST)
BITS F312 Neural Networks

Architecture : [784 → 20 → 10]
Trained with stochastic gradient descent (SGD) -  30 epochs, mini-batch=10, eta=3.0
Model loaded from: GRP11_2022A8PS0595G_2022A8PS0066G.json (no retraining)


First get these files in the same folder (pre-requisites)
    mnist_loader.py - (from NN deep learning repo)
    GRP11_2022A8PS0595G_2022A8PS0066G.json      - (pretrained model weights)
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte

then run 
python GRP11_2022A8PS0595G_2022A8PS0066G.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mnist_loader

favourite_nueron = 6       


NEURON_NAMES = {
    1:  "Left legged",
    2:  "Central line",
    3:  "Bindhi",
    4:  "Donut",
    5:  "Square",
    6:  "Hockey stick",
    7:  "Thick donut",
    8:  "Scramble",
    9:  "Grandpa stick",
    10: "Right legged",
    11: "Captain Hook",
    12: "The E",
    13: "Hangman noose",
    14: "Stroked 7",
    15: "Equator",
    16: "Black hole",
    17: "Scrambled",
    18: "Bottom loop",
    19: "Seahorse",
    20: "The Sun",
}



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(json_path="GRP11_2022A8PS0595G_2022A8PS0066G.json"):
    with open(json_path, "r") as f:
        model = json.load(f)
    weights = [np.array(w) for w in model["weights"]]
    biases  = [np.array(b) for b in model["biases"]]
    return weights, biases


def get_hidden_activations(test_data, W1, b1):
    activations, labels, images = [], [], []
    for x, y in test_data:
        z1 = W1 @ x + b1
        a1 = sigmoid(z1)
        activations.append(a1.flatten())
        labels.append(int(y))
        images.append(x)
    return np.array(activations), np.array(labels), images


# TASK 1 - Visualise each hidden neurons weight in HeatMap.

class Task1:
 

    @staticmethod
    def run_all(W1, save_path="task1_heatmaps.png"):
        fig, axes = plt.subplots(4, 5, figsize=(14, 11))
        fig.suptitle("Task 1 — Hidden Neuron Weight Heatmaps\n"
                     "(Red = excitatory, Blue = inhibitory)", fontsize=14, y=1.01)
        for j, ax in enumerate(axes.flat):
            w_j  = W1[j].reshape(28, 28)
            vmax = np.max(np.abs(w_j))
            im   = ax.imshow(w_j, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f"Neuron {j+1}", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Task 1] Saved: {save_path}")

    @staticmethod
    def run_single(W1, neuron_idx, save_path=None):
        j    = neuron_idx - 1           # change  1-indexed → 0-indexed if needed
        w_j  = W1[j].reshape(28, 28)
        vmax = np.max(np.abs(w_j))

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(w_j, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"Neuron {neuron_idx} Weight Heatmap\n"
                     f"({NEURON_NAMES[neuron_idx]})", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        path = save_path or f"task1_neuron_{neuron_idx:02d}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Task 1] Saved: {path}")



# Task 2 — Top-8 activating images.

class Task2:

    @staticmethod
    def run_all(activations, labels, images):
        for j in range(activations.shape[1]):
            Task2._plot_neuron(j + 1, activations, labels, images)

    @staticmethod
    def run_single(neuron_idx, activations, labels, images, save_path=None):
        Task2._plot_neuron(neuron_idx, activations, labels, images, save_path)

    @staticmethod
    def _plot_neuron(neuron_idx, activations, labels, images, save_path=None):
        j           = neuron_idx - 1
        neuron_acts = activations[:, j]
        top8_idx    = np.argsort(neuron_acts)[-8:][::-1]

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"Task 2 — Neuron {neuron_idx}: Top-8 Activating Images\n"
                     f"({NEURON_NAMES[neuron_idx]})", fontsize=12)

        for rank, idx in enumerate(top8_idx):
            ax  = axes[rank // 4][rank % 4]
            img = images[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {labels[idx]}\nAct: {neuron_acts[idx]:.4f}",
                         fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        path = save_path or f"task2_neuron_{neuron_idx:02d}.png"
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [Task 2] Saved: {path}")


# Task 3 — Activation distribution per digit class.

class Task3:

    @staticmethod
    def run_all(activations, labels, save_path="task3_activation_distribution.png"):
        digits = np.arange(10)
        fig, axes = plt.subplots(4, 5, figsize=(18, 13))
        fig.suptitle("Task 3 — Average Hidden Neuron Activation per Digit Class",
                     fontsize=15, y=1.01)

        for j, ax in enumerate(axes.flat):
            avg = np.array([activations[labels == d, j].mean() for d in digits])
            dominant = np.argmax(avg)
            colors = ['steelblue'] * 10
            colors[dominant] = 'crimson'
            ax.bar(digits, avg, color=colors, edgecolor='black', linewidth=0.4)
            ax.set_xticks(digits)
            ax.set_xticklabels([str(d) for d in digits], fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Digit class", fontsize=7)
            ax.set_ylabel("Avg activation", fontsize=7)
            ax.set_title(f"Neuron {j+1}", fontsize=9)
            ax.tick_params(axis='y', labelsize=7)
            ax.text(dominant, avg[dominant] + 0.02,
                    f"{avg[dominant]:.2f}",
                    ha='center', va='bottom', fontsize=6.5, color='crimson')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Task 3] Saved: {save_path}")

    @staticmethod
    def run_single(neuron_idx, activations, labels, save_path=None):
        j      = neuron_idx - 1
        digits = np.arange(10)
        avg    = np.array([activations[labels == d, j].mean() for d in digits])

        dominant = np.argmax(avg)
        colors = ['steelblue'] * 10
        colors[dominant] = 'crimson'

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(digits, avg, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Digit class (0–9)")
        ax.set_ylabel("Average activation")
        ax.set_title(f"Neuron {neuron_idx} — Average Activation per Digit Class\n"
                     f"({NEURON_NAMES[neuron_idx]})")
        ax.text(dominant, avg[dominant] + 0.02,
                f"{avg[dominant]:.2f}",
                ha='center', va='bottom', fontsize=9, color='crimson')

        plt.tight_layout()
        path = save_path or f"task3_neuron_{neuron_idx:02d}.png"
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [Task 3] Saved: {path}")

# main() — runs ONLY the favorite neuron. Our selected nueron is 6


def main():
    n = favourite_nueron
    print(f"\n{'='*55}")
    print(f"  Favorite Neuron : {n}  —  {NEURON_NAMES[n]}")
    print(f"{'='*55}\n")

    print("Loading model")
    weights, biases = load_model("GRP11_2022A8PS0595G_2022A8PS0066G.json")
    W1 = weights
    b1 = biases

    _, _, test_data = mnist_loader.load_data_wrapper()

    activations, labels, images = get_hidden_activations(test_data, W1, b1)

    print(f"\n─> Task 1: Weight heatmap for Neuron {n} ──")
    Task1.run_single(W1, n)

    print(f"\n─> Task 2: Top-8 activating images for Neuron {n} ──")
    Task2.run_single(n, activations, labels, images)

    print(f"\n─> Task 3: Activation distribution for Neuron {n} ──")
    Task3.run_single(n, activations, labels)

    print(f"\n Output files:")
    print(f"  task1_neuron_{n:02d}.png")
    print(f"  task2_neuron_{n:02d}.png")
    print(f"  task3_neuron_{n:02d}.png")


if __name__ == "__main__":
    main()