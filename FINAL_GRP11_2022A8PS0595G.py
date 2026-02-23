"""
GRP11_2022A8PS0595G.py
══════════════════════════════════════════════════════════════════════
Assignment 1 — Understanding Hidden Layer Representations (MNIST)
BITS F312 Neural Networks

Architecture : [784 → 20 → 10]
Trained with : SGD, 30 epochs, mini-batch=10, eta=3.0
Model loaded from: model.json (no retraining)

Usage:
    python GRP11_2022A8PS0595G.py

Requires in the same directory:
    mnist_loader.py
    model.json
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mnist_loader

# ══════════════════════════════════════════════════════════════════════
# ▶▶  SET YOUR FAVORITE NEURON HERE (1 to 20)
FAVORITE_NEURON = 14        # ← change this to whichever neuron you like
# ══════════════════════════════════════════════════════════════════════

# Informal names for all 20 neurons (based on Task 1 + Task 2 analysis)
NEURON_NAMES = {
    1:  "Curved multi-stroke detector",
    2:  "Vertical stroke detector",
    3:  "Open arc / tall shape detector",
    4:  "Closed oval loop detector",
    5:  "Cross-junction detector",
    6:  "Diagonal slash detector",
    7:  "Round loop detector",
    8:  "S-curve / swirl detector",
    9:  "Seven detector (variant A)",
    10: "Round + looping shape detector",
    11: "Two detector",
    12: "Double-arc detector",
    13: "Bottom-loop + sweep detector",
    14: "Seven detector (variant B)",
    15: "Broad mixed-stroke detector",
    16: "Closed oval detector",
    17: "S/Z-shape detector",
    18: "Six detector",
    19: "Five detector",
    20: "Slanted multi-curve detector",
}


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(json_path="model.json"):
    with open(json_path, "r") as f:
        model = json.load(f)
    weights = [np.array(w) for w in model["weights"]]
    biases  = [np.array(b) for b in model["biases"]]
    return weights, biases


def get_hidden_activations(test_data, W1, b1):
    """Returns activations (N,20), labels (N,), images list."""
    activations, labels, images = [], [], []
    for x, y in test_data:
        z1 = W1 @ x + b1
        a1 = sigmoid(z1)
        activations.append(a1.flatten())
        labels.append(int(y))
        images.append(x)
    return np.array(activations), np.array(labels), images


# ══════════════════════════════════════════════════════════════════════
# Task 1 — Weight heatmaps
# ══════════════════════════════════════════════════════════════════════

class Task1:
    """Visualise each hidden neuron's 784 weights as a 28×28 heatmap."""

    @staticmethod
    def run_all(W1, save_path="task1_heatmaps.png"):
        """Plot all 20 neurons in a 4×5 grid and save."""
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
        """Plot heatmap for one neuron (0-indexed internally)."""
        j    = neuron_idx - 1           # convert 1-indexed → 0-indexed
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


# ══════════════════════════════════════════════════════════════════════
# Task 2 — Top-8 activating images
# ══════════════════════════════════════════════════════════════════════

class Task2:
    """Find and display the 8 images that most strongly activate each neuron."""

    @staticmethod
    def run_all(activations, labels, images):
        """Save one 2×4 grid per neuron (20 files total)."""
        for j in range(activations.shape[1]):
            Task2._plot_neuron(j + 1, activations, labels, images)

    @staticmethod
    def run_single(neuron_idx, activations, labels, images, save_path=None):
        """Plot top-8 grid for one neuron."""
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


# ══════════════════════════════════════════════════════════════════════
# Task 3 — Activation distribution per digit class
# ══════════════════════════════════════════════════════════════════════

class Task3:
    """Bar chart of average neuron activation for each digit class 0–9."""

    @staticmethod
    def run_all(activations, labels, save_path="task3_activation_distribution.png"):
        """Save combined 4×5 figure for all 20 neurons."""
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
        """Bar chart for one neuron."""
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


# ══════════════════════════════════════════════════════════════════════
# main() — runs ONLY the favorite neuron, completes in < 10 seconds
# ══════════════════════════════════════════════════════════════════════

def main():
    n = FAVORITE_NEURON
    print(f"\n{'='*55}")
    print(f"  Favorite Neuron : {n}  —  {NEURON_NAMES[n]}")
    print(f"{'='*55}\n")

    # 1. Load model (no retraining)
    print("Loading model from model.json ...")
    weights, biases = load_model("model.json")
    W1, W2 = weights
    b1, b2 = biases

    # 2. Load test data
    print("Loading MNIST test data ...")
    _, _, test_data = mnist_loader.load_data_wrapper()

    # 3. Compute hidden activations
    print("Computing hidden layer activations ...")
    activations, labels, images = get_hidden_activations(test_data, W1, b1)

    # 4. Task 1 — weight heatmap for favorite neuron
    print(f"\n── Task 1: Weight heatmap for Neuron {n} ──")
    Task1.run_single(W1, n)

    # 5. Task 2 — top-8 activating images for favorite neuron
    print(f"\n── Task 2: Top-8 activating images for Neuron {n} ──")
    Task2.run_single(n, activations, labels, images)

    # 6. Task 3 — activation bar chart for favorite neuron
    print(f"\n── Task 3: Activation distribution for Neuron {n} ──")
    Task3.run_single(n, activations, labels)

    print(f"\n✓ Done! Output files:")
    print(f"  task1_neuron_{n:02d}.png")
    print(f"  task2_neuron_{n:02d}.png")
    print(f"  task3_neuron_{n:02d}.png")


if __name__ == "__main__":
    main()