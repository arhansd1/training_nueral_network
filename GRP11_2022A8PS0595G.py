"""
tasks_1_2.py
────────────
Task 1: Visualize each hidden neuron's weights as a 28×28 heatmap
Task 2: For each neuron, find top-8 images that excite it most

Run:
    python tasks_1_2.py

Requires in same directory:
    network.py, mnist_loader.py, model.json
    + the 4 MNIST .idx files
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mnist_loader

# ── Sigmoid (needed to compute hidden activations) ────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ── Load trained weights from JSON ───────────────────────────────────────────
def load_model(json_path="model.json"):
    with open(json_path, "r") as f:
        model = json.load(f)
    weights = [np.array(w) for w in model["weights"]]  # [W1(20×784), W2(10×20)]
    biases  = [np.array(b) for b in model["biases"]]   # [b1(20×1),  b2(10×1)]
    return weights, biases

# ── Compute hidden layer activations for all test images ─────────────────────
def get_hidden_activations(test_data, W1, b1):
    """
    Returns:
        activations : np.array shape (10000, 20)  — neuron j's activation for each image
        labels      : list of 10000 ints          — true digit label
        images      : list of 10000 (784,1) arrays
    """
    activations = []
    labels = []
    images = []

    for x, y in test_data:
        z1 = W1 @ x + b1          # (20,1)
        a1 = sigmoid(z1)           # (20,1)
        activations.append(a1.flatten())   # shape (20,)
        labels.append(int(y))
        images.append(x)

    return np.array(activations), labels, images   # (10000, 20), list, list


# ════════════════════════════════════════════════════════════════════════════════
# TASK 1 — Weight heatmaps
# ════════════════════════════════════════════════════════════════════════════════

def task1_weight_heatmaps(W1, save=True):
    """
    W1 shape: (20, 784)
    For each of the 20 neurons, reshape its 784 weights → 28×28 and plot.

    Diverging colormap (RdBu_r):
        Red  = positive weight  → pixel excites this neuron
        Blue = negative weight  → pixel suppresses this neuron
        White= near-zero weight → neuron ignores this pixel
    """
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.suptitle("Task 1 — Hidden Neuron Weight Heatmaps\n"
                 "(Red = excitatory, Blue = inhibitory)", fontsize=14, y=1.01)

    for j, ax in enumerate(axes.flat):
        w_j = W1[j].reshape(28, 28)          # the neuron's weight image

        # symmetric colorscale so white = 0
        vmax = np.max(np.abs(w_j))

        im = ax.imshow(w_j, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"Neuron {j+1}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        plt.savefig("task1_heatmaps.png", dpi=150, bbox_inches='tight')
        print("Saved: task1_heatmaps.png")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
# TASK 2 — Top-8 activating images per neuron
# ════════════════════════════════════════════════════════════════════════════════

def task2_top_activating(activations, labels, images, save=True):
    """
    activations : (10000, 20)
    For each neuron j, sort all 10000 images by activation[j] descending,
    pick top 8, display them in a 2×4 grid with true label + activation value.
    """
    n_neurons = activations.shape[1]

    for j in range(n_neurons):
        neuron_acts = activations[:, j]                      # (10000,)
        top8_idx    = np.argsort(neuron_acts)[-8:][::-1]    # indices of top 8

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"Task 2 — Neuron {j+1}: Top-8 Activating Images",
                     fontsize=13)

        for rank, idx in enumerate(top8_idx):
            ax  = axes[rank // 4][rank % 4]
            img = images[idx].reshape(28, 28)

            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {labels[idx]}\nAct: {neuron_acts[idx]:.4f}",
                         fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        if save:
            fname = f"task2_neuron_{j+1:02d}.png"
            plt.savefig(fname, dpi=120, bbox_inches='tight')
            print(f"Saved: {fname}")
        plt.show()
        plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load model
    print("Loading model weights from model.json ...")
    weights, biases = load_model("model.json")
    W1, W2 = weights   # W1: (20,784)   W2: (10,20)
    b1, b2 = biases    # b1: (20,1)     b2: (10,1)

    # 2. Load test data
    print("Loading MNIST test data ...")
    _, _, test_data = mnist_loader.load_data_wrapper()

    # 3. Compute hidden activations for all 10,000 test images
    print("Computing hidden layer activations ...")
    activations, labels, images = get_hidden_activations(test_data, W1, b1)
    print(f"  activations matrix shape: {activations.shape}")  # (10000, 20)

    # 4. Task 1
    print("\n── TASK 1: Weight Heatmaps ──")
    task1_weight_heatmaps(W1, save=True)

    # 5. Task 2
    print("\n── TASK 2: Top-8 Activating Images per Neuron ──")
    task2_top_activating(activations, labels, images, save=True)

    print("\nDone! Check task1_heatmaps.png and task2_neuron_XX.png files.")