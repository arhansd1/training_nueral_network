"""
task3.py
────────
Task 3: For each hidden neuron, plot a bar chart showing average activation
        per digit class (0–9). Used to determine class-selective vs feature-selective.

Run:
    python task3.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mnist_loader

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def load_model(json_path="model.json"):
    with open(json_path, "r") as f:
        model = json.load(f)
    weights = [np.array(w) for w in model["weights"]]
    biases  = [np.array(b) for b in model["biases"]]
    return weights, biases

def get_hidden_activations(test_data, W1, b1):
    activations, labels = [], []
    for x, y in test_data:
        z1 = W1 @ x + b1
        a1 = sigmoid(z1)
        activations.append(a1.flatten())
        labels.append(int(y))
    return np.array(activations), np.array(labels)

# ════════════════════════════════════════════════════════════════════════════════
# TASK 3 — Average activation per digit class, all 20 neurons in one figure
# ════════════════════════════════════════════════════════════════════════════════

def task3_activation_distribution(activations, labels, save_path="task3_activation_distribution.png"):
    """
    activations : (10000, 20)
    labels      : (10000,) int array

    For each neuron j, compute mean activation grouped by digit class 0-9.
    Plot as bar chart. All 20 neurons in one 4×5 figure.
    """
    n_neurons = activations.shape[1]
    digits    = np.arange(10)

    fig, axes = plt.subplots(4, 5, figsize=(18, 13))
    fig.suptitle("Task 3 — Average Hidden Neuron Activation per Digit Class",
                 fontsize=15, y=1.01)

    for j, ax in enumerate(axes.flat):
        avg_per_class = np.array([
            activations[labels == d, j].mean()
            for d in digits
        ])

        # color bars: highlight the dominant class in red, rest in steelblue
        dominant = np.argmax(avg_per_class)
        colors = ['steelblue'] * 10
        colors[dominant] = 'crimson'

        bars = ax.bar(digits, avg_per_class, color=colors, edgecolor='black', linewidth=0.4)

        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Digit class", fontsize=7)
        ax.set_ylabel("Avg activation", fontsize=7)
        ax.set_title(f"Neuron {j+1}", fontsize=9)
        ax.tick_params(axis='y', labelsize=7)

        # annotate the peak value
        ax.text(dominant, avg_per_class[dominant] + 0.02,
                f"{avg_per_class[dominant]:.2f}",
                ha='center', va='bottom', fontsize=6.5, color='crimson')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════════
# ALSO save individual neuron bar charts (useful for report)
# ════════════════════════════════════════════════════════════════════════════════

def task3_individual_charts(activations, labels):
    n_neurons = activations.shape[1]
    digits    = np.arange(10)

    for j in range(n_neurons):
        avg_per_class = np.array([
            activations[labels == d, j].mean()
            for d in digits
        ])

        dominant = np.argmax(avg_per_class)
        colors = ['steelblue'] * 10
        colors[dominant] = 'crimson'

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(digits, avg_per_class, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Digit class (0–9)")
        ax.set_ylabel("Average activation")
        ax.set_title(f"Neuron {j+1} — Average Activation per Digit Class")

        fname = f"task3_neuron_{j+1:02d}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading model ...")
    weights, biases = load_model("model.json")
    W1, W2 = weights
    b1, b2 = biases

    print("Loading MNIST test data ...")
    _, _, test_data = mnist_loader.load_data_wrapper()

    print("Computing activations ...")
    activations, labels = get_hidden_activations(test_data, W1, b1)
    print(f"  Shape: {activations.shape}")

    print("\n── TASK 3: Combined figure (all 20 neurons) ──")
    task3_activation_distribution(activations, labels)

    print("\n── TASK 3: Individual bar charts (20 files) ──")
    task3_individual_charts(activations, labels)

    print("\nDone!")
    print("  task3_activation_distribution.png  ← main combined figure")
    print("  task3_neuron_01.png ... task3_neuron_20.png  ← individual charts")