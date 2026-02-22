"""
train_and_save.py
─────────────────
Run this ONCE to train the network and save weights to model.json.

Usage:
    python train_and_save.py

Requires in the same directory:
    network.py
    mnist_loader.py
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
"""

import json
import numpy as np
import network
import mnist_loader

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(f"  Training samples  : {len(training_data)}")
print(f"  Validation samples: {len(validation_data)}")
print(f"  Test samples      : {len(test_data)}")

# ── 2. Build network [784 → 20 → 10] ──────────────────────────────────────────
net = network.Network([784, 20, 10])

# ── 3. Train ───────────────────────────────────────────────────────────────────
# These hyper-parameters reliably reach >94% in ~20 epochs
print("\nTraining network [784, 20, 10]...")
net.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)

# ── 4. Final accuracy ──────────────────────────────────────────────────────────
accuracy = net.evaluate(test_data)
print(f"\nFinal test accuracy: {accuracy} / {len(test_data)}  ({100*accuracy/len(test_data):.2f}%)")

# ── 5. Save weights & biases to JSON ──────────────────────────────────────────
OUTPUT_FILE = "model.json"

model_params = {
    "sizes":   net.sizes,
    "weights": [w.tolist() for w in net.weights],   # list of 2D lists
    "biases":  [b.tolist() for b in net.biases]     # list of 2D lists
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(model_params, f)

print(f"\nModel saved to '{OUTPUT_FILE}'")
print("You can now run your main assignment file without retraining.")