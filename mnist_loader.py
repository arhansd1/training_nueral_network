"""
mnist_loader.py  — Python 3 compatible version
Reads raw MNIST .idx binary files directly (no pkl.gz needed).

Expected files in the SAME directory as this script:
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
"""

import struct
import numpy as np
import os

# ── locate files relative to this script ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(_HERE, filename)

# ── low-level readers ──────────────────────────────────────────────────────────
def _read_images(filepath):
    with open(filepath, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Bad magic number in {filepath}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0  # normalise

def _read_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Bad magic number in {filepath}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)

# ── public API (mirrors Nielsen's load_data_wrapper) ───────────────────────────
def load_data():
    """
    Returns (training_data, validation_data, test_data).
    Each is a tuple (images_array, labels_array).
    training:   50 000 images  (first 50k of the 60k training set)
    validation: 10 000 images  (last  10k of the 60k training set)
    test:       10 000 images
    """
    train_images = _read_images(_path('archive/train-images.idx3-ubyte'))
    train_labels = _read_labels(_path('archive/train-labels.idx1-ubyte'))
    test_images  = _read_images(_path('archive/t10k-images.idx3-ubyte'))
    test_labels  = _read_labels(_path('archive/t10k-labels.idx1-ubyte'))

    # split training into 50k train + 10k validation
    training_data   = (train_images[:50000], train_labels[:50000])
    validation_data = (train_images[50000:], train_labels[50000:])
    test_data       = (test_images, test_labels)
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Convenience wrapper used by network.py.

    training_data  : list of 50 000 tuples (x, y)
        x : (784,1) ndarray  — pixel values in [0,1]
        y : (10,1)  ndarray  — one-hot vector

    validation_data: list of 10 000 tuples (x, y)
    test_data      : list of 10 000 tuples (x, y)
        x : (784,1) ndarray
        y : int              — digit label 0-9
    """
    tr_d, va_d, te_d = load_data()

    # --- training ---
    training_inputs  = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [_vectorized(y) for y in tr_d[1]]
    training_data    = list(zip(training_inputs, training_results))

    # --- validation ---
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data   = list(zip(validation_inputs, va_d[1]))

    # --- test ---
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data   = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


def _vectorized(j):
    """One-hot encode digit j into a (10,1) vector."""
    e = np.zeros((10, 1))
    e[int(j)] = 1.0
    return e