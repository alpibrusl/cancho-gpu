#!/usr/bin/env python3
"""Image data for the CNN of docs/resident.md section 18: a synthetic set (class-conditional textures plus noise; no download) and a reader for CIFAR-10's python batches
(`cifar-10-batches-py/`, from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) for when that archive is on the machine.

The synthetic classes are 10 textures of 32 x 32 x 3: each class is two sinusoidal gratings (its own orientations and spatial frequencies) with its own colour mix, and every sample
has a random phase, a random amplitude, a random shift of the colour and Gaussian noise, so that the label is a property of the pattern and not of any pixel."""
import os
import pickle

import numpy as np


def synthetic(n, seed, h=32, noise=1.5, classes=10, jitter=0.5, signal=0.6):
    """n images [n, h, h, 3] (float, unit variance overall) and labels [n] ints. Class c is a pair of gratings; a sample's gratings have their orientation and frequency jittered
    (`jitter`: radians and relative frequency), a random phase, a random class-independent grating of the same strength is added as a distractor, and Gaussian noise of `noise`."""
    rs = np.random.RandomState(1234)  # the classes: the same for every call
    freq = rs.uniform(0.08, 0.40, size=(classes, 2))
    theta = rs.uniform(0, np.pi, size=(classes, 2))
    mix = rs.normal(size=(classes, 2, 3))
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, classes, size=n)
    yy, xx = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
    out = np.zeros((n, h, h, 3), dtype=np.float32)

    def grating(th, fr, ph):
        return np.sin((np.cos(th) * xx + np.sin(th) * yy) * 2 * np.pi * fr + ph)

    for i in range(n):
        c = labels[i]
        img = np.zeros((h, h, 3))
        for k in range(2):
            g = grating(theta[c, k] + rng.normal(0, jitter), freq[c, k] * (1 + rng.normal(0, 0.15 * jitter)), rng.uniform(0, 2 * np.pi))
            img += signal * rng.uniform(0.7, 1.3) * g[:, :, None] * mix[c, k][None, None, :]
        g = grating(rng.uniform(0, np.pi), rng.uniform(0.08, 0.40), rng.uniform(0, 2 * np.pi))  # the distractor
        img += g[:, :, None] * rng.normal(size=3)[None, None, :]
        img += rng.normal(0, 0.3, size=3)[None, None, :]
        img += rng.normal(0, noise, size=img.shape)
        out[i] = img
    out /= out.std()
    return out, labels


def cifar(directory):
    """(train images [50000, 32, 32, 3], labels, test images [10000, ...], labels) from cifar-10-batches-py, normalised per channel by the training set's mean and deviation."""
    def load(name):
        with open(os.path.join(directory, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32), np.array(d[b"labels"])
    parts = [load(f"data_batch_{i}") for i in range(1, 6)]
    xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    xte, yte = load("test_batch")
    mean, std = xtr.mean((0, 1, 2)), xtr.std((0, 1, 2))
    return (xtr - mean) / std, ytr, (xte - mean) / std, yte


def batches(images, labels, batch, nsteps, seed):
    """[nsteps, batch, ...] images and labels: shuffled epochs one after another (the last incomplete batch of each epoch dropped)."""
    rng = np.random.RandomState(seed)
    per = len(images) // batch
    idx = np.concatenate([rng.permutation(len(images))[: per * batch] for _ in range(-(-nsteps // per))])[: nsteps * batch]
    idx = idx.reshape(nsteps, batch)
    return images[idx], labels[idx]
