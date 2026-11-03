"""Synthetic generator for round-8 tests and the K0 coverage check (no real data). Classes are i.i.d. draws of
(text, class mean); images = class mean + isotropic noise (+ an optional constant offset in the null block)."""
import numpy as np


def texts(rng, K, q):
    """Unit-norm texts with a 1/j spectrum, then centred across classes (as the real CLIP texts are in run_r8)."""
    T = rng.standard_normal((K, q)) * np.sqrt(1.0 / np.arange(1, q + 1))
    T /= np.linalg.norm(T, axis=1, keepdims=True); return T - T.mean(0)


def synth(seed, K=1000, n=10, k=48, r=64, q=768, sz=0.3, ss=1.0, sb=0.5, sw=1.0, offset=0.0):
    """Returns Z (null block, N x k), S (row block, N x r), y, T. sz / ss: per-class rms of the text-predictable part of the
    null / row class means; sb: rms of a text-unrelated class part; sw: rms of within-class noise; offset: norm of a constant
    null vector added to every image."""
    rng = np.random.default_rng(seed); T = texts(rng, K, q); y = np.repeat(np.arange(K), n)
    def block(p, s):
        G = rng.standard_normal((p, q)); tp = T @ G.T; tp *= 1.0 / np.sqrt(np.mean(np.sum(tp ** 2, 1)))
        e = rng.standard_normal((K, p)) / np.sqrt(p)
        return s * tp + sb * e
    mz, ms = block(k, sz), block(r, ss)
    Z = mz[y] + sw * rng.standard_normal((K * n, k)) / np.sqrt(k)
    S = ms[y] + sw * rng.standard_normal((K * n, r)) / np.sqrt(r)
    if offset: v = rng.standard_normal(k); Z = Z + offset * v / np.linalg.norm(v)
    return Z, S, y, T
