from __future__ import annotations

import numpy as np

WORD = np.uint64(0xFFFFFFFF)

ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))

ERFINV_TAIL = (3.03697567e-10, 2.93243101e-8, 1.22150334e-6, 2.84108955e-5, 3.93552968e-4,
               3.02698812e-3, 4.83185798e-3, -2.64646143e-1, 8.40016484e-1)

ERFINV_BODY = (5.43877832e-9, 1.43285448e-7, 1.22774793e-6, 1.12963626e-7, -5.61530760e-5,
               -1.47697632e-4, 2.31468678e-3, 1.15392581e-2, -2.32015476e-1, 8.86226892e-1)


def make_key(seed: int) -> np.ndarray:
    return np.array([(int(seed) >> 32) & 0xFFFFFFFF, int(seed) & 0xFFFFFFFF], dtype=np.uint32)


def rotate_left(x: np.ndarray, r: int) -> np.ndarray:
    return ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & WORD


# external: MLX random.cpp — threefry2x32 over a counter pair, the first half of the words from the first lane and the second half from the second
def hash_threefry(key: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k0, k1 = np.uint64(key[0]), np.uint64(key[1])
    ks = (k0, k1, (k0 ^ k1 ^ np.uint64(0x1BD11BDA)) & WORD)
    x0 = (first.astype(np.uint64) + ks[0]) & WORD
    x1 = (second.astype(np.uint64) + ks[1]) & WORD
    for i in range(5):
        for r in ROTATIONS[i % 2]:
            x0 = (x0 + x1) & WORD
            x1 = rotate_left(x1, r) ^ x0
        x0 = (x0 + ks[(i + 1) % 3]) & WORD
        x1 = (x1 + ks[(i + 2) % 3] + np.uint64(i + 1)) & WORD
    return x0.astype(np.uint32), x1.astype(np.uint32)


def draw_bits(key: np.ndarray, n: int) -> np.ndarray:
    half, odd = n // 2, n % 2
    out = np.zeros(n, dtype=np.uint32)
    counter = np.arange(half, dtype=np.uint64)
    a, b = hash_threefry(key, counter, counter + np.uint64(half + odd))
    out[:half] = a
    out[half + odd:] = b
    if odd:
        out[half] = hash_threefry(key, np.array([half], dtype=np.uint64),
                                  np.array([0], dtype=np.uint64))[0][0]
    return out


def split_key(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = draw_bits(key, 4).reshape(2, 2)
    return pair[0], pair[1]


def fuse_multiply_add(a: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    return (a.astype(np.float64) * b.astype(np.float64) + np.float64(np.float32(c))).astype(np.float32)


def apply_polynomial(coefficients: tuple[float, ...], t: np.ndarray) -> np.ndarray:
    p = np.full_like(t, np.float32(coefficients[0]))
    for c in coefficients[1:]:
        p = fuse_multiply_add(p, t, c)
    return p


def compute_erfinv(a: np.ndarray) -> np.ndarray:
    t = np.log(fuse_multiply_add(a, -a, 1.0).astype(np.float64)).astype(np.float32)
    p = np.where(np.abs(t) > np.float32(6.125), apply_polynomial(ERFINV_TAIL, t),
                 apply_polynomial(ERFINV_BODY, t))
    return (a * p).astype(np.float32)


def draw_normal(shape: tuple[int, ...], key: np.ndarray) -> np.ndarray:
    n = int(np.prod(shape))
    unit = draw_bits(key, n).astype(np.float32) / np.float32(4294967295.0)
    unit = np.minimum(unit, np.nextafter(np.float32(1), np.float32(0)))
    low = np.nextafter(np.float32(-1), np.float32(0))
    uniform = ((np.float32(1) - low) * unit + low).astype(np.float32)
    return (np.float32(np.sqrt(2.0)) * compute_erfinv(uniform)).astype(np.float32).reshape(shape)
