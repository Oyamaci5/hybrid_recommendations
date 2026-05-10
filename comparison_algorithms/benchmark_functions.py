"""
TWOA-style 23 classical benchmark functions (F1–F23).
Each factory returns dict: func, lb, ub, dim, fmin, name, type, key (F1..F23).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np

# --- Hartman 3D (F19) — SFU / Surjanovic ------------------------------------
_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array(
    [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ]
)
_H3_P = 1e-4 * np.array(
    [
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ]
)

# --- Hartman 6D (F20) — SFU hart6m (classic -sum alpha*exp(-inner)) --------
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_H6_P = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)

# --- Shekel foxholes F14: 5x5 grid in [-32,32] ----------------------------
_F14_A = np.array(
    [[-32.0 + 16.0 * jc, -32.0 + 16.0 * jr] for jr in range(5) for jc in range(5)],
    dtype=float,
)

# --- Kowalik F15 -----------------------------------------------------------
_F15_A = np.array(
    [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0863, 0.0574, 0.109, 0.0768, 0.0563, 0.0348]
)
_F15_B = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])

# --- Shekel F21–F23 (SFU shekelm) -----------------------------------------
_SHEKEL_C = np.array(
    [
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
    ],
    dtype=float,
)
_SHEKEL_BETA = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=float)


def _u_penalty(x: np.ndarray, a: float, k: int, m: int) -> float:
    """Penalty u(x, a, k, m) for F12/F13."""
    s = 0.0
    for xi in x:
        if xi > a:
            s += k * (xi - a) ** m
        elif xi < -a:
            s += k * (-xi - a) ** m
    return s


def get_all_benchmarks(f7_noise: float = 0.0) -> List[Dict[str, Any]]:
    """
    Return 23 benchmark dicts. f7_noise: additive constant for noisy Quartic (F7), fixed per run.
    """
    benchmarks: List[Dict[str, Any]] = []

    # F1 Sphere
    def f1(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    benchmarks.append(
        {
            "key": "F1",
            "name": "F1 (Sphere)",
            "type": "unimodal",
            "dim": 30,
            "lb": [-100.0] * 30,
            "ub": [100.0] * 30,
            "fmin": 0.0,
            "func": f1,
        }
    )

    # F2
    def f2(x):
        x = np.abs(np.asarray(x, dtype=float))
        return float(np.sum(x) + np.prod(x))

    benchmarks.append(
        {
            "key": "F2",
            "name": "F2",
            "type": "unimodal",
            "dim": 30,
            "lb": [-10.0] * 30,
            "ub": [10.0] * 30,
            "fmin": 0.0,
            "func": f2,
        }
    )

    # F3 high-conditioned elliptic (nested sums)
    def f3(x):
        x = np.asarray(x, dtype=float)
        d = len(x)
        s = 0.0
        for i in range(d):
            inner = float(np.sum(x[: i + 1]))
            s += inner * inner
        return s

    benchmarks.append(
        {
            "key": "F3",
            "name": "F3",
            "type": "unimodal",
            "dim": 30,
            "lb": [-100.0] * 30,
            "ub": [100.0] * 30,
            "fmin": 0.0,
            "func": f3,
        }
    )

    # F4
    def f4(x):
        x = np.asarray(x, dtype=float)
        return float(np.max(np.abs(x)))

    benchmarks.append(
        {
            "key": "F4",
            "name": "F4",
            "type": "unimodal",
            "dim": 30,
            "lb": [-100.0] * 30,
            "ub": [100.0] * 30,
            "fmin": 0.0,
            "func": f4,
        }
    )

    # F5 Rosenbrock
    def f5(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)
        )

    benchmarks.append(
        {
            "key": "F5",
            "name": "F5 (Rosenbrock)",
            "type": "unimodal",
            "dim": 30,
            "lb": [-30.0] * 30,
            "ub": [30.0] * 30,
            "fmin": 0.0,
            "func": f5,
        }
    )

    # F6 Step (Yao): sum( floor(x_i + 0.5)^2 )
    def f6(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(np.floor(x + 0.5) ** 2))

    benchmarks.append(
        {
            "key": "F6",
            "name": "F6 (Step)",
            "type": "unimodal",
            "dim": 30,
            "lb": [-100.0] * 30,
            "ub": [100.0] * 30,
            "fmin": 0.0,
            "func": f6,
        }
    )

    # F7 noisy Quartic
    noise = float(f7_noise)

    def f7(x):
        x = np.asarray(x, dtype=float)
        d = len(x)
        i = np.arange(1, d + 1, dtype=float)
        return float(np.sum(i * (x**4)) + noise)

    benchmarks.append(
        {
            "key": "F7",
            "name": "F7 (Noisy Quartic)",
            "type": "unimodal",
            "dim": 30,
            "lb": [-1.28] * 30,
            "ub": [1.28] * 30,
            "fmin": 0.0,
            "func": f7,
        }
    )

    # F8 Schwefel 2.26
    def f8(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

    benchmarks.append(
        {
            "key": "F8",
            "name": "F8 (Schwefel)",
            "type": "multimodal",
            "dim": 30,
            "lb": [-500.0] * 30,
            "ub": [500.0] * 30,
            "fmin": 0.0,
            "func": f8,
        }
    )

    # F9 Rastrigin
    def f9(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        return float(np.sum(x * x - 10.0 * np.cos(2 * math.pi * x) + 10.0))

    benchmarks.append(
        {
            "key": "F9",
            "name": "F9 (Rastrigin)",
            "type": "multimodal",
            "dim": 30,
            "lb": [-5.12] * 30,
            "ub": [5.12] * 30,
            "fmin": 0.0,
            "func": f9,
        }
    )

    # F10 Ackley
    def f10(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        s1 = float(np.sum(x * x))
        s2 = float(np.sum(np.cos(2 * math.pi * x)))
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(s1 / n))
            - np.exp(s2 / n)
            + 20.0
            + math.e
        )

    benchmarks.append(
        {
            "key": "F10",
            "name": "F10 (Ackley)",
            "type": "multimodal",
            "dim": 30,
            "lb": [-32.0] * 30,
            "ub": [32.0] * 30,
            "fmin": 0.0,
            "func": f10,
        }
    )

    # F11 Griewank
    def f11(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        idx = np.arange(1, n + 1, dtype=float)
        prod = float(np.prod(np.cos(x / np.sqrt(idx))))
        return float(np.sum(x * x) / 4000.0 - prod + 1.0)

    benchmarks.append(
        {
            "key": "F11",
            "name": "F11 (Griewank)",
            "type": "multimodal",
            "dim": 30,
            "lb": [-600.0] * 30,
            "ub": [600.0] * 30,
            "fmin": 0.0,
            "func": f11,
        }
    )

    # F12
    def f12(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        y = 1.0 + (x + 1.0) / 4.0
        t1 = 10.0 * math.sin(math.pi * y[0]) ** 2
        mid = 0.0
        for i in range(n - 1):
            mid += (y[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * y[i + 1]) ** 2)
        t2 = (y[-1] - 1.0) ** 2
        pen = _u_penalty(x, 10.0, 100, 4)
        return float((math.pi / n) * (t1 + mid + t2) + pen)

    benchmarks.append(
        {
            "key": "F12",
            "name": "F12",
            "type": "multimodal",
            "dim": 30,
            "lb": [-50.0] * 30,
            "ub": [50.0] * 30,
            "fmin": 0.0,
            "func": f12,
        }
    )

    # F13
    def f13(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        t0 = math.sin(3 * math.pi * x[0]) ** 2
        mid = 0.0
        for i in range(n - 1):
            mid += (x[i] - 1.0) ** 2 * (1.0 + math.sin(3 * math.pi * x[i + 1]) ** 2)
        tlast = (x[-1] - 1.0) ** 2 * (1.0 + math.sin(2 * math.pi * x[-1]) ** 2)
        pen = _u_penalty(x, 5.0, 100, 4)
        return float(0.1 * (t0 + mid + tlast) + pen)

    benchmarks.append(
        {
            "key": "F13",
            "name": "F13",
            "type": "multimodal",
            "dim": 30,
            "lb": [-50.0] * 30,
            "ub": [50.0] * 30,
            "fmin": 0.0,
            "func": f13,
        }
    )

    # F14 Shekel foxholes
    def f14(x):
        x = np.asarray(x, dtype=float)
        s = 0.0
        for j in range(25):
            acc = 0.0
            for i in range(2):
                acc += (x[i] - _F14_A[j, i]) ** 6
            s += 1.0 / ((j + 1) + acc)
        return float((1.0 / 500.0 + s) ** (-1))

    benchmarks.append(
        {
            "key": "F14",
            "name": "F14 (Shekel Foxholes)",
            "type": "fixed_multimodal",
            "dim": 2,
            "lb": [-65.0, -65.0],
            "ub": [65.0, 65.0],
            "fmin": 1.0,
            "func": f14,
        }
    )

    # F15 Kowalik
    def f15(x):
        x = np.asarray(x, dtype=float)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        s = 0.0
        for i in range(11):
            bi = _F15_B[i]
            num = x1 * (bi * bi + bi * x2)
            den = bi * bi + bi * x3 + x4
            s += (_F15_A[i] - num / den) ** 2
        return float(s)

    benchmarks.append(
        {
            "key": "F15",
            "name": "F15 (Kowalik)",
            "type": "fixed_multimodal",
            "dim": 4,
            "lb": [-5.0] * 4,
            "ub": [5.0] * 4,
            "fmin": 0.0003,
            "func": f15,
        }
    )

    # F16 Six-Hump Camel
    def f16(x):
        x1, x2 = float(x[0]), float(x[1])
        return float(
            (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2
            + x1 * x2
            + (-4.0 + 4.0 * x2**2) * x2**2
        )

    benchmarks.append(
        {
            "key": "F16",
            "name": "F16 (Six-Hump Camel)",
            "type": "fixed_multimodal",
            "dim": 2,
            "lb": [-5.0, -5.0],
            "ub": [5.0, 5.0],
            "fmin": -1.0316,
            "func": f16,
        }
    )

    # F17 Branin
    def f17(x):
        x1, x2 = float(x[0]), float(x[1])
        pi = math.pi
        a = 1.0
        b = 5.1 / (4.0 * pi * pi)
        c = 5.0 / pi
        r = 6.0
        s = 1.0
        t = 1.0 / (8.0 * pi)
        return float(
            a * (x2 - b * x1 * x1 + c * x1 - r) ** 2
            + s * (1.0 - t) * math.cos(x1)
            + s
        )

    benchmarks.append(
        {
            "key": "F17",
            "name": "F17 (Branin)",
            "type": "fixed_multimodal",
            "dim": 2,
            "lb": [-5.0, 0.0],
            "ub": [10.0, 15.0],
            "fmin": 0.398,
            "func": f17,
        }
    )

    # F18 Goldstein-Price
    def f18(x):
        x1, x2 = float(x[0]), float(x[1])
        a = (
            1.0
            + (x1 + x2 + 1.0) ** 2
            * (
                19.0
                - 14.0 * x1
                + 3.0 * x1 * x1
                - 14.0 * x2
                + 6.0 * x1 * x2
                + 3.0 * x2 * x2
            )
        )
        b = (
            30.0
            + (2.0 * x1 - 3.0 * x2) ** 2
            * (
                18.0
                - 32.0 * x1
                + 12.0 * x1 * x1
                + 48.0 * x2
                - 36.0 * x1 * x2
                + 27.0 * x2 * x2
            )
        )
        return float(a * b)

    benchmarks.append(
        {
            "key": "F18",
            "name": "F18 (Goldstein-Price)",
            "type": "fixed_multimodal",
            "dim": 2,
            "lb": [-2.0, -2.0],
            "ub": [2.0, 2.0],
            "fmin": 3.0,
            "func": f18,
        }
    )

    # F19 Hartman 3D
    def f19(x):
        x = np.asarray(x, dtype=float)
        outer = 0.0
        for ii in range(4):
            inner = float(np.sum(_H3_A[ii] * (x - _H3_P[ii]) ** 2))
            outer += _H3_ALPHA[ii] * math.exp(-inner)
        return float(-outer)

    benchmarks.append(
        {
            "key": "F19",
            "name": "F19 (Hartman3)",
            "type": "fixed_multimodal",
            "dim": 3,
            "lb": [0.0, 0.0, 0.0],
            "ub": [1.0, 1.0, 1.0],
            "fmin": -3.86278,
            "func": f19,
        }
    )

    # F20 Hartman 6D
    def f20(x):
        x = np.asarray(x, dtype=float)
        outer = 0.0
        for ii in range(4):
            inner = float(np.sum(_H6_A[ii] * (x - _H6_P[ii]) ** 2))
            outer += _H6_ALPHA[ii] * math.exp(-inner)
        return float(-outer)

    benchmarks.append(
        {
            "key": "F20",
            "name": "F20 (Hartman6)",
            "type": "fixed_multimodal",
            "dim": 6,
            "lb": [0.0] * 6,
            "ub": [1.0] * 6,
            "fmin": -3.32,
            "func": f20,
        }
    )

    def _make_shekel(m: int) -> Callable[..., float]:
        Cm = _SHEKEL_C[:, :m]
        bm = _SHEKEL_BETA[:m]

        def shekel(x):
            x = np.asarray(x, dtype=float)
            outer = 0.0
            for ii in range(m):
                inner = float(np.sum((x - Cm[:, ii]) ** 2))
                outer += 1.0 / (inner + bm[ii])
            return float(-outer)

        return shekel

    benchmarks.append(
        {
            "key": "F21",
            "name": "F21 (Shekel5)",
            "type": "fixed_multimodal",
            "dim": 4,
            "lb": [0.0] * 4,
            "ub": [10.0] * 4,
            "fmin": -10.1532,
            "func": _make_shekel(5),
        }
    )
    benchmarks.append(
        {
            "key": "F22",
            "name": "F22 (Shekel7)",
            "type": "fixed_multimodal",
            "dim": 4,
            "lb": [0.0] * 4,
            "ub": [10.0] * 4,
            "fmin": -10.4028,
            "func": _make_shekel(7),
        }
    )
    benchmarks.append(
        {
            "key": "F23",
            "name": "F23 (Shekel10)",
            "type": "fixed_multimodal",
            "dim": 4,
            "lb": [0.0] * 4,
            "ub": [10.0] * 4,
            "fmin": -10.5363,
            "func": _make_shekel(10),
        }
    )

    return benchmarks


def benchmark_map(f7_noise: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """key 'F1'..'F23' -> benchmark dict (copy-safe: new F7 closure per noise)."""
    return {b["key"]: b for b in get_all_benchmarks(f7_noise)}


def type_display(btype: str) -> str:
    return {
        "unimodal": "Unimodal",
        "multimodal": "Multimodal",
        "fixed_multimodal": "Fixed_Multimodal",
    }.get(btype, btype)
