"""K=10 separation assignments: cluster size distributions."""
from collections import Counter
from pathlib import Path

import numpy as np

root = Path("mealpy/results/assignments/ml100k")
ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
SUFFIXES = {
    "euc_multi": "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k10",
    "fuzzy_fcm": "_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k10",
    "euc_knnmae": "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k10_knnmae",
}


def stats(assign: np.ndarray) -> dict:
    c = Counter(assign.astype(int).tolist())
    sizes = sorted(c.values())
    return {
        "active": sum(1 for s in sizes if s > 0),
        "min": min(sizes),
        "max": max(sizes),
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "sizes": sizes,
        "counts": dict(sorted(c.items())),
        "singletons": sum(1 for s in sizes if s == 1),
        "le3": sum(1 for s in sizes if s <= 3),
    }


def main() -> None:
    for proto, suf in SUFFIXES.items():
        print("=" * 72)
        print(f"{proto}  K=10")
        print("=" * 72)
        for algo in ALGOS:
            p = root / f"{algo}{suf}" / "assignments.npy"
            if not p.is_file():
                print(f"  {algo}: MISSING")
                continue
            s = stats(np.load(p))
            print(
                f"  {algo}: clusters={s['active']} "
                f"min={s['min']} max={s['max']} mean={s['mean']:.1f} std={s['std']:.1f} "
                f"singletons={s['singletons']} n<=3={s['le3']}"
            )
            print(f"    sizes sorted: {s['sizes']}")
            if s["singletons"]:
                ids = [k for k, v in s["counts"].items() if v == 1]
                print(f"    singleton ids: {ids}")


if __name__ == "__main__":
    main()
