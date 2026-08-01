"""Cluster size distributions for separation-protocol assignments."""
from collections import Counter
from pathlib import Path

import numpy as np

root = Path("mealpy/results/assignments/ml100k")
ALGOS = ["B0_KMEANS", "B1_HHO", "HA_AVOAHGS", "IWO_HHO"]
PROTO_SUFFIX = {
    "euc_multi": "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}",
    "fuzzy_fcm": "_fuzzy_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}",
    "euc_knnmae": "_euc_imkpp_nogs_trainonly_rand_f1_none_wnmf20_k{k}_knnmae",
}
KS = [14, 21, 30, 50, 70]


def stats(assign: np.ndarray) -> dict:
    c = Counter(assign.astype(int).tolist())
    sizes = sorted(c.values())
    return {
        "active": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "sizes": sizes,
        "singletons": sum(1 for s in sizes if s == 1),
        "le3": sum(1 for s in sizes if s <= 3),
        "singleton_ids": [k for k, v in c.items() if v == 1],
    }


def main() -> None:
    for k in KS:
        print("\n" + "#" * 72)
        print(f"K = {k}")
        print("#" * 72)
        for proto, tmpl in PROTO_SUFFIX.items():
            suf = tmpl.format(k=k)
            print(f"\n--- {proto} ---")
            for algo in ALGOS:
                p = root / f"{algo}{suf}" / "assignments.npy"
                if not p.is_file():
                    print(f"  {algo}: MISSING")
                    continue
                s = stats(np.load(p))
                extra = ""
                if s["singletons"]:
                    extra = f"  singleton_ids={s['singleton_ids']}"
                print(
                    f"  {algo}: n={s['active']} min={s['min']} max={s['max']} "
                    f"mean={s['mean']:.1f} std={s['std']:.1f} "
                    f"singletons={s['singletons']} n<=3={s['le3']}"
                )
                print(f"    sizes: {s['sizes']}{extra}")


if __name__ == "__main__":
    main()
