"""
Meta-sezgisel kümeleme ablation'ı — küme/atama farklarını NE büyütüyor?

WNMF50 + euclidean + official fold-1, B0_KMEANS vs B_AVOA vs B1_HHO. Plan:
D-makaleler-meta-with-recommendation-hh-imperative-garden.md.

OFAT (one-factor-at-a-time) eksenleri:
  - fitness : wcss        vs knn_mae   (--fitness)
  - kmref   : overwrite   vs repair    vs capped(iter=1)
  - init    : mkpp        vs random    (--init-mode)
  - k       : 6, 10, 14
  - b0_ninit: 10          vs 1         (--b0-n-init; sadece B0)

Her hücre = tek generate_assignments çağrısı (3 algo). Eval, user'ın baseline
sayılarını üreten wnmf_experiment calc_avg_rating ile birebir (algo başına, tam
suffix). Çıktı: results/meta_ablation/ altında JSON + CSV.

Kullanım:
  python -m experiments.run_k_meta_ablation --phase assign --cells base,kmref_repair
  python -m experiments.run_k_meta_ablation --phase all     --cells smoke
  python -m experiments.run_k_meta_ablation --list-cells
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "mealpy" / "generate_assignments.py"
EXP = REPO / "wnmf" / "wnmf_experiment.py"

FOLD = 1
LATENT = 50
ALGOS = ["B0_KMEANS", "B_AVOA", "B1_HHO"]
META_ALGOS = ["B_AVOA", "B1_HHO"]

# Maliyet knob'ları (ablation hızı için makul; tezde artırılabilir).
WCSS_EPOCH = 400      # wcss meta arama epoch (early-stop yok → assignments kökü)
WCSS_POP = 30
KNN_ITER = 60         # knn_mae centroid arama epoch (per-eval ~0.4s; maliyet sınırı)
KNN_AGENTS = 20

OUT_DIR = REPO / "results" / "meta_ablation"
OUT_ANALYZE = OUT_DIR / "ablation_cluster_analysis.json"
OUT_METRICS = OUT_DIR / "ablation_metrics.csv"
WNMF_U_DUMP = OUT_DIR / "wnmf_u"   # --save-wnmf-u hedefi (ml100k_U.npy)

ASSIGN_ROOT = REPO / "mealpy" / "results" / "assignments"  # early-stop yok


@dataclass
class Cell:
    cid: str
    k: int = 6
    init: str = "mkpp"          # mkpp | random
    fitness: str = "wcss"       # wcss | knn_mae
    kmref: str = "overwrite"    # overwrite | repair | capped
    b0_n_init: int = 10
    algos: List[str] = field(default_factory=lambda: list(ALGOS))


def _init_token(init: str) -> str:
    return "_irand" if init == "random" else "_imkpp"


def algo_suffix(algo: str, cell: Cell) -> str:
    """generate_assignments ile birebir aynı, label SONRASI klasör soneki.

    out_suffix(_euc + init + _nogs + _trainonly_official_fF)
      + assign_suffix(_none_wnmf{L}_k{K}_pwcss [+_knnmae][+_kmref][+_ninitN]).
    """
    s = (
        "_euc"
        + _init_token(cell.init)
        + "_nogs"
        + f"_trainonly_official_f{FOLD}"
        + f"_none_wnmf{LATENT}_k{cell.k}_pwcss"
    )
    if cell.fitness == "knn_mae":
        s += "_knnmae"
    if algo in META_ALGOS and cell.kmref == "overwrite":
        s += "_kmref"
    if algo in META_ALGOS and cell.kmref == "capped":
        # overwrite bayrağı + iter=1 → generate_assignments soneki "_kmrefcap1".
        s += "_kmrefcap1"
    if algo == "B0_KMEANS" and cell.b0_n_init != 10:
        s += f"_ninit{cell.b0_n_init}"
    return s


def assign_dir(algo: str, cell: Cell) -> Path:
    return ASSIGN_ROOT / "ml100k" / f"{algo}{algo_suffix(algo, cell)}"


# ----------------------------------------------------------------------------
# Hücre kataloğu (OFAT, base etrafında)
# ----------------------------------------------------------------------------

def catalog() -> Dict[str, Cell]:
    cells = [
        # base = mevcut kurulum (wcss + overwrite + mkpp + k6 + n_init10)
        Cell("base"),
        # --- kmref ekseni ---
        Cell("kmref_repair", kmref="repair"),
        Cell("kmref_capped", kmref="capped"),
        # --- fitness ekseni ---
        Cell("fit_knnmae", fitness="knn_mae"),
        Cell("fit_knnmae_repair", fitness="knn_mae", kmref="repair"),
        # --- init ekseni ---
        Cell("init_random", init="random"),
        Cell("init_random_repair", init="random", kmref="repair"),
        # --- k ekseni ---
        Cell("k10", k=10),
        Cell("k14", k=14),
        Cell("k10_repair", k=10, kmref="repair"),
        Cell("k14_repair", k=14, kmref="repair"),
        # --- B0 n_init adillik diagnostiği (sadece B0) ---
        Cell("b0_ninit1", b0_n_init=1, algos=["B0_KMEANS"]),
        Cell("k14_b0_ninit1", k=14, b0_n_init=1, algos=["B0_KMEANS"]),
    ]
    return {c.cid: c for c in cells}


CELL_GROUPS = {
    "smoke": ["base", "kmref_repair"],
    "kmref": ["base", "kmref_repair", "kmref_capped"],
    "fitness": ["base", "fit_knnmae", "fit_knnmae_repair"],
    "init": ["base", "init_random", "init_random_repair"],
    "k": ["base", "k10", "k14", "kmref_repair", "k10_repair", "k14_repair"],
    "b0": ["base", "b0_ninit1", "k14", "k14_b0_ninit1"],
}


def resolve_cells(spec: str) -> List[Cell]:
    cat = catalog()
    if spec == "all":
        return list(cat.values())
    wanted: List[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CELL_GROUPS:
            wanted.extend(CELL_GROUPS[tok])
        elif tok in cat:
            wanted.append(tok)
        else:
            raise SystemExit(f"Bilinmeyen hücre/grup: {tok}")
    seen = set()
    out = []
    for cid in wanted:
        if cid not in seen:
            seen.add(cid)
            out.append(cat[cid])
    return out


# ----------------------------------------------------------------------------
# WNMF U dump (knn_mae fitness için bir kez)
# ----------------------------------------------------------------------------

def ensure_wnmf_u() -> Path:
    """knn_mae fitness --wnmf-model-path ister. WNMF U'yu (k/init bağımsız) bir
    kez dök; sonra tüm knn_mae hücreleri yeniden kullanır."""
    u_path = WNMF_U_DUMP / "ml100k_U.npy"
    if u_path.is_file():
        return u_path
    WNMF_U_DUMP.mkdir(parents=True, exist_ok=True)
    cmd = base_assign_cmd(Cell("u_dump", algos=["B0_KMEANS"])) + [
        "--save-wnmf-u", str(WNMF_U_DUMP),
    ]
    print("WNMF-U DUMP:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), check=True)
    if not u_path.is_file():
        raise SystemExit(f"WNMF U dökülemedi: {u_path} yok")
    return u_path


# ----------------------------------------------------------------------------
# Atama üretimi
# ----------------------------------------------------------------------------

def base_assign_cmd(cell: Cell) -> List[str]:
    return [
        sys.executable, "-u", str(GEN),
        "--dataset", "100k",
        "--algo", *cell.algos,
        "--no-prune", "--no-gray-sheep",
        "--preprocess", "none",
        "--feature-extraction", "wnmf",
        "--svd-components", str(LATENT),
        "--wnmf-epochs", "50",
        "--legacy-wnmf-suffix",
        "--init-mode", cell.init if cell.init != "random" else "random",
        "--cluster-metric", "euclidean",
        "--cluster-objective", "wcss",
        "--train-only", "--eval-split", "official",
        "--fold", str(FOLD),
        "--k", str(cell.k),
        "--b0-n-init", str(cell.b0_n_init),
        "--jobs", "1",
    ]


def phase_assign(cells: List[Cell], *, skip_existing: bool) -> int:
    for cell in cells:
        cmd = base_assign_cmd(cell)
        cmd += ["--fitness", cell.fitness]
        if cell.fitness == "knn_mae":
            u_path = ensure_wnmf_u()
            cmd += [
                "--wnmf-model-path", str(u_path),
                "--centroid-iter", str(KNN_ITER),
                "--centroid-agents", str(KNN_AGENTS),
            ]
        else:
            cmd += ["--baseline-epoch", str(WCSS_EPOCH), "--pop-size", str(WCSS_POP)]

        if cell.kmref == "overwrite":
            cmd += ["--kmeans-refine-overwrite", "--kmeans-refine-iter", "300"]
        elif cell.kmref == "capped":
            cmd += ["--kmeans-refine-overwrite", "--kmeans-refine-iter", "1"]
        # repair: --kmeans-refine-overwrite YOK → _repair_empty_clusters devrede.

        if skip_existing:
            cmd.append("--skip-existing")
        print(f"\n=== ASSIGN [{cell.cid}] ===", flush=True)
        print(" ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f"  ASSIGN [{cell.cid}] BAŞARISIZ rc={rc}", file=sys.stderr)
            return rc
    return 0


# ----------------------------------------------------------------------------
# Küme analizi + ARI
# ----------------------------------------------------------------------------

def _cluster_stats(a: np.ndarray) -> dict:
    ids = a.astype(int)
    k = int(ids.max()) + 1
    sizes = [int(np.sum(ids == c)) for c in range(k)]
    return {
        "k_active": int(len(np.unique(ids))),
        "size_min": int(min(sizes)),
        "size_max": int(max(sizes)),
        "size_std": round(float(np.std(sizes)), 3),
        "sizes": sizes,
    }


def phase_analyze(cells: List[Cell]) -> dict:
    out: Dict[str, dict] = {}
    for cell in cells:
        labels: Dict[str, np.ndarray] = {}
        per_algo: Dict[str, dict] = {}
        for algo in cell.algos:
            d = assign_dir(algo, cell)
            f = d / "assignments.npy"
            if not f.is_file():
                per_algo[algo] = {"error": f"yok: {d}"}
                continue
            a = np.load(f)
            labels[algo] = a
            per_algo[algo] = _cluster_stats(a)
        pairwise = {}
        algo_list = [a for a in cell.algos if a in labels]
        for i in range(len(algo_list)):
            for j in range(i + 1, len(algo_list)):
                a1, a2 = algo_list[i], algo_list[j]
                la, lb = labels[a1], labels[a2]
                if len(la) != len(lb):
                    continue
                ari = float(adjusted_rand_score(la, lb))
                same = float(np.mean(la == lb))
                pairwise[f"{a1}|{a2}"] = {
                    "ari": round(ari, 4),
                    "same_label_pct": round(100 * same, 2),
                }
        out[cell.cid] = {
            "config": {
                "k": cell.k, "init": cell.init, "fitness": cell.fitness,
                "kmref": cell.kmref, "b0_n_init": cell.b0_n_init,
            },
            "per_algo": per_algo,
            "pairwise": pairwise,
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ANALYZE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT_ANALYZE}")
    for cid, info in out.items():
        pw = info["pairwise"]
        pw_str = "  ".join(
            f"{k.split('|')[0][:5]}~{k.split('|')[1][:5]} ARI={v['ari']:.2f}"
            for k, v in pw.items()
        )
        print(f"  [{cid}] {info['config']}  {pw_str}")
    return out


# ----------------------------------------------------------------------------
# Eval — wnmf_experiment calc_avg_rating (algo başına, tam suffix)
# ----------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"\| CalcAvgRating\]\s*MAE=(?P<mae>[\d.]+)\s+RMSE=(?P<rmse>[\d.]+)\s*\|"
    r"\s*P@10=(?P<p10>[\d.]+)\s+R@10=(?P<r10>[\d.]+)\s+NDCG@10=(?P<ndcg>[\d.]+)"
)


def _eval_one(algo: str, cell: Cell) -> Optional[dict]:
    """wnmf_experiment calc_avg_rating çalıştırıp metrikleri STDOUT'tan ayıkla.

    (CSV-newest-file yerine stdout regex: --assign-suffix klasör eşleşmesi
    tutmazsa veya boş CSV oluşursa sessizce None döner; faz çökmes.)
    """
    suf = algo_suffix(algo, cell)
    cmd = [
        sys.executable, "-u", str(EXP),
        "--dataset", "100k",
        "--eval-split", "official",
        "--fold", str(FOLD),
        "--mode", "baselines",
        "--paper-mode",
        "--no-global", "--no-cluster-knn",
        "--k", str(cell.k),
        "--algo", algo,
        "--assign-root", str(ASSIGN_ROOT.as_posix()),
        "--assign-suffix", suf,
        "--top-n", "10",
        "--relevance-threshold", "4.0",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except Exception as exc:
        print(f"  EVAL [{cell.cid}/{algo}] subprocess hata: {exc}", file=sys.stderr)
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    if "bulunamad" in out:
        print(f"  EVAL [{cell.cid}/{algo}] assignment bulunamadı (suffix={suf})",
              file=sys.stderr)
        return None
    m = _METRIC_RE.search(out)
    if not m:
        print(f"  EVAL [{cell.cid}/{algo}] metrik satırı bulunamadı (rc={proc.returncode})",
              file=sys.stderr)
        return None
    return {
        "cell": cell.cid, "algo": algo,
        "k": cell.k, "init": cell.init, "fitness": cell.fitness,
        "kmref": cell.kmref, "b0_n_init": cell.b0_n_init,
        "mae": float(m.group("mae")), "rmse": float(m.group("rmse")),
        "ndcg_at_10": float(m.group("ndcg")), "precision_at_10": float(m.group("p10")),
    }


def phase_eval(cells: List[Cell]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        for algo in cell.algos:
            print(f"\n=== EVAL [{cell.cid}/{algo}] ===", flush=True)
            res = _eval_one(algo, cell)
            if res is not None:
                rows.append(res)
    df = pd.DataFrame(rows)
    if not df.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_METRICS, index=False)
        print(f"\n-> {OUT_METRICS}")
        # Hücre içi MAE spread'i (algoritmalar arası fark) — asıl ilgilendiğimiz.
        for cid, g in df.groupby("cell"):
            if len(g) > 1 and g["mae"].notna().all():
                spread = float(g["mae"].max() - g["mae"].min())
                print(f"  [{cid}] MAE spread={spread:.4f}  " +
                      "  ".join(f"{a}={m:.4f}" for a, m in zip(g["algo"], g["mae"])))
        print(df.to_string(index=False))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assign", "analyze", "eval", "all"], default="all")
    ap.add_argument("--cells", default="smoke",
                    help="virgülle hücre/grup id'leri ya da 'all'. Gruplar: "
                         + ", ".join(CELL_GROUPS))
    ap.add_argument("--skip-existing", action="store_true", default=False)
    ap.add_argument("--list-cells", action="store_true", default=False)
    args = ap.parse_args()

    if args.list_cells:
        for cid, c in catalog().items():
            print(f"  {cid:20s} k={c.k} init={c.init} fitness={c.fitness} "
                  f"kmref={c.kmref} b0_ninit={c.b0_n_init} algos={c.algos}")
        print("\nGruplar:")
        for g, ids in CELL_GROUPS.items():
            print(f"  {g:10s} -> {ids}")
        return 0

    cells = resolve_cells(args.cells)
    print(f"Hücreler ({len(cells)}): {[c.cid for c in cells]}")

    if args.phase in ("assign", "all"):
        rc = phase_assign(cells, skip_existing=args.skip_existing)
        if rc != 0:
            return rc
    if args.phase in ("analyze", "all"):
        phase_analyze(cells)
    if args.phase in ("eval", "all"):
        phase_eval(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
