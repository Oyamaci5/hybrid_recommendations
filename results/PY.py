from pathlib import Path
import re
import pandas as pd

ROOT = Path(r"D:/hybrid_recommendations/results/wnmf/ml100k")
CSV_GLOB = "**/wnmf_results_ml100k_k*_sharedV*.csv"
SCENARIOS = ("global", "cluster_knn", "cluster_avg", "cluster_sharedV")


def _extract_meta(path: Path) -> dict:
    parts = path.parts
    meta = {
        "file": str(path.relative_to(ROOT)),
        "k_from_path": None,
        "fold": None,
        "run_id": None,
    }
    for part in parts:
        m_k = re.fullmatch(r"k(\d+)", part)
        if m_k:
            meta["k_from_path"] = int(m_k.group(1))
        m_fold = re.fullmatch(r"fold(\d+)", part)
        if m_fold:
            meta["fold"] = int(m_fold.group(1))
        m_run = re.fullmatch(r"run(\d+)", part)
        if m_run:
            meta["run_id"] = int(m_run.group(1))
    return meta


def _collect_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob(CSV_GLOB)):
        try:
            df = pd.read_csv(path, comment="#")
        except Exception:
            continue
        required = {"scenario", "latent_dim", "mae", "rmse", "algo_label"}
        if not required.issubset(df.columns):
            continue

        meta = _extract_meta(path)
        local = df.copy()
        local["scenario"] = local["scenario"].astype(str)
        local["latent_dim"] = pd.to_numeric(local["latent_dim"], errors="coerce")
        local["mae"] = pd.to_numeric(local["mae"], errors="coerce")
        local["rmse"] = pd.to_numeric(local["rmse"], errors="coerce")
        local["assignment_k"] = pd.to_numeric(local.get("assignment_k"), errors="coerce")
        local = local[local["scenario"].isin(SCENARIOS)].dropna(subset=["latent_dim", "mae", "rmse"])
        if local.empty:
            continue

        for _, r in local.iterrows():
            rows.append(
                {
                    "scenario": r["scenario"],
                    "algo": r.get("algo_label"),
                    "k": int(r["assignment_k"]) if pd.notna(r.get("assignment_k")) else meta["k_from_path"],
                    "latent": int(r["latent_dim"]),
                    "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]),
                    "fold": meta["fold"],
                    "run_id": meta["run_id"],
                    "file": meta["file"],
                }
            )
    return pd.DataFrame(rows)


def main():
    all_rows = _collect_rows()
    if all_rows.empty:
        print(f"No eligible rows under: {ROOT}")
        return

    # Senaryo bazlı en iyi satır (min mae, tie-break rmse)
    # Not: groupby().first() kolonları farklı satırlardan birleştirebilir.
    # Burada her senaryo için tek ve tutarlı satır seçiyoruz.
    best = (
        all_rows.sort_values(["scenario", "mae", "rmse"])
        .groupby("scenario", group_keys=False)
        .head(1)
        [["scenario", "run_id", "fold", "k", "latent", "algo", "mae", "rmse", "file"]]
        .sort_values(["mae", "rmse"])
    )
    print("\n=== Best row per scenario (all runs) ===")
    print(best.to_string(index=False))

    # Tek tablo: en iyi 50 satır
    top = all_rows.sort_values(["mae", "rmse"]).head(50)
    print("\n=== Top 50 rows (run_id / fold / k / latent / algo) ===")
    print(
        top[
            ["scenario", "run_id", "fold", "k", "latent", "algo", "mae", "rmse", "file"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()