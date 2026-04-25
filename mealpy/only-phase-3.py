"""
Sadece Aşama 3 — Tam Matris
Küratörlü sabit alt küme (TOP10 + literatür + MUST); Aşama 1/2 taraması yok.
Mealpy tam adları PHASE3_FIXED_ALGORITHM_FULL_NAMES içinde; sınıflar
get_all_algorithms_v3() ile eşlenir; `SA.GaussianSA` gibi Original olmayan
sınıflar `_phase3_extra_mealpy_entries()` ile eklenir.
N kez tekrar; çıktılar results/phase3/last/<başlangıç..>/ altına yazılır
(--start-run veya ONLY_PHASE3_RUN_START, varsayılan 7).

Ortak fonksiyonlar mealpy-algorithms-comparision.py dosyasından yüklenir.
"""

import argparse
import os
import importlib.util
import sys
import sqlite3
import time
import csv
from datetime import datetime
from typing import List
import numpy as np
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Aşama 3 run_phase paralellik; gerçek değer __main__ içinde resolve_phase3_parallel_workers() ile atanır.
PHASE3_PARALLEL_WORKERS = max(1, min(8, (os.cpu_count() or 4)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PHASE3_RUNS_ROOT = os.path.join(RESULTS_DIR, "phase3", "new_runs")
PHASE3_OUT_NAME_DEFAULT = "new_runs"
PHASE3_DB_PATH = os.path.join(RESULTS_DIR, "assignment_experiments.sqlite")
PHASE3_DB_TABLE = "mealpy-comparision"
DB_WRITE_ENABLED = True

# Phase-3 sabit algoritma listesi (Mealpy full_name)
# Not: "BAT" için Mealpy sınıfı BA.OriginalBA kullanılır.
PHASE3_FIXED_ALGORITHM_FULL_NAMES = (
    "CircleSA.OriginalCircleSA",
    "BES.OriginalBES",
    "TSO.OriginalTSO",
    "HHO.OriginalHHO",
    "HGS.OriginalHGS",
    "WOA.OriginalWOA",
    "BA.OriginalBA",        # BAT
    "GA.MultiGA",           # non-Original, _phase3_extra_mealpy_entries() ile eklenir
    "DE.OriginalDE",
    "GWO.OriginalGWO",
    "GOA.OriginalGOA",
    "PSO.OriginalPSO",
    "ABC.OriginalABC",
    "AGTO.OriginalAGTO",
    "AVOA.OriginalAVOA",
    "MFO.OriginalMFO",
    "TWO.OriginalTWO",
    "SMA.OriginalSMA",
    "BeesA.OriginalBeesA",
)


def _phase3_family_label(full_name: str) -> str:
    """
    Basit aile etiketi:
      - global_local: hibrit adlar (X+Y / X||Y)
      - memetic: GA.MultiGA
      - global: diğer tek-algoritma yaklaşımları
    """
    n = str(full_name)
    if "+" in n or "||" in n:
        return "global_local"
    if n == "GA.MultiGA":
        return "memetic"
    return "global"


def _phase3_rationale(full_name: str) -> str:
    fam = _phase3_family_label(full_name)
    if fam == "global_local":
        return "Hybrid global+local arama; kesif ve somuruyu birlikte dengeler."
    if fam == "memetic":
        return "Memetic GA; evrimsel arama ile yerel iyilestirmeyi birlestirir."
    return "Tek-katman global arama; karsilastirma ve saglam baseline icin uygun."


def _write_algorithm_notes(run_root: str, algos_phase3: List[dict]) -> None:
    path = os.path.join(run_root, "algorithm_selection_notes.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "family", "rationale"])
        for a in algos_phase3:
            name = a["full_name"]
            writer.writerow([name, _phase3_family_label(name), _phase3_rationale(name)])

# Tekrar sayısı (ardışık klasör: start, start+1, …)
N_PHASE3_REPEATS = 20
# İlk tekrar klasör numarası (--start-run / ONLY_PHASE3_RUN_START yoksa kullanılır)
PHASE3_RUN_START_DEFAULT = 17
PHASE3_EPOCH_DEFAULT = 50
PHASE3_POP_SIZE_DEFAULT = 30
PHASE3_K3_DEFAULT = 30
PHASE3_K_BEH_DEFAULT = 60

os.makedirs(RESULTS_DIR, exist_ok=True)

# Ortak fonksiyonları ana scriptten yükle (dosya adında '-' olduğu için importlib kullanıyoruz)
SHARED_PATH = os.path.join(BASE_DIR, "mealpy-algorithms-comparision.py")
spec = importlib.util.spec_from_file_location("mealpy_algorithms_comparison", SHARED_PATH)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

load_movielens = shared.load_movielens
mkmeans_plus_plus_init = shared.mkmeans_plus_plus_init
get_all_algorithms_v3 = shared.get_all_algorithms_v3
run_phase = shared.run_phase
rank_and_filter = shared.rank_and_filter
run_behavior_analysis = shared.run_behavior_analysis


def parse_phase3_cli(argv=None):
    """CLI argumanlarini cozer."""
    p = argparse.ArgumentParser(
        description="Sadece Aşama 3 — tekrarlı tam matris çalıştırması",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Aşama 3 run_phase paralel süreç sayısı (1=sıralı). "
        "Verilmezse ONLY_PHASE3_WORKERS, o da yoksa CPU.",
    )
    p.add_argument(
        "--start-run",
        type=int,
        default=None,
        metavar="N",
        help="İlk tekrar klasörü results/phase3/new_runs/N. "
        "Verilmezse ONLY_PHASE3_RUN_START, o da yoksa %d." % PHASE3_RUN_START_DEFAULT,
    )
    p.add_argument("--repeats", type=int, default=N_PHASE3_REPEATS)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--out-name", type=str, default=PHASE3_OUT_NAME_DEFAULT)
    p.add_argument("--epoch", type=int, default=PHASE3_EPOCH_DEFAULT)
    p.add_argument("--pop-size", type=int, default=PHASE3_POP_SIZE_DEFAULT)
    p.add_argument("--k3", type=int, default=PHASE3_K3_DEFAULT)
    p.add_argument("--k-beh", type=int, default=PHASE3_K_BEH_DEFAULT)
    p.add_argument(
        "--fast-5min",
        action="store_true",
        help="Hizli preset: epoch=8, pop_size=10, k3=20, k_beh=30, repeats<=10, max-minutes=5",
    )
    ns, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    if ns.workers is not None:
        workers = max(1, int(ns.workers))
    else:
        ev = os.environ.get("ONLY_PHASE3_WORKERS", "").strip()
        if ev:
            workers = max(1, int(ev))
        else:
            workers = max(1, min(8, (os.cpu_count() or 4)))

    if ns.start_run is not None:
        run_start = max(1, int(ns.start_run))
    else:
        evs = os.environ.get("ONLY_PHASE3_RUN_START", "").strip()
        if evs:
            run_start = max(1, int(evs))
        else:
            run_start = max(1, int(PHASE3_RUN_START_DEFAULT))

    ns.workers = workers
    ns.start_run = run_start
    ns.repeats = max(1, int(ns.repeats))
    ns.epoch = max(1, int(ns.epoch))
    ns.pop_size = max(2, int(ns.pop_size))
    ns.k3 = max(2, int(ns.k3))
    ns.k_beh = max(2, int(ns.k_beh))

    if ns.fast_5min:
        ns.epoch = 8
        ns.pop_size = 10
        ns.k3 = 20
        ns.k_beh = 30
        ns.repeats = min(ns.repeats, 10)
        if ns.max_minutes is None:
            ns.max_minutes = 5.0

    return ns


def init_phase3_db(db_path=PHASE3_DB_PATH, table_name=PHASE3_DB_TABLE):
    """Phase-3 davranış karşılaştırma tablosunu oluşturur."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    idx_suffix = table_name.replace("-", "_")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_num INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                final_wcss REAL,
                silhouette REAL,
                davies_bouldin REAL,
                convergence_speed REAL,
                exploration_ratio REAL,
                execution_time_sec REAL,
                ch_score REAL,
                category TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()}
        if "silhouette" not in cols:
            conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN silhouette REAL')
        if "davies_bouldin" not in cols:
            conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN davies_bouldin REAL')
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{idx_suffix}_run ON "{table_name}"(run_num)'
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{idx_suffix}_algo ON "{table_name}"(algorithm)'
        )


def save_behavior_to_db(run_num, behavior_results,
                        db_path=PHASE3_DB_PATH, table_name=PHASE3_DB_TABLE):
    """Her algoritma için behavior metriklerini tek tek DB'ye yazar."""
    if not behavior_results:
        return 0

    now = datetime.now().isoformat()
    rows = []
    for r in behavior_results:
        rows.append((
            int(run_num),
            str(r.get("algorithm")),
            float(r["final_wcss"]) if r.get("final_wcss") is not None else None,
            float(r["silhouette"]) if r.get("silhouette") is not None else None,
            float(r["davies_bouldin"]) if r.get("davies_bouldin") is not None else None,
            float(r["convergence_speed"]) if r.get("convergence_speed") is not None else None,
            float(r["exploration_ratio"]) if r.get("exploration_ratio") is not None else None,
            float(r["execution_time_sec"]) if r.get("execution_time_sec") is not None else None,
            float(r["ch_score"]) if r.get("ch_score") is not None else None,
            str(r.get("category")) if r.get("category") is not None else None,
            now,
        ))

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            f"""
            INSERT INTO "{table_name}"
                (run_num, algorithm, final_wcss, silhouette, davies_bouldin,
                 convergence_speed, exploration_ratio, execution_time_sec,
                 ch_score, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _collect_numbered_run_dirs(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    pairs = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.isdigit():
            pairs.append((int(name), p))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def aggregate_all_runs(root_dir):
    """Tüm tekrarları tekil all-runs CSV dosyalarında birleştirir."""
    import pandas as pd

    behavior_all, success_all, ranked_all = [], [], []
    for run_dir in _collect_numbered_run_dirs(root_dir):
        run_num = int(os.path.basename(run_dir))
        behavior_csv = os.path.join(run_dir, "behavior_analysis.csv")
        success_csv = os.path.join(run_dir, "phase3_success.csv")
        ranked_csv = os.path.join(run_dir, "ranked_scores.csv")

        if os.path.exists(behavior_csv):
            bdf = pd.read_csv(behavior_csv)
            # Yeni metrik kolonlarını (eski run'larla da uyumlu) garanti et.
            for col in ("execution_time_sec", "ch_score"):
                if col not in bdf.columns:
                    bdf[col] = np.nan
            bdf.insert(0, "run_num", run_num)
            behavior_all.append(bdf)
        if os.path.exists(success_csv):
            sdf = pd.read_csv(success_csv)
            sdf.insert(0, "run_num", run_num)
            success_all.append(sdf)
        if os.path.exists(ranked_csv):
            rdf = pd.read_csv(ranked_csv)
            rdf.insert(0, "run_num", run_num)
            ranked_all.append(rdf)

    if behavior_all:
        pd.concat(behavior_all, ignore_index=True).to_csv(
            os.path.join(root_dir, "behavior_analysis_all_runs.csv"), index=False
        )
    if success_all:
        pd.concat(success_all, ignore_index=True).to_csv(
            os.path.join(root_dir, "phase3_success_all_runs.csv"), index=False
        )
    if ranked_all:
        pd.concat(ranked_all, ignore_index=True).to_csv(
            os.path.join(root_dir, "ranked_scores_all_runs.csv"), index=False
        )


def l2_normalize_users(matrix):
    """
    Kullanıcı satırlarını L2-normalize eder.
    Sıfır normlu satırlar aynen bırakılır (0'a bölme engellenir).
    """
    mat = np.asarray(matrix, dtype=np.float32)
    return normalize(mat, norm='l2', axis=1).astype(np.float32, copy=False)


def _phase3_extra_mealpy_entries():
    """
    get_all_algorithms_v3 yalnızca sınıf adı Original* olanları toplar.
    GaussianSA vb. burada tamamlanır.
    """
    from mealpy.physics_based import SA, TWO
    from mealpy.evolutionary_based import GA

    return {
        "SA.GaussianSA": {
            "full_name": "SA.GaussianSA",
            "module": "mealpy.physics_based.SA",
            "class_name": "GaussianSA",
            "class": SA.GaussianSA,
        },
        "GA.MultiGA": {
            "full_name": "GA.MultiGA",
            "module": "mealpy.evolutionary_based.GA",
            "class_name": "MultiGA",
            "class": GA.MultiGA,
        },
        "TWO.OriginalTWO": {
            "full_name": "TWO.OriginalTWO",
            "module": "mealpy.physics_based.TWO",
            "class_name": "OriginalTWO",
            "class": TWO.OriginalTWO,
        },
    }


def load_algorithms_by_full_names(full_names, all_algos):
    """Verilen Mealpy tam adlarını sırayla all_algos (+ ek sözlük) ile eşler."""
    by_name = {a["full_name"]: a for a in all_algos}
    by_name.update(_phase3_extra_mealpy_entries())
    missing = [n for n in full_names if n not in by_name]
    if missing:
        raise KeyError(
            "Algoritma listesinde yok (catalog + ekler): " + ", ".join(missing)
        )
    return [by_name[n] for n in full_names]


def run_one_phase3_repeat(
    run_idx,
    full_matrix,
    algos_phase3,
    *,
    run_ordinal=1,
    n_repeats=None,
    K3=30,
    K_beh=60,
    epoch=50,
    pop_size=30,
):
    """
    Tek bir tekrar: phase3_* csv, ranked_scores, FINAL_RECOMMENDATIONS, behavior_analysis.
    run_idx: klasör adı (sayı); run_ordinal: bu oturumdaki sıra (1..n_repeats).
    """
    if n_repeats is None:
        n_repeats = N_PHASE3_REPEATS
    run_dir = os.path.join(PHASE3_RUNS_ROOT, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)

    # Her tekrarda farklı rastgele başlangıç / alt örnek (istatistiksel çeşitlilik)
    seed = 10_000 + run_idx
    init_full = mkmeans_plus_plus_init(
        full_matrix, K=K3, n_solutions=50, seed=seed
    )

    df3 = run_phase(
        phase_num=3,
        algo_list=algos_phase3,
        matrix=full_matrix,
        K=K3,
        initial_solutions=init_full,
        epoch=epoch,
        pop_size=pop_size,
        time_limit=None,
        save_path=run_dir,
        parallel_workers=PHASE3_PARALLEL_WORKERS,
    )

    final = rank_and_filter(df3, top_n=5, save_path=run_dir)
    final.to_csv(
        os.path.join(run_dir, "FINAL_RECOMMENDATIONS.csv"),
        index=False,
    )

    print(f"\n>>> Tekrar {run_idx}: DAVRANIŞ ANALİZİ (ML-100K tam matris)")
    matrix_beh = full_matrix
    init_beh = mkmeans_plus_plus_init(
        matrix_beh, K=K_beh, n_solutions=20, seed=seed + 2
    )
    behavior_results, _, _ = run_behavior_analysis(
        algos_phase3,
        matrix_beh,
        K_beh,
        initial_solutions=init_beh,
        epoch=epoch,
        pop_size=20,
        save_path=run_dir,
    )
    # run_phase metriklerini (silhouette, davies_bouldin) behavior sonuçlarına ekle.
    phase_metrics = {}
    if df3 is not None and not df3.empty:
        for _, prow in df3.iterrows():
            phase_metrics[str(prow.get("algorithm"))] = {
                "silhouette": prow.get("silhouette"),
                "davies_bouldin": prow.get("davies_bouldin"),
            }
    for r in behavior_results:
        extra = phase_metrics.get(str(r.get("algorithm")), {})
        r["silhouette"] = extra.get("silhouette")
        r["davies_bouldin"] = extra.get("davies_bouldin")

    if DB_WRITE_ENABLED:
        try:
            inserted = save_behavior_to_db(run_idx, behavior_results)
            print(f"DB kayıt: {inserted} satır -> {PHASE3_DB_PATH} | tablo: {PHASE3_DB_TABLE}")
        except Exception as e:
            print(f"DB uyarı: kayıt atlanıyor, CSV devam ediyor. Hata: {e}")
    else:
        print("DB yazımı devre dışı; yalnız CSV çıktıları üretildi.")

    info = f"""Sadece Aşama 3 — toplu tekrar
Klasör: results/phase3/new_runs/{run_idx}/

Bu çalıştırmada üretilen dosyalar:
  - phase3_complete.csv, phase3_success.csv, phase3_failed.csv
  - phase3_partial.csv (her 10 algoritmada bir ara kayıt)
  - ranked_scores.csv (bileşik skorlar / sıralama)
  - FINAL_RECOMMENDATIONS.csv (üst 5 öneri)
  - behavior_analysis.csv

Kaynak: only-phase-3.py → PHASE3_FIXED_ALGORITHM_FULL_NAMES ({len(algos_phase3)} secilmis)
Paralel Aşama 3 işçi: {PHASE3_PARALLEL_WORKERS} (--workers veya ONLY_PHASE3_WORKERS)
Bu oturum: {run_ordinal}. tekrar / {n_repeats} | Klasör no: {run_idx}
"""
    with open(os.path.join(run_dir, "RUN_INFO.txt"), "w", encoding="utf-8") as f:
        f.write(info)

    print(f"Sonuclar: {run_dir}")
    return run_dir


if __name__ == "__main__":
    ns = parse_phase3_cli()
    PHASE3_PARALLEL_WORKERS = ns.workers
    PHASE3_RUN_START = ns.start_run
    N_PHASE3_REPEATS = ns.repeats
    PHASE3_RUNS_ROOT = os.path.join(RESULTS_DIR, "phase3", ns.out_name)
    _last_run_idx = PHASE3_RUN_START + N_PHASE3_REPEATS - 1

    print("=" * 60)
    print("SADECE AŞAMA 3 — ÇOKLU TEKRAR")
    print(
        f"Cikti koku: {PHASE3_RUNS_ROOT}/{PHASE3_RUN_START} .. {_last_run_idx} "
        f"({N_PHASE3_REPEATS} tekrar)"
    )
    _ev = os.environ.get("ONLY_PHASE3_WORKERS")
    _evs = os.environ.get("ONLY_PHASE3_RUN_START")
    print(
        f"Aşama 3 paralellik: {PHASE3_PARALLEL_WORKERS} süreç "
        f"(sıralı: --workers 1 veya ONLY_PHASE3_WORKERS=1)"
    )
    print(
        f"  → ONLY_PHASE3_WORKERS={_ev!r} | ONLY_PHASE3_RUN_START={_evs!r} "
        f"| os.cpu_count()={os.cpu_count()}"
    )
    print(
        f"Parametreler: epoch={ns.epoch}, pop_size={ns.pop_size}, "
        f"k3={ns.k3}, k_beh={ns.k_beh}, max_minutes={ns.max_minutes}"
    )
    print("=" * 60)

    full_matrix_raw = load_movielens(
        os.path.join(os.path.dirname(BASE_DIR), "data", "ml-100k", "u.data")
    )
    full_matrix = l2_normalize_users(full_matrix_raw)
    print("ML-100K matrisine satır-bazlı L2 normalization uygulandı.")
    # Aşama 1/2 yok: sabit liste + mealpy sınıf eşlemesi
    all_algos = get_all_algorithms_v3()
    algos_phase3 = load_algorithms_by_full_names(
        PHASE3_FIXED_ALGORITHM_FULL_NAMES, all_algos
    )

    fam_counts = {"global": 0, "global_local": 0, "memetic": 0}
    for a in algos_phase3:
        fam = _phase3_family_label(a["full_name"])
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    print(f"Sabit alt küme → {len(algos_phase3)} algoritma:")
    print(
        "Aile sayilari -> "
        f"global: {fam_counts.get('global', 0)}, "
        f"global_local: {fam_counts.get('global_local', 0)}, "
        f"memetic: {fam_counts.get('memetic', 0)}"
    )
    for a in algos_phase3:
        print(f"  {a['full_name']}  [{_phase3_family_label(a['full_name'])}]")

    os.makedirs(PHASE3_RUNS_ROOT, exist_ok=True)
    _write_algorithm_notes(PHASE3_RUNS_ROOT, algos_phase3)
    try:
        init_phase3_db()
        print(f"DB hazır: {PHASE3_DB_PATH} | tablo: {PHASE3_DB_TABLE}")
    except Exception as e:
        DB_WRITE_ENABLED = False
        print(f"DB init uyarı: DB devre dışı, CSV ile devam. Hata: {e}")

    t_global_start = time.time()
    for run_ordinal, run_idx in enumerate(
        range(PHASE3_RUN_START, PHASE3_RUN_START + N_PHASE3_REPEATS),
        start=1,
    ):
        if ns.max_minutes is not None:
            elapsed_min = (time.time() - t_global_start) / 60.0
            if elapsed_min >= float(ns.max_minutes):
                print(
                    f"\nZaman limiti doldu ({elapsed_min:.2f} dk >= {ns.max_minutes} dk). "
                    "Yeni tekrar baslatilmiyor."
                )
                break
        print("\n" + "=" * 60)
        print(
            f"TEKRAR klasör {run_idx} (oturum {run_ordinal}/{N_PHASE3_REPEATS})"
        )
        print("=" * 60)
        run_one_phase3_repeat(
            run_idx,
            full_matrix,
            algos_phase3,
            run_ordinal=run_ordinal,
            n_repeats=N_PHASE3_REPEATS,
            K3=ns.k3,
            K_beh=ns.k_beh,
            epoch=ns.epoch,
            pop_size=ns.pop_size,
        )
        aggregate_all_runs(PHASE3_RUNS_ROOT)

    print("\n" + "=" * 60)
    print("TÜM TEKRARLAR BİTTİ")
    print(f"Kök: {PHASE3_RUNS_ROOT}")
    print("=" * 60)
