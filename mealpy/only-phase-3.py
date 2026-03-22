"""
Sadece Aşama 3 — Tam Matris
Küratörlü sabit alt küme (TOP10 + literatür + MUST); Aşama 1/2 taraması yok.
Mealpy tam adları PHASE3_FIXED_ALGORITHM_FULL_NAMES içinde; sınıflar
get_all_algorithms_v3() ile eşlenir.
N kez tekrar; çıktılar results/phase3/last/<1..N>/ altına yazılır.

Ortak fonksiyonlar mealpy-algorithms-comparision.py dosyasından yüklenir.
"""

import os
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Aşama 3 run_phase: eşzamanlı süreç sayısı (1 = sıralı). Ortam: ONLY_PHASE3_WORKERS=4
_pw_env = os.environ.get("ONLY_PHASE3_WORKERS", "").strip()
if _pw_env:
    PHASE3_PARALLEL_WORKERS = max(1, int(_pw_env))
else:
    PHASE3_PARALLEL_WORKERS = max(1, min(8, (os.cpu_count() or 4)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PHASE3_LAST_ROOT = os.path.join(RESULTS_DIR, "phase3", "last")

# Seçilmiş 20 algoritma (sıra: TOP10 → LİTERATÜR → MUST → ekler); Mealpy full_name
PHASE3_FIXED_ALGORITHM_FULL_NAMES = (
    # TOP10
    "HGS.OriginalHGS",
    "HHO.OriginalHHO",
    "AEO.OriginalAEO",
    "CircleSA.OriginalCircleSA",
    "MFO.OriginalMFO",
    "SquirrelSA.OriginalSquirrelSA",
    "INFO.OriginalINFO",
    "NGO.OriginalNGO",
    "BWO.OriginalBWO",
    # LİTERATÜR
    "WOA.OriginalWOA",
    "GWO.OriginalGWO",
    "PSO.OriginalPSO",
    # MUST
    "OOA.OriginalOOA",
    "SFOA.OriginalSFOA",
    "DOA.OriginalDOA",
    # EK
    "BeesA.OriginalBeesA",
    "HBA.OriginalHBA",
    "MPA.OriginalMPA",
    "AGTO.OriginalAGTO",
    "SA.OriginalSA",
)

# Tekrar sayısı (results/phase3/last/1 .. last/N)
N_PHASE3_REPEATS = 10

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
sample_matrix = shared.sample_matrix


def load_algorithms_by_full_names(full_names, all_algos):
    """Verilen Mealpy tam adlarını sırayla all_algos ile eşler."""
    by_name = {a["full_name"]: a for a in all_algos}
    missing = [n for n in full_names if n not in by_name]
    if missing:
        raise KeyError(
            "get_all_algorithms_v3 içinde bulunamayan adlar: " + ", ".join(missing)
        )
    return [by_name[n] for n in full_names]


def run_one_phase3_repeat(
    run_idx,
    full_matrix,
    algos_phase3,
    *,
    K3=90,
    K_beh=60,
    epoch=50,
    pop_size=30,
):
    """
    Tek bir tekrar: phase3_* csv, ranked_scores, FINAL_RECOMMENDATIONS, behavior_analysis.
    """
    run_dir = os.path.join(PHASE3_LAST_ROOT, str(run_idx))
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

    print(f"\n>>> Tekrar {run_idx}: DAVRANIŞ ANALİZİ")
    matrix_beh = sample_matrix(
        full_matrix, n_users=200, n_items=300, seed=seed + 1
    )
    init_beh = mkmeans_plus_plus_init(
        matrix_beh, K=K_beh, n_solutions=20, seed=seed + 2
    )
    run_behavior_analysis(
        algos_phase3,
        matrix_beh,
        K_beh,
        initial_solutions=init_beh,
        epoch=epoch,
        pop_size=20,
        save_path=run_dir,
    )

    info = f"""Sadece Aşama 3 — toplu tekrar
Klasör: results/phase3/last/{run_idx}/

Bu çalıştırmada üretilen dosyalar:
  - phase3_complete.csv, phase3_success.csv, phase3_failed.csv
  - phase3_partial.csv (her 10 algoritmada bir ara kayıt)
  - ranked_scores.csv (bileşik skorlar / sıralama)
  - FINAL_RECOMMENDATIONS.csv (üst 5 öneri)
  - behavior_analysis.csv

Kaynak: only-phase-3.py → PHASE3_FIXED_ALGORITHM_FULL_NAMES (20 seçilmiş)
Paralel Aşama 3 işçi: {PHASE3_PARALLEL_WORKERS} (ONLY_PHASE3_WORKERS ile değişir)
Tekrar: {run_idx} / {N_PHASE3_REPEATS}
"""
    with open(os.path.join(run_dir, "RUN_INFO.txt"), "w", encoding="utf-8") as f:
        f.write(info)

    print(f"Sonuçlar: results/phase3/last/{run_idx}/")
    return run_dir


if __name__ == "__main__":
    print("=" * 60)
    print("SADECE AŞAMA 3 — ÇOKLU TEKRAR")
    print(f"Çıktı kökü: results/phase3/last/1 .. {N_PHASE3_REPEATS}")
    print(
        f"Aşama 3 paralellik: {PHASE3_PARALLEL_WORKERS} süreç "
        f"(sıralı için ONLY_PHASE3_WORKERS=1)"
    )
    print("=" * 60)

    full_matrix = load_movielens(
        os.path.join(os.path.dirname(BASE_DIR), "data", "ml-100k", "u.data")
    )
    # Aşama 1/2 yok: sabit 20'li liste + mealpy sınıf eşlemesi
    all_algos = get_all_algorithms_v3()
    algos_phase3 = load_algorithms_by_full_names(
        PHASE3_FIXED_ALGORITHM_FULL_NAMES, all_algos
    )

    print(f"Sabit alt küme → {len(algos_phase3)} algoritma:")
    for a in algos_phase3:
        print(f"  {a['full_name']}")

    os.makedirs(PHASE3_LAST_ROOT, exist_ok=True)

    for run_idx in range(1, N_PHASE3_REPEATS + 1):
        print("\n" + "=" * 60)
        print(f"TEKRAR {run_idx} / {N_PHASE3_REPEATS}")
        print("=" * 60)
        run_one_phase3_repeat(run_idx, full_matrix, algos_phase3)

    print("\n" + "=" * 60)
    print("TÜM TEKRARLAR BİTTİ")
    print(f"Kök: {PHASE3_LAST_ROOT}")
    print("=" * 60)
