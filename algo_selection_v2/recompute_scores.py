"""
Algoritma secimi v2 — temiz WCSS / Silhouette / Davies-Bouldin karsilastirmasi.

Eski karsilastirmadaki hatalari duzeltir:
  1. WCSS artik STANDART tanim: atanan merkeze kare Oklid uzakligi toplami (SSE).
     (Eski kod varsayilan olarak Pearson mesafesi topluyordu -> literatur WCSS'i degil.)
  2. Tum metrikler (WCSS, Sil, DB) AYNI uzayda ve AYNI metrikle hesaplanir.
     (Eski kod: WCSS=pearson, Sil=pearson-precomputed 300 orneklem, DB=euclid ham matris.)
  3. Silhouette orneklemi yok: 943 kullanici icin tam hesap, sabit sonuc.
     (Eski kod: seed'siz np.random.choice -> tekrar uretilemez.)
  4. Her algoritma icin N tekrar (farkli seed) -> mean±std + ortalama rank + Friedman.
     (Eski kod: tek kosu -> %1.5'lik WCSS farklari gurultuden ayirt edilemez.)
  5. kmref her iki modda da olculur: meta-sonrasi ham WCSS ve Lloyd-rafine WCSS.
     Literatur protokolu (Katarya, Thakrar, GOA): meta -> K-means init => kmref ACIK.
  6. Baseline'lar tabloda: KMeans++ (n_init=10) ve random-init KMeans.
     Meta bir algoritma KMeans++'i gecemiyorsa katkisi yoktur — secim kapisi budur.

Kullanim (repo kokunden):
  python algo_selection_v2/recompute_scores.py --quick                  # duman testi
  python algo_selection_v2/recompute_scores.py --runs 5                 # secim turu
  python algo_selection_v2/recompute_scores.py --runs 30 --epoch 100    # final
  python algo_selection_v2/recompute_scores.py --features path/to.npy   # WNMF uzayi

Cikti: algo_selection_v2/results/per_run.csv, summary.csv, friedman.txt
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 1) VERI YUKLEME  (kontrol: shape, rating araligi, duplicate, fold ayrikligi)
# --------------------------------------------------------------------------

def load_ml100k_train(data_dir: Path, fold: int = 1) -> np.ndarray:
    f = data_dir / f"u{fold}.base"
    df = pd.read_csv(f, sep="\t", names=["u", "i", "r", "t"])
    # --- zorunlu kontroller ---
    assert df["r"].between(1, 5).all(), "Rating 1-5 disinda deger var!"
    assert not df.duplicated(["u", "i"]).any(), "Duplicate (user,item) var!"
    test = pd.read_csv(data_dir / f"u{fold}.test", sep="\t", names=["u", "i", "r", "t"])
    overlap = pd.merge(df[["u", "i"]], test[["u", "i"]], on=["u", "i"])
    assert len(overlap) == 0, f"Train/test kesisimi bos degil: {len(overlap)} satir!"
    mat = np.zeros((943, 1682), dtype=np.float64)
    mat[df["u"] - 1, df["i"] - 1] = df["r"]
    print(f"[VERI OK] fold {fold}: {len(df)} rating, {len(test)} test, "
          f"sparsity={1 - len(df) / mat.size:.4f}")
    return mat


def build_features(train: np.ndarray, mode: str, dim: int,
                   npy_path: str | None) -> np.ndarray:
    if mode == "npy":
        X = np.load(npy_path)
        assert X.shape[0] == train.shape[0], "Ozellik satir sayisi != kullanici sayisi"
    elif mode == "svd":
        X = TruncatedSVD(n_components=dim, random_state=42).fit_transform(train)
    elif mode == "nmf":
        from sklearn.decomposition import NMF
        X = NMF(n_components=dim, init="nndsvda", max_iter=400,
                random_state=42).fit_transform(train)
    else:  # raw
        X = train.copy()
    # --- zorunlu kontroller ---
    assert np.isfinite(X).all(), "Ozellik matrisi NaN/inf iceriyor!"
    assert (X.std(axis=0) > 1e-12).all(), "Varyansi sifir ozellik boyutu var!"
    h = hashlib.md5(X.tobytes()).hexdigest()[:10]
    print(f"[OZELLIK OK] mode={mode} shape={X.shape} hash={h} "
          f"(tum algoritmalar ayni hash'i kullanmali)")
    return np.ascontiguousarray(X, dtype=np.float64)


# --------------------------------------------------------------------------
# 2) METRIKLER — hepsi ayni uzay, ayni metrik (Oklid)
# --------------------------------------------------------------------------

def sse_wcss(X: np.ndarray, centroids: np.ndarray) -> tuple[float, np.ndarray]:
    """Standart k-means SSE: atanan merkeze KARE Oklid uzakligi toplami."""
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(1)
    return float(d2[np.arange(len(X)), labels].sum()), labels


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    out = {"silhouette": np.nan, "davies_bouldin": np.nan,
           "n_clusters_used": int(len(np.unique(labels)))}
    if out["n_clusters_used"] >= 2:
        out["silhouette"] = float(silhouette_score(X, labels, metric="euclidean"))
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    return out


# --------------------------------------------------------------------------
# 3) MEALPY UYUMLULUK KATMANI (2.x ve 3.x)
# --------------------------------------------------------------------------

def discover_algorithms(names: list[str]) -> dict:
    import inspect
    import pkgutil

    import mealpy
    found: dict = {}
    wanted = set(names)
    for _, modname, _ in pkgutil.walk_packages(mealpy.__path__,
                                               mealpy.__name__ + ".",
                                               onerror=lambda x: None):
        try:
            module = __import__(modname, fromlist="x")
        except Exception:
            continue
        for cname, obj in inspect.getmembers(module, inspect.isclass):
            full = f"{modname.split('.')[-1]}.{cname}"
            if full in wanted:
                found[full] = obj
    missing = wanted - set(found)
    if missing:
        print(f"[UYARI] mealpy'de bulunamayan algoritmalar (atlanacak): {sorted(missing)}")
    return found


def solve_meta(algo_cls, obj_func, lb, ub, epoch, pop, seed, max_fe=None):
    """max_fe verilirse butce NFE ile esitlenir (adil kiyas); epoch tavana cekilir."""
    import mealpy
    major = int(str(getattr(mealpy, "__version__", "2")).split(".")[0])
    np.random.seed(seed)
    if max_fe:
        epoch = max(epoch, (max_fe * 3) // pop)  # epoch degil, max_fe durdursun
    model = algo_cls(epoch=epoch, pop_size=pop)
    if major >= 3:
        from mealpy import FloatVar
        prob = {"obj_func": obj_func, "bounds": FloatVar(lb=lb, ub=ub),
                "minmax": "min", "log_to": None}
        term = {"max_fe": int(max_fe)} if max_fe else None
        g = model.solve(prob, seed=seed, termination=term)
        best_pos, best_fit = np.array(g.solution), float(g.target.fitness)
    else:
        prob = {"fit_func": obj_func, "lb": list(lb), "ub": list(ub),
                "minmax": "min", "log_to": None}
        if max_fe:
            prob["termination"] = {"mode": "FE", "quantity": int(max_fe)}
        best_pos, best_fit = model.solve(prob)
        best_pos, best_fit = np.array(best_pos), float(best_fit)
    curve = list(getattr(model.history, "list_global_best_fit", []) or [])
    nfe = getattr(model, "nfe_counter", None)
    return best_pos, best_fit, curve, nfe


# --------------------------------------------------------------------------
# 4) TEK KOSU: meta -> (kontroller) -> kmref -> metrikler
# --------------------------------------------------------------------------

def run_one(name, algo_cls, X, K, epoch, pop, seed, global_sse,
            max_fe=None) -> dict:
    D = X.shape[1]
    lb = np.tile(X.min(0), K)  # K*D duzeni: [c1|c2|...]
    ub = np.tile(X.max(0), K)

    def obj(sol):
        c = np.asarray(sol).reshape(K, D)
        w, lab = sse_wcss(X, c)
        n_empty = K - len(np.unique(lab))
        return w + n_empty * (global_sse / K)  # bos kume cezasi

    t0 = time.time()
    pos, fit, curve, nfe = solve_meta(algo_cls, obj, lb, ub, epoch, pop, seed,
                                      max_fe=max_fe)
    t_meta = time.time() - t0

    # --- kosu ici kontroller ---
    if len(curve) > 1:
        assert all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1)), \
            f"{name}: fitness egrisi monoton degil (best-so-far artmis)!"
    cent = pos.reshape(K, D)
    wcss_meta, lab_meta = sse_wcss(X, cent)
    n_empty = K - len(np.unique(lab_meta))

    # --- kmref: meta centroid'leri Lloyd K-means'e init olarak ver ---
    km = KMeans(n_clusters=K, init=cent, n_init=1, random_state=seed).fit(X)
    wcss_ref, lab_ref = sse_wcss(X, km.cluster_centers_)
    assert wcss_ref <= wcss_meta + 1e-6, f"{name}: kmref WCSS'i artirdi (imkansiz)!"

    row = {"algorithm": name, "seed": seed, "time_s": round(t_meta, 1),
           "nfe": nfe,
           "wcss_meta": wcss_meta, "wcss_kmref": wcss_ref,
           "kmref_gain_pct": 100 * (wcss_meta - wcss_ref) / wcss_meta,
           "lloyd_iters": int(km.n_iter_), "empty_clusters_meta": n_empty}
    row.update({f"{k}_kmref": v for k, v in cluster_metrics(X, lab_ref).items()})
    row.update({f"{k}_meta": v for k, v in cluster_metrics(X, lab_meta).items()})
    return row


def run_baselines(X, K, seeds, global_sse) -> list[dict]:
    rows = []
    for seed in seeds:
        for bname, init, n_init in [("B0_KMEANS++", "k-means++", 10),
                                    ("B0_RANDOM", "random", 1)]:
            t0 = time.time()
            km = KMeans(n_clusters=K, init=init, n_init=n_init,
                        random_state=seed).fit(X)
            w, lab = sse_wcss(X, km.cluster_centers_)
            row = {"algorithm": bname, "seed": seed,
                   "time_s": round(time.time() - t0, 1),
                   "wcss_meta": np.nan, "wcss_kmref": w, "kmref_gain_pct": np.nan,
                   "lloyd_iters": int(km.n_iter_), "empty_clusters_meta": 0}
            row.update({f"{k}_kmref": v for k, v in cluster_metrics(X, lab).items()})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 5) OZET + FRIEDMAN
# --------------------------------------------------------------------------

def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    g = per_run.groupby("algorithm")
    s = g.agg(wcss_kmref_mean=("wcss_kmref", "mean"),
              wcss_kmref_std=("wcss_kmref", "std"),
              wcss_meta_mean=("wcss_meta", "mean"),
              sil_mean=("silhouette_kmref", "mean"),
              sil_std=("silhouette_kmref", "std"),
              db_mean=("davies_bouldin_kmref", "mean"),
              db_std=("davies_bouldin_kmref", "std"),
              time_mean=("time_s", "mean"),
              n_runs=("seed", "count")).reset_index()
    # rank: seed bazinda WCSS sirasi -> ortalama rank (dusuk iyi)
    ranks = (per_run.pivot_table(index="seed", columns="algorithm",
                                 values="wcss_kmref")
             .rank(axis=1).mean(0).rename("avg_rank_wcss"))
    s = s.merge(ranks, left_on="algorithm", right_index=True)
    return s.sort_values("avg_rank_wcss")


def friedman(per_run: pd.DataFrame, out: Path):
    piv = per_run.pivot_table(index="seed", columns="algorithm", values="wcss_kmref")
    piv = piv.dropna(axis=1)
    if piv.shape[1] >= 3 and piv.shape[0] >= 3:
        stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
        msg = (f"Friedman (WCSS_kmref, {piv.shape[0]} seed x {piv.shape[1]} algo): "
               f"chi2={stat:.3f}, p={p:.5f}\n"
               f"NOT: p<0.05 ise algoritmalar arasi fark anlamli -> post-hoc "
               f"(Holm duzeltmeli Wilcoxon) gerekir.\n"
               f"NOT: Anlamli fark YOKSA bu da sonuctur: 'bu uzayda metalar "
               f"esdeger, secim hiz/kararlilikla yapilir' denir.")
    else:
        msg = "Friedman icin yetersiz veri (>=3 seed ve >=3 algoritma gerekli)."
    out.write_text(msg, encoding="utf-8")
    print(msg)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "data" / "ml-100k"))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--feature-mode", choices=["raw", "svd", "nmf", "npy"],
                    default="svd")
    ap.add_argument("--resume", action="store_true",
                    help="per_run.csv'deki (algo,seed) ciftlerini atla ve ekle")
    ap.add_argument("--max-fe", type=int, default=None,
                    help="adil butce: tum algoritmalar bu NFE'de durur (orn 15000)")
    ap.add_argument("--out", default="per_run.csv",
                    help="cikti dosya adi (paralel kosularda cakismayi onler)")
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--features", default=None, help=".npy yolu (WNMF uzayi icin)")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--algos-file", default=str(ROOT / "algos.txt"))
    ap.add_argument("--quick", action="store_true", help="2 algo, 2 seed, epoch=5")
    args = ap.parse_args()

    names = [ln.strip() for ln in Path(args.algos_file).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    if args.quick:
        names, args.runs, args.epoch = names[:2], 2, 5
    if args.features:
        args.feature_mode = "npy"

    train = load_ml100k_train(Path(args.data_dir), args.fold)
    X = build_features(train, args.feature_mode, args.dim, args.features)
    global_sse = float(((X - X.mean(0)) ** 2).sum())
    print(f"[REFERANS] K=1 global SSE = {global_sse:.1f} "
          f"(tum WCSS'ler bunun altinda olmali)")

    algos = discover_algorithms(names)
    seeds = list(range(42, 42 + args.runs))

    done: set = set()
    rows: list[dict] = []
    per_run_path = RESULTS / args.out
    if args.resume and per_run_path.exists():
        old = pd.read_csv(per_run_path)
        rows = old.to_dict("records")
        done = set(zip(old["algorithm"], old["seed"]))
        print(f"[RESUME] {len(done)} tamamlanmis (algo,seed) atlanacak")

    if not any(a.startswith("B0") for a, _ in done):
        rows += run_baselines(X, args.k, seeds, global_sse)
    for i, (name, cls) in enumerate(algos.items(), 1):
        for seed in seeds:
            if (name, seed) in done:
                continue
            try:
                r = run_one(name, cls, X, args.k, args.epoch, args.pop, seed,
                            global_sse, max_fe=args.max_fe)
                print(f"[{i}/{len(algos)}] {name} seed={seed} "
                      f"WCSS_meta={r['wcss_meta']:.1f} -> kmref={r['wcss_kmref']:.1f} "
                      f"sil={r['silhouette_kmref']:.3f} nfe={r['nfe']} ({r['time_s']}s)")
                rows.append(r)
            except Exception as e:
                print(f"[{i}/{len(algos)}] {name} seed={seed} HATA: {e}")
                rows.append({"algorithm": name, "seed": seed, "error": str(e)})
        pd.DataFrame(rows).to_csv(per_run_path, index=False)  # ara kayit

    per_run = pd.DataFrame(rows)
    per_run.to_csv(per_run_path, index=False)
    ok = per_run[per_run.get("error").isna()] if "error" in per_run else per_run
    summ = summarize(ok)
    summ.to_csv(RESULTS / "summary.csv", index=False)
    friedman(ok, RESULTS / "friedman.txt")
    print("\n=== OZET (ilk 10) ===")
    print(summ.head(10).to_string(index=False))
    print(f"\nCikti: {RESULTS}")

    # --- secim kapisi kontrolu ---
    b0 = summ.loc[summ.algorithm == "B0_KMEANS++", "wcss_kmref_mean"]
    if len(b0):
        beat = summ[(summ.algorithm.str.startswith("B0") == False)
                    & (summ.wcss_kmref_mean < float(b0.iloc[0]))]
        print(f"\n[SECIM KAPISI] KMeans++'i gecen meta sayisi: {len(beat)}"
              f" / {len(summ) - 2}")
        if len(beat) == 0:
            print("  -> Hicbir meta KMeans++'i gecemedi: bu uzayda meta-init'in "
                  "katkisi yok demektir. Uzay/K/fitness yeniden ele alinmali.")


if __name__ == "__main__":
    main()
