"""
SQLite kayıt: assignments, wnmf_results, runs.
Tek kopya — mealpy ve wnmf ortak kullanır; DB: <repo>/results/assignment_experiments.sqlite

assignments satırı, çıktı klasöründeki .npy dosyalarıyla aynı üç diziyi BLOB olarak tutar:
assignments.npy, gray_sheep_mask.npy, lof_scores.npy (LOF yoksa boş float64 dizi).
"""
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_REPO_ROOT, 'results', 'assignment_experiments.sqlite')

# init_db: şema uyumsuzluğunda bu tablolarda satır varsa otomatik DROP yapılmaz (SchemaMigrationError).
_CORE_TABLES = ('runs', 'assignments', 'wnmf_results')


class SchemaMigrationError(RuntimeError):
    """assignments şeması kodla uyumlu değil; DB'de veri var, tablolar silinmedi."""


def _core_tables_have_rows(conn):
    for tbl in _CORE_TABLES:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (tbl,),
        ).fetchone():
            continue
        n = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
        if n:
            return True
    return False

# Disk: assignments.npy, gray_sheep_mask.npy, lof_scores.npy — aynı veri BLOB sütunlarda.
ASSIGNMENTS_COLUMN_NAMES = frozenset({
    'id', 'dataset', 'algo', 'k', 'preprocessing',
    'run_id', 'seed', 'wcss', 'gray_count', 'gray_ratio', 'lof_threshold',
    'n_users', 'cluster_min', 'cluster_max', 'cluster_avg',
    'assignments_npy', 'gray_mask_npy', 'lof_scores_npy', 'created_at',
})
_ASSIGNMENTS_NOTNULL_BLOBS = frozenset({
    'assignments_npy', 'gray_mask_npy', 'lof_scores_npy',
})

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    command         TEXT NOT NULL,
    dataset         TEXT,
    k               INTEGER,
    scenario        TEXT,
    preprocessing   TEXT,
    latent_dim      INTEGER,
    epochs_global   INTEGER,
    epochs_cluster  INTEGER,
    reg             REAL,
    lr              REAL,
    note            TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS assignments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset         TEXT NOT NULL,
    algo            TEXT NOT NULL,
    k               INTEGER NOT NULL,
    preprocessing   TEXT NOT NULL,
    run_id          INTEGER REFERENCES runs(id),
    seed            INTEGER,
    wcss            REAL,
    gray_count      INTEGER,
    gray_ratio      REAL,
    lof_threshold   REAL,
    n_users         INTEGER,
    cluster_min     INTEGER,
    cluster_max     INTEGER,
    cluster_avg     REAL,
    assignments_npy BLOB NOT NULL,
    gray_mask_npy   BLOB NOT NULL,
    lof_scores_npy  BLOB NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assignments_lookup
ON assignments(dataset, algo, k, preprocessing);

CREATE INDEX IF NOT EXISTS idx_assignments_run
ON assignments(run_id);

CREATE TABLE IF NOT EXISTS wnmf_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id       INTEGER REFERENCES assignments(id),
    run_id              INTEGER REFERENCES runs(id),
    scenario            TEXT,
    mae                 REAL,
    rmse                REAL,
    gray_mae            REAL,
    gray_rmse           REAL,
    white_mae           REAL,
    white_rmse          REAL,
    precision_at_10     REAL,
    recall_at_10        REAL,
    f1_at_10            REAL,
    ndcg_at_10          REAL,
    n_train             INTEGER,
    n_test              INTEGER,
    latent_dim          INTEGER,
    epochs_global       INTEGER,
    epochs_cluster      INTEGER,
    reg                 REAL,
    lr                  REAL,
    time_seconds        REAL,
    cv_fold             INTEGER,
    cv_n_splits         INTEGER,
    is_cv_mean          INTEGER DEFAULT 0,
    mean_mae            REAL,
    mean_rmse           REAL,
    fold_mae_values     TEXT,
    fold_rmse_values    TEXT,
    created_at          TEXT
);
"""


def _assignments_table_matches(conn):
    """Sütun adları ve üç BLOB için NOT NULL beklenen şema ile uyumlu mu?"""
    rows = conn.execute('PRAGMA table_info(assignments)').fetchall()
    if not rows:
        return False
    names = {r[1] for r in rows}
    if names != ASSIGNMENTS_COLUMN_NAMES:
        return False
    by_name = {r[1]: r for r in rows}
    for col in _ASSIGNMENTS_NOTNULL_BLOBS:
        info = by_name.get(col)
        if info is None or info[3] != 1:
            return False
    return True


def _recreate_all_tables(conn):
    conn.executescript(
        """
        PRAGMA foreign_keys=OFF;
        DROP TABLE IF EXISTS wnmf_results;
        DROP TABLE IF EXISTS assignments;
        DROP TABLE IF EXISTS runs;
        PRAGMA foreign_keys=ON;
        """
    )
    conn.executescript(SCHEMA)


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """
    - assignments şeması uyumluysa: CREATE IF NOT EXISTS / indeksler; mevcut veri korunur.
    - Uyumlu değilse ve runs/assignments/wnmf_results boşsa: tablolar silinip SCHEMA uygulanır.
    - Uyumlu değilse ama bu tablolarda satır varsa: SchemaMigrationError (veri silinmez).
    """
    with get_connection() as conn:
        if not _assignments_table_matches(conn):
            if _core_tables_have_rows(conn):
                raise SchemaMigrationError(
                    f'{DB_PATH}: assignments şeması kodla uyumsuz; '
                    f'{", ".join(_CORE_TABLES)} tablolarından en az birinde kayıt var. '
                    'Otomatik silme yapılmadı. Yedek alın, dosyayı silerek sıfırdan oluşturun '
                    'veya veriyi koruyan bir migrasyon ekleyin.'
                )
            _recreate_all_tables(conn)
        else:
            conn.executescript(SCHEMA)
        _ensure_wnmf_results_columns(conn)


def _ensure_wnmf_results_columns(conn):
    """CV metrikleri için geriye uyumlu kolon ekleme."""
    rows = conn.execute('PRAGMA table_info(wnmf_results)').fetchall()
    existing = {r[1] for r in rows}
    needed = {
        'cv_fold': 'ALTER TABLE wnmf_results ADD COLUMN cv_fold INTEGER',
        'cv_n_splits': 'ALTER TABLE wnmf_results ADD COLUMN cv_n_splits INTEGER',
        'is_cv_mean': 'ALTER TABLE wnmf_results ADD COLUMN is_cv_mean INTEGER DEFAULT 0',
        'mean_mae': 'ALTER TABLE wnmf_results ADD COLUMN mean_mae REAL',
        'mean_rmse': 'ALTER TABLE wnmf_results ADD COLUMN mean_rmse REAL',
        'fold_mae_values': 'ALTER TABLE wnmf_results ADD COLUMN fold_mae_values TEXT',
        'fold_rmse_values': 'ALTER TABLE wnmf_results ADD COLUMN fold_rmse_values TEXT',
        'ndcg_at_10': 'ALTER TABLE wnmf_results ADD COLUMN ndcg_at_10 REAL',
        # JOIN olmadan doğrudan sorgulanabilsin diye denormalize kolonlar
        'algo':          'ALTER TABLE wnmf_results ADD COLUMN algo TEXT',
        'dataset':       'ALTER TABLE wnmf_results ADD COLUMN dataset TEXT',
        'k':             'ALTER TABLE wnmf_results ADD COLUMN k INTEGER',
        'assign_suffix': 'ALTER TABLE wnmf_results ADD COLUMN assign_suffix TEXT',
        'knn':           'ALTER TABLE wnmf_results ADD COLUMN knn INTEGER',
        'similarity':    'ALTER TABLE wnmf_results ADD COLUMN similarity TEXT',
        'fold':          'ALTER TABLE wnmf_results ADD COLUMN fold INTEGER',
    }
    for col, sql in needed.items():
        if col not in existing:
            conn.execute(sql)


def start_run(
    command,
    dataset=None,
    k=None,
    scenario=None,
    preprocessing=None,
    latent_dim=None,
    epochs_global=None,
    epochs_cluster=None,
    reg=None,
    lr=None,
    note=None,
):
    """
    Yeni bir run kaydı oluştur, run_id döndür.
    Komut başında çağrılır.
    """
    cmd = command or ' '.join(sys.argv)
    now = datetime.now().isoformat()
    sql = """
    INSERT INTO runs
        (command, dataset, k, scenario, preprocessing,
         latent_dim, epochs_global, epochs_cluster,
         reg, lr, note, started_at, status)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'running')
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (
                cmd,
                dataset,
                k,
                scenario,
                preprocessing,
                latent_dim,
                epochs_global,
                epochs_cluster,
                reg,
                lr,
                note,
                now,
            ),
        )
        return cur.lastrowid


def finish_run(run_id, status='done'):
    """Run'ı tamamla."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, status=? WHERE id=?",
            (now, status, run_id),
        )


def get_last_run_id():
    """Son run_id'yi döndür."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row['id'] if row else None


def get_assignment_id(dataset, algo, k, preprocessing):
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id FROM assignments
               WHERE dataset=? AND algo=? AND k=? AND preprocessing=?
               ORDER BY wcss ASC LIMIT 1""",
            (dataset, algo, k, preprocessing),
        ).fetchone()
        return int(row['id']) if row else None


def save_assignment(
    dataset,
    algo,
    k,
    preprocessing,
    wcss,
    gray_count,
    gray_ratio,
    lof_threshold,
    n_users,
    cluster_min,
    cluster_max,
    cluster_avg,
    assignments_arr,
    gray_mask_arr,
    lof_scores_arr=None,
    run_id=None,
    seed=None,
):
    import io

    import numpy as np

    def arr_to_blob(arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    lof_blob = (
        arr_to_blob(lof_scores_arr)
        if lof_scores_arr is not None
        else arr_to_blob(np.array([], dtype=np.float64))
    )

    now = datetime.now().isoformat()
    sql = """
    INSERT INTO assignments
        (dataset, algo, k, preprocessing, run_id, seed,
         wcss, gray_count, gray_ratio, lof_threshold,
         n_users, cluster_min, cluster_max, cluster_avg,
         assignments_npy, gray_mask_npy, lof_scores_npy,
         created_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with get_connection() as conn:
        conn.execute(
            sql,
            (
                dataset,
                algo,
                k,
                preprocessing,
                run_id,
                seed,
                wcss,
                gray_count,
                gray_ratio,
                lof_threshold,
                n_users,
                cluster_min,
                cluster_max,
                cluster_avg,
                arr_to_blob(assignments_arr),
                arr_to_blob(gray_mask_arr),
                lof_blob,
                now,
            ),
        )
    print(
        f"  DB kayıt: {dataset}/{algo}/k={k}/prep={preprocessing} "
        f"WCSS={wcss:.2f} gray={gray_count} ({gray_ratio*100:.1f}%)"
    )


def load_assignment(
    dataset,
    algo,
    k,
    preprocessing='none',
    strategy='best_wcss',
    run_id=None,
):
    """
    strategy:
        'best_wcss' → en düşük WCSS'li assignment
        'latest'    → en son çalıştırılan
        'by_run'    → belirli run_id (run_id parametresi gerekli)
    """
    import io

    import numpy as np

    def blob_to_arr(blob):
        if blob is None:
            return None
        return np.load(io.BytesIO(blob))

    if strategy == 'by_run' and run_id is not None:
        sql = """
        SELECT * FROM assignments
        WHERE dataset=? AND algo=? AND k=?
              AND preprocessing=? AND run_id=?
        ORDER BY wcss ASC LIMIT 1
        """
        params = (dataset, algo, k, preprocessing, run_id)
    elif strategy == 'latest':
        sql = """
        SELECT * FROM assignments
        WHERE dataset=? AND algo=? AND k=? AND preprocessing=?
        ORDER BY created_at DESC LIMIT 1
        """
        params = (dataset, algo, k, preprocessing)
    else:
        sql = """
        SELECT * FROM assignments
        WHERE dataset=? AND algo=? AND k=? AND preprocessing=?
        ORDER BY wcss ASC LIMIT 1
        """
        params = (dataset, algo, k, preprocessing)

    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()

    if row is None:
        return None

    r = dict(row)
    return {
        'id': r['id'],
        'run_id': r['run_id'],
        'seed': r['seed'],
        'wcss': r['wcss'],
        'gray_count': r['gray_count'],
        'gray_ratio': r['gray_ratio'],
        'lof_threshold': r['lof_threshold'],
        'n_users': r['n_users'],
        'cluster_min': r['cluster_min'],
        'cluster_max': r['cluster_max'],
        'cluster_avg': r['cluster_avg'],
        'created_at': r['created_at'],
        'assignments': blob_to_arr(r['assignments_npy']),
        'gray_mask': blob_to_arr(r['gray_mask_npy']),
        'lof_scores': blob_to_arr(r['lof_scores_npy']),
    }


def get_assignment_stats(dataset, algo, k, preprocessing):
    """
    Aynı parametrelerle kaç run var, WCSS dağılımı nedir?
    """
    sql = """
    SELECT
        COUNT(*) as n_runs,
        MIN(wcss) as wcss_min,
        MAX(wcss) as wcss_max,
        AVG(wcss) as wcss_avg,
        MIN(created_at) as first_run,
        MAX(created_at) as last_run
    FROM assignments
    WHERE dataset=? AND algo=? AND k=? AND preprocessing=?
    """
    with get_connection() as conn:
        row = conn.execute(sql, (dataset, algo, k, preprocessing)).fetchone()
    return dict(row) if row else None


def list_available_assignments(where_sql="", params=()):
    sql = f"""
    SELECT dataset, algo, k, preprocessing,
           COUNT(*) as n_runs,
           MIN(wcss) as wcss_min,
           MAX(wcss) as wcss_max,
           AVG(wcss) as wcss_avg,
           MAX(created_at) as last_run
    FROM assignments
    {where_sql}
    GROUP BY dataset, algo, k, preprocessing
    ORDER BY dataset, k, preprocessing, wcss_min
    """
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def print_available_assignments(where_sql="", params=()):
    rows = list_available_assignments(where_sql, params)
    print(
        f"{'Dataset':<8} {'K':>4} {'Prep':<15} {'Algoritma':<18} "
        f"{'#Run':>5} {'BestWCSS':>9} {'WorstWCSS':>10} {'LastRun':>12}"
    )
    for r in rows:
        lr = r['last_run'] or ''
        lr_short = lr[:10] if lr else ''
        print(
            f"  {r['dataset']:<8} {r['k']:>4} {r['preprocessing']:<15} "
            f"{r['algo']:<18} {r['n_runs']:>5} "
            f"{r['wcss_min']:>9.2f} {r['wcss_max']:>10.2f} "
            f"{lr_short:>12}"
        )


def save_wnmf_result(
    dataset,
    algo,
    k,
    preprocessing,
    scenario,
    mae,
    rmse,
    gray_mae,
    gray_rmse,
    white_mae,
    white_rmse,
    precision_at_10,
    recall_at_10,
    f1_at_10,
    n_train,
    n_test,
    latent_dim,
    epochs_global,
    epochs_cluster,
    reg,
    lr,
    time_seconds,
    ndcg_at_10=None,
    cv_fold=None,
    cv_n_splits=None,
    is_cv_mean=0,
    mean_mae=None,
    mean_rmse=None,
    fold_mae_values=None,
    fold_rmse_values=None,
    run_id=None,
    assignment_id_override=None,
    assign_suffix=None,
    knn=None,
    similarity=None,
    fold=None,
):
    assignment_id = (
        assignment_id_override
        if assignment_id_override is not None
        else get_assignment_id(dataset, algo, k, preprocessing)
    )
    now = datetime.now().isoformat()
    sql = """
    INSERT INTO wnmf_results
        (assignment_id, run_id, scenario, mae, rmse, gray_mae, gray_rmse,
         white_mae, white_rmse, precision_at_10, recall_at_10, f1_at_10, ndcg_at_10,
         n_train, n_test, latent_dim, epochs_global, epochs_cluster,
         reg, lr, time_seconds, cv_fold, cv_n_splits, is_cv_mean,
         mean_mae, mean_rmse, fold_mae_values, fold_rmse_values, created_at,
         algo, dataset, k, assign_suffix, knn, similarity, fold)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with get_connection() as conn:
        conn.execute(
            sql,
            (
                assignment_id,
                run_id,
                scenario,
                mae,
                rmse,
                gray_mae,
                gray_rmse,
                white_mae,
                white_rmse,
                precision_at_10,
                recall_at_10,
                f1_at_10,
                ndcg_at_10,
                n_train,
                n_test,
                latent_dim,
                epochs_global,
                epochs_cluster,
                reg,
                lr,
                time_seconds,
                cv_fold,
                cv_n_splits,
                int(bool(is_cv_mean)),
                mean_mae,
                mean_rmse,
                fold_mae_values,
                fold_rmse_values,
                now,
                algo,
                dataset,
                k,
                assign_suffix,
                knn,
                similarity,
                fold,
            ),
        )


def _fmt_mae(x):
    if x is None:
        return '   n/a '
    return f'{x:7.4f}'


def _fmt_gray(x):
    if x is None:
        return '   n/a '
    return f'{x:8.4f}'


def _fmt_white(x):
    if x is None:
        return '   n/a   '
    return f'{x:9.4f}'


if __name__ == '__main__':
    init_db()
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'runs':
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, started_at, status, dataset, k, "
                "preprocessing, scenario, note, command "
                "FROM runs ORDER BY id DESC LIMIT 20"
            ).fetchall()
        print(f"\n{'='*80}")
        print('Son 20 Run')
        print(f"{'='*80}")
        for r in rows:
            note = f" [{r['note']}]" if r['note'] else ''
            k_disp = str(r['k']) if r['k'] is not None else '?'
            print(
                f"  #{r['id']:<4} {r['started_at'][:19]} "
                f"{r['status']:<8} "
                f"ds={r['dataset'] or '?':<6} "
                f"k={k_disp:<4} "
                f"prep={r['preprocessing'] or '?':<15}"
                f"{note}"
            )
            # Windows cp1254 konsolunda Unicode ok karakteri encode hatası verebiliyor.
            print(f"         -> {r['command'][:80]}")

    elif cmd == 'run':
        rid = int(sys.argv[2])
        with get_connection() as conn:
            run = conn.execute(
                "SELECT * FROM runs WHERE id=?", (rid,)
            ).fetchone()
            # Önce denormalize kolonlardan oku (JOIN gerekmez, assignment_id NULL olsa bile çalışır)
            results = conn.execute(
                """SELECT
                       COALESCE(r.algo, a.algo, '?')         AS algo,
                       COALESCE(r.assign_suffix, a.preprocessing, '?') AS suffix,
                       COALESCE(r.k, a.k, 0)                 AS k,
                       r.scenario,
                       r.similarity,
                       r.knn,
                       r.fold,
                       r.mae, r.rmse,
                       r.gray_mae, r.white_mae,
                       r.precision_at_10, r.recall_at_10, r.ndcg_at_10,
                       r.cv_fold, r.time_seconds
                   FROM wnmf_results r
                   LEFT JOIN assignments a ON a.id = r.assignment_id
                   WHERE r.run_id=?
                   ORDER BY r.algo, r.mae""",
                (rid,),
            ).fetchall()
        if run:
            print(f"\nRun #{rid}")
            print(f"  Komut    : {run['command']}")
            print(f"  Basladı  : {run['started_at']}")
            print(f"  Bitti    : {run['finished_at'] or 'devam ediyor'}")
            print(f"  Durum    : {run['status']}")
            if run['note']:
                print(f"  Not      : {run['note']}")
            print(f"\n  Sonuclar ({len(results)} satir):")
            print(
                f"  {'Algo':<18} {'K':>3} {'Senaryo':<22} {'Sim':<12} {'KNN':>4} "
                f"{'MAE':>7} {'RMSE':>7} {'P@10':>6} {'R@10':>6} {'NDCG':>6} "
                f"{'Fold':>5} {'Sn':>6}"
            )
            print('  ' + '-' * 110)
            for r in results:
                fold_s = str(r['fold'] or r['cv_fold'] or '')
                sim_s = (r['similarity'] or '')[:11]
                knn_s = str(r['knn'] or '')
                print(
                    f"  {(r['algo'] or '?'):<18} {(r['k'] or 0):>3} "
                    f"{(r['scenario'] or ''):<22} {sim_s:<12} {knn_s:>4} "
                    f"{_fmt_mae(r['mae'])} {_fmt_mae(r['rmse'])} "
                    f"{_fmt_mae(r['precision_at_10'])} {_fmt_mae(r['recall_at_10'])} "
                    f"{_fmt_mae(r['ndcg_at_10'])} "
                    f"{fold_s:>5} {(r['time_seconds'] or 0):>6.1f}"
                )
        else:
            print(f"Run #{rid} bulunamadi.", file=sys.stderr)
            sys.exit(1)

    elif cmd == 'stats':
        # python assignment_db.py stats ml100k H4_MFO+HHO 30 none
        if len(sys.argv) >= 6:
            ds, algo, k, prep = (
                sys.argv[2],
                sys.argv[3],
                int(sys.argv[4]),
                sys.argv[5],
            )
            stats = get_assignment_stats(ds, algo, k, prep)
            if stats and stats['n_runs']:
                print(f"\n{ds}/{algo}/k={k}/prep={prep}")
                print(f"  Run sayısı : {stats['n_runs']}")
                print(f"  WCSS min   : {stats['wcss_min']:.4f}")
                print(f"  WCSS max   : {stats['wcss_max']:.4f}")
                print(f"  WCSS ort   : {stats['wcss_avg']:.4f}")
                fr = stats['first_run'] or ''
                lr = stats['last_run'] or ''
                print(f"  İlk run    : {fr[:19]}")
                print(f"  Son run    : {lr[:19]}")
            else:
                print(f"Kayıt yok: {ds}/{algo}/k={k}/prep={prep}", file=sys.stderr)
        else:
            print(
                'Kullanım: python assignment_db.py stats <dataset> <algo> <k> <preprocessing>',
                file=sys.stderr,
            )
            sys.exit(2)

    elif cmd == 'results':
        # python assignment_db.py results [algo] [dataset]
        # Tüm wnmf_results satırlarını algo/dataset filtresiyle göster — JOIN gerektirmez
        algo_filter = sys.argv[2] if len(sys.argv) > 2 else None
        ds_filter   = sys.argv[3] if len(sys.argv) > 3 else None
        where_parts = []
        params_r = []
        if algo_filter:
            where_parts.append('r.algo LIKE ?')
            params_r.append(f'%{algo_filter}%')
        if ds_filter:
            where_parts.append('r.dataset=?')
            params_r.append(ds_filter)
        where_sql = ('WHERE ' + ' AND '.join(where_parts)) if where_parts else ''
        with get_connection() as conn:
            rows = conn.execute(
                f"""SELECT r.run_id,
                           COALESCE(r.algo, a.algo, '?')  AS algo,
                           COALESCE(r.dataset, a.dataset, '?') AS ds,
                           COALESCE(r.k, a.k, 0)          AS k,
                           r.scenario, r.similarity, r.knn, r.fold,
                           r.mae, r.rmse,
                           r.precision_at_10, r.ndcg_at_10,
                           r.assign_suffix, r.created_at
                    FROM wnmf_results r
                    LEFT JOIN assignments a ON a.id = r.assignment_id
                    {where_sql}
                    ORDER BY r.created_at DESC
                    LIMIT 200""",
                params_r,
            ).fetchall()
        print(
            f"\n  {'RunID':>6} {'Algo':<18} {'DS':<7} {'K':>3} "
            f"{'Senaryo':<22} {'Sim':<12} {'KNN':>4} {'Fold':>5} "
            f"{'MAE':>7} {'RMSE':>7} {'P@10':>6} {'NDCG':>6}  Suffix"
        )
        print('  ' + '-' * 115)
        for r in rows:
            fold_s = str(r['fold'] or '')
            sim_s  = (r['similarity'] or '')[:11]
            knn_s  = str(r['knn'] or '')
            suffix_s = (r['assign_suffix'] or '')[:35]
            print(
                f"  {(r['run_id'] or 0):>6} {(r['algo'] or '?'):<18} "
                f"{(r['ds'] or '?'):<7} {(r['k'] or 0):>3} "
                f"{(r['scenario'] or ''):<22} {sim_s:<12} {knn_s:>4} {fold_s:>5} "
                f"{_fmt_mae(r['mae'])} {_fmt_mae(r['rmse'])} "
                f"{_fmt_mae(r['precision_at_10'])} {_fmt_mae(r['ndcg_at_10'])}  "
                f"{suffix_s}"
            )
        print(f"\n  Toplam: {len(rows)} satir")

    else:
        print(
            'Kullanim: python assignment_db.py runs | run <id> | results [algo] [dataset] '
            '| stats <ds> <algo> <k> <prep>  (repo kokundan calistirin)'
        )
