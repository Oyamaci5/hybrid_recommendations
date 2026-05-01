import json
import csv
import numpy as np


def save_json(path: str, obj: object) -> None:
    """NumPy tiplerini serileştirilebilir yapıya döndürür."""
    def _cv(o):
        if isinstance(o, dict):
            return {k: _cv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_cv(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    data = _cv(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv_rows(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Sözlük listesini tek bir CSV dosyasına yazar."""
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fnames = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows)


def save_results(obj: object, path) -> None:
    save_json(str(path), obj)
