#!/usr/bin/env python
"""
Scan installed mealpy: collect class docstrings (Notes, hyper-parameter hints, __init__ Args).
Writes CSV and Excel under mealpy/results/.
"""
from __future__ import annotations

import csv
import importlib
import inspect
import os
import pkgutil
import re
import sys
from typing import Any

import mealpy
from mealpy.optimizer import Optimizer

# --- output paths (relative to this file) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "results")
CSV_PATH = os.path.join(OUT_DIR, "mealpy_class_notes.csv")
XLSX_PATH = os.path.join(OUT_DIR, "mealpy_class_notes.xlsx")
MEALPY_ROOT = os.path.dirname(os.path.abspath(mealpy.__file__))

SKIP_PREFIXES = ("mealpy.utils", "mealpy.tests")


def _iter_mealpy_modules():
    for mod in pkgutil.walk_packages(mealpy.__path__, mealpy.__name__ + "."):
        name = mod.name
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        try:
            yield importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 — best effort over whole package
            print(f"[skip import] {name}: {e}", file=sys.stderr)


def _optimizer_subclasses(module) -> list[type]:
    out: list[type] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, Optimizer) and obj is not Optimizer:
                out.append(obj)
        except TypeError:
            pass
    return sorted(out, key=lambda c: c.__name__)


def _extract_notes_block(doc: str) -> str:
    if not doc:
        return ""
    # "Notes:\n    ..." (colon style)
    m = re.search(
        r"(?ms)^\s*Notes\s*:\s*\n(.*?)(?=^\s*(?:Examples|References|Hyper-parameters|Links)\b|\Z)",
        doc,
    )
    if m:
        return m.group(1).strip()
    # reST "Notes\n~~~~~\n"
    m = re.search(
        r"(?ms)^\s*Notes\s*\n\s*~+\s*\n(.*?)(?=^\s*(?:Examples|References|Hyper-parameters|Links)\b|\Z)",
        doc,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_hyperparam_block(doc: str) -> str:
    if not doc:
        return ""
    m = re.search(
        r"(?ms)(^\s*Hyper-parameters\b.*$.*?)(?=^\s*(?:Examples|References|Notes|Links)\b|\Z)",
        doc,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_init_args(doc: str) -> str:
    if not doc:
        return ""
    m = re.search(r"(?ms)^\s*Args\s*:\s*\n(.*)", doc)
    if not m:
        return ""
    rest = m.group(1)
    # stop at next major section
    m2 = re.search(
        r"(?ms)^(.*?)(?=^\s*(?:Returns|Raises|Examples|Note)\s*:|^\s*Note\s*$|\Z)",
        rest,
    )
    return (m2.group(1) if m2 else rest).strip()


def _flags(text: str) -> dict[str, bool]:
    t = text.lower()
    return {
        "flag_weak_or_poor": bool(
            re.search(r"\bweak\b|not good|poor performance|struggle", t)
        ),
        "flag_unclear_paper_matlab": bool(
            re.search(
                r"unclear|difficult to read|lacking in meaning|meaningful purpose|"
                r"differs slightly|matlab code|compare the matlab",
                t,
            )
        ),
        "flag_author_concern": bool(
            re.search(r"concerning|further investigation|verify the benchmark|reliability", t)
        ),
        "flag_hyperparam_tune": bool(
            re.search(r"hyper-parameters should fine-tune|fine-tune in approximate range", t)
        ),
        "flag_positive_paper_match": bool(
            re.search(
                r"exactly as described in the paper|matches the paper|as the best value in the paper",
                t,
            )
        ),
    }


def _row(cls: type) -> dict[str, Any]:
    mod = cls.__module__
    src = inspect.getsourcefile(cls) or ""
    rel_src = os.path.relpath(src, MEALPY_ROOT) if src.startswith(MEALPY_ROOT) else src

    cdoc = inspect.getdoc(cls) or ""
    init_doc = ""
    if cls.__init__ is not object.__init__:
        init_doc = inspect.getdoc(cls.__init__) or ""

    notes = _extract_notes_block(cdoc)
    hyper_class = _extract_hyperparam_block(cdoc)
    init_args = _extract_init_args(init_doc)

    combined_for_flags = "\n\n".join(
        x for x in (cdoc, init_doc) if x
    )
    f = _flags(combined_for_flags)

    first_line = (cdoc.splitlines() or [""])[0].strip()

    return {
        "module": mod,
        "class_name": cls.__name__,
        "source_file_under_mealpy": rel_src.replace("\\", "/"),
        "title_first_line": first_line,
        "has_notes_section": bool(notes),
        "notes_text": notes,
        "has_hyperparameters_in_class_doc": bool(hyper_class),
        "hyperparameters_class_doc": hyper_class,
        "has_init_args_doc": bool(init_args),
        "init_args_doc": init_args,
        "full_class_docstring": cdoc,
        **f,
        "mealpy_version": getattr(mealpy, "__version__", ""),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    seen: set[type] = set()
    rows: list[dict[str, Any]] = []

    for module in _iter_mealpy_modules():
        for cls in _optimizer_subclasses(module):
            if cls in seen:
                continue
            seen.add(cls)
            rows.append(_row(cls))

    rows.sort(key=lambda r: (r["module"], r["class_name"]))
    if not rows:
        print("No Optimizer subclasses found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {CSV_PATH}")

    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_excel(XLSX_PATH, index=False)
        print(f"Wrote -> {XLSX_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"Excel skipped ({e}); install pandas+openpyxl if you need .xlsx", file=sys.stderr)


if __name__ == "__main__":
    main()
