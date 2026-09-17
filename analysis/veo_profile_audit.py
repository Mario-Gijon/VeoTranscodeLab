#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_STATIC_ATOL = 1e-9
DEFAULT_NEAR_STATIC_ATOL = 5e-3
_MISSING = object()


@dataclass
class FieldAudit:
    path: str
    category: str
    value_type: str
    count: int
    max_abs_delta: float | None = None
    mean_abs_delta: float | None = None
    max_rel_delta: float | None = None
    notes: str | None = None
    values: list[Any] | None = None
    transform_rotation_delta_deg: float | None = None
    transform_translation_delta: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def try_parse_numeric_csv(value: str) -> np.ndarray | None:
    if "," not in value:
        return None
    try:
        vals = [float(x.strip()) for x in value.split(",")]
    except ValueError:
        return None
    return np.asarray(vals, dtype=np.float64) if vals else None


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            out[prefix] = value
            return out
        for key, child in value.items():
            p = f"{prefix}.{key}" if prefix else key
            out.update(flatten(child, p))
        return out

    if isinstance(value, list):
        if value and all(is_numeric_scalar(x) for x in value):
            out[prefix] = value
            return out
        if not value:
            out[prefix] = value
            return out
        for idx, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{idx}]"))
        return out

    out[prefix] = value
    return out


def numeric_array(value: Any) -> np.ndarray | None:
    if is_numeric_scalar(value):
        return np.asarray([float(value)], dtype=np.float64)
    if isinstance(value, list) and all(is_numeric_scalar(x) for x in value):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        return try_parse_numeric_csv(value)
    return None


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a.T @ b
    c = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c))


def transform_deltas(arrays: list[np.ndarray]) -> tuple[float, float] | None:
    if not arrays or not all(a.size == 16 for a in arrays):
        return None
    mats = [a.reshape(4, 4) for a in arrays]
    ref = mats[0]
    max_rot = 0.0
    max_trans = 0.0
    for mat in mats[1:]:
        max_rot = max(max_rot, rotation_angle_deg(ref[:3, :3], mat[:3, :3]))
        max_trans = max(max_trans, float(np.linalg.norm(ref[:3, 3] - mat[:3, 3])))
    return max_rot, max_trans


def classify_numeric(
    path: str,
    values: list[Any],
    static_atol: float,
    near_static_atol: float,
) -> FieldAudit:
    arrays = [numeric_array(v) for v in values]
    if any(a is None for a in arrays):
        raise RuntimeError("Internal numeric classification error")
    arrays = [a for a in arrays if a is not None]

    if len({a.shape for a in arrays}) != 1:
        return FieldAudit(
            path=path,
            category="STRUCTURAL",
            value_type="numeric-shape-mismatch",
            count=len(values),
            values=values,
        )

    ref = arrays[0]
    abs_diffs = [np.abs(a - ref) for a in arrays[1:]]
    if abs_diffs:
        max_abs = max(float(np.max(d)) for d in abs_diffs)
        mean_abs = float(np.mean(np.concatenate([d.ravel() for d in abs_diffs])))
        max_rel = 0.0
        for a in arrays[1:]:
            denom = np.maximum(np.maximum(np.abs(ref), np.abs(a)), 1e-12)
            max_rel = max(max_rel, float(np.max(np.abs(a - ref) / denom)))
    else:
        max_abs = mean_abs = max_rel = 0.0

    if max_abs <= static_atol:
        category = "STATIC"
    elif max_abs <= near_static_atol:
        category = "NEAR_STATIC"
    else:
        category = "MATCH_SPECIFIC"

    if ref.size == 1:
        value_type = "numeric-scalar"
    elif ref.size == 16:
        value_type = "numeric-4x4"
    elif ref.size == 9:
        value_type = "numeric-3x3"
    else:
        value_type = f"numeric-vector[{ref.size}]"

    rot = trans = None
    notes = None
    td = transform_deltas(arrays)
    if td is not None:
        rot, trans = td
        notes = f"4x4 transform: rotation delta={rot:.6f} deg, translation delta={trans:.6f}"

    return FieldAudit(
        path=path,
        category=category,
        value_type=value_type,
        count=len(values),
        max_abs_delta=max_abs,
        mean_abs_delta=mean_abs,
        max_rel_delta=max_rel,
        notes=notes,
        values=values,
        transform_rotation_delta_deg=rot,
        transform_translation_delta=trans,
    )


def classify_field(
    path: str,
    values: list[Any],
    static_atol: float,
    near_static_atol: float,
) -> FieldAudit:
    if any(v is _MISSING for v in values):
        return FieldAudit(
            path=path,
            category="STRUCTURAL",
            value_type="missing",
            count=len(values),
            notes="Missing from one or more files",
            values=[None if v is _MISSING else v for v in values],
        )

    numeric = [numeric_array(v) for v in values]
    if all(v is not None for v in numeric):
        return classify_numeric(path, values, static_atol, near_static_atol)

    if all(v == values[0] for v in values[1:]):
        return FieldAudit(
            path=path,
            category="STATIC",
            value_type=type(values[0]).__name__,
            count=len(values),
            values=values,
        )

    return FieldAudit(
        path=path,
        category="MATCH_SPECIFIC",
        value_type=type(values[0]).__name__,
        count=len(values),
        notes="Non-numeric values differ",
        values=values,
    )


def audit_files(
    files: list[Path],
    static_atol: float,
    near_static_atol: float,
) -> list[FieldAudit]:
    docs = [flatten(load_json(path)) for path in files]
    keys = sorted(set().union(*(d.keys() for d in docs)))
    return [
        classify_field(
            key,
            [doc.get(key, _MISSING) for doc in docs],
            static_atol,
            near_static_atol,
        )
        for key in keys
    ]


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4:
        return f"{v:.3e}"
    return f"{v:.6f}"


def print_report(files: list[Path], audits: list[FieldAudit]) -> None:
    print("Veo profile audit")
    print("=================")
    for idx, path in enumerate(files, 1):
        print(f"[{idx}] {path}")
    print()

    for category in ("STATIC", "NEAR_STATIC", "MATCH_SPECIFIC", "STRUCTURAL"):
        rows = [a for a in audits if a.category == category]
        print(f"{category} ({len(rows)})")
        print("-" * (len(category) + len(str(len(rows))) + 3))
        for a in rows:
            extra = ""
            if a.max_abs_delta is not None:
                extra += f" | max_abs={fmt(a.max_abs_delta)} mean_abs={fmt(a.mean_abs_delta)}"
            if a.transform_rotation_delta_deg is not None:
                extra += (
                    f" | rot={a.transform_rotation_delta_deg:.6f} deg"
                    f" trans={a.transform_translation_delta:.6f}"
                )
            print(a.path + extra)
        print()

    if len(files) < 3:
        print(
            "NOTE: with fewer than three recordings, STATIC/NEAR_STATIC is only "
            "a provisional empirical classification, not proof of Veo semantics."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Veo .veo files and classify calibration fields."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--static-atol", type=float, default=DEFAULT_STATIC_ATOL)
    parser.add_argument("--near-static-atol", type=float, default=DEFAULT_NEAR_STATIC_ATOL)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("Provide at least two .veo files.")
    if args.near_static_atol < args.static_atol:
        parser.error("--near-static-atol must be >= --static-atol")

    missing = [str(p) for p in args.files if not p.is_file()]
    if missing:
        parser.error("Missing file(s): " + ", ".join(missing))

    audits = audit_files(args.files, args.static_atol, args.near_static_atol)
    print_report(args.files, audits)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "files": [str(p) for p in args.files],
            "thresholds": {
                "static_atol": args.static_atol,
                "near_static_atol": args.near_static_atol,
            },
            "warning": (
                "Classifications are empirical across the supplied files only "
                "and are not proof that a field is hardware-static."
            ),
            "fields": [asdict(a) for a in audits],
        }
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
