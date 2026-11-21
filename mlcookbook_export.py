"""Export verified movement windows as split-preserving tabular features."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wearable_pipeline import IMU_COLUMNS, sha256, utc_now, write_json

SPLITS = ("train", "validation", "test")
STATISTICS = ("mean", "std", "min", "max", "rms", "median", "iqr", "mean_abs_delta")
LIMITATION = (
    "Use train.csv for exploratory tabular analysis. Open export.json in the standalone Movement Study "
    "app or run wearable.py evaluate for group-disjoint evaluation using the recorded splits. "
    "Use DEEPCRAFT for temporal CNN training and MCU C export."
)


def window_features(values, axes):
    """Physical-unit summary features; no fitted normalization or label inputs."""
    signals = [(name, values[:, :, i]) for i, name in enumerate(axes)]
    signals.append(("accel_mag", np.linalg.norm(values[:, :, :3], axis=2)))
    if len(axes) == 6:
        signals.append(("gyro_mag", np.linalg.norm(values[:, :, 3:], axis=2)))
    names, columns = [], []
    for name, signal in signals:
        summaries = (
            signal.mean(axis=1), signal.std(axis=1), signal.min(axis=1),
            signal.max(axis=1), np.sqrt(np.mean(signal ** 2, axis=1)),
            np.median(signal, axis=1),
            np.percentile(signal, 75, axis=1) - np.percentile(signal, 25, axis=1),
            np.abs(np.diff(signal, axis=1)).mean(axis=1),
        )
        names.extend(f"{name}_{statistic}" for statistic in STATISTICS)
        columns.extend(summaries)
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite window features")
    return names, features


def export_mlcookbook(dataset, output):
    """Verify prepared artifacts, preserve split membership, write a new export."""
    dataset, output = Path(dataset), Path(output)
    report = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("group_by") not in ("subject", "session_id"):
        raise ValueError("unsupported prepared dataset schema or grouping")
    synthetic = report.get("synthetic", False)
    if type(synthetic) is not bool:
        raise ValueError("invalid synthetic marker in prepared dataset")
    if any(type(item.get("synthetic", False)) is not bool or item.get("synthetic", False) != synthetic
           for item in report.get("sessions", [])):
        raise ValueError("session synthetic markers disagree with the prepared dataset")
    for name in ("windows.npz", "windows.csv"):
        if sha256(dataset / name) != report.get("artifacts_sha256", {}).get(name):
            raise ValueError(f"checksum mismatch: {name}")
    with np.load(dataset / "windows.npz", allow_pickle=False) as data:
        if not {"X", "y", "split", "session_id", "labels", "features"}.issubset(data.files):
            raise ValueError("missing prepared tensor arrays")
        values = np.asarray(data["X"], dtype=np.float64)
        y, splits, sessions = data["y"], data["split"], data["session_id"]
        labels, axes = data["labels"].tolist(), data["features"].tolist()
    units = ["g"] * 3 + (["deg/s"] * 3 if len(axes) == 6 else [])
    if (len(axes) not in (3, 6) or axes != IMU_COLUMNS[:len(axes)]
            or axes != report.get("features") or report.get("units") != units
            or report.get("normalization", {}).get("applied") is not False):
        raise ValueError("expected unnormalized ax/ay/az in g, optionally gx/gy/gz in deg/s")
    if (values.ndim != 3 or values.shape[0] == 0 or values.shape[1] < 2
            or values.shape[1:] != (report.get("window_samples"), len(axes))
            or not np.isfinite(values).all()):
        raise ValueError("invalid window tensor")
    n = len(values)
    if (y.shape != (n,) or splits.shape != (n,) or sessions.shape != (n,)
            or not np.issubdtype(y.dtype, np.integer) or not labels
            or any(not isinstance(label, str) or not label or label in ("unknown", "unlabeled") for label in labels)
            or len(set(labels)) != len(labels) or labels != report.get("labels")
            or int(y.min()) < 0 or int(y.max()) >= len(labels)
            or set(splits.tolist()) != set(SPLITS)):
        raise ValueError("invalid labels or split arrays")
    with (dataset / "windows.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    if len(index) != n:
        raise ValueError("window index length mismatch")
    group_splits, session_splits, seen_windows = {}, {}, set()
    for i, row in enumerate(index):
        split, session, label = str(splits[i]), str(sessions[i]), labels[int(y[i])]
        if (row.get("session_id"), row.get("split"), row.get("label")) != (session, split, label):
            raise ValueError("window index does not match tensor rows")
        group = row.get(report["group_by"])
        if not group or group == "unknown" or report.get("group_splits", {}).get(group) != split:
            raise ValueError("window group does not match recorded split")
        if group_splits.setdefault(group, split) != split or session_splits.setdefault(session, split) != split:
            raise ValueError("a group or session occurs in multiple splits")
        try:
            start, end, run = int(row["start_ts_ms"]), int(row["end_ts_ms"]), int(row["run"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid window provenance") from exc
        key = (session, run, start, end)
        if run < 0 or start < 0 or end <= start or key in seen_windows:
            raise ValueError("invalid or duplicate window provenance")
        seen_windows.add(key)
    for split in SPLITS:
        counts = dict(Counter(labels[int(y[i])] for i in range(n) if splits[i] == split))
        if set(counts) != set(labels) or counts != report.get("class_window_counts", {}).get(split):
            raise ValueError("class coverage differs from prepared dataset")
    names, features = window_features(values, axes)
    # Validate completely before creating output; never overwrite an earlier export.
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        positions = [i for i in range(n) if splits[i] == split]
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(names + ["label"])
            writer.writerows([*features[i].tolist(), labels[int(y[i])]] for i in positions)
        with (output / f"{split}_index.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source_window"] + list(index[0]))
            writer.writeheader()
            writer.writerows(dict(source_window=i, **index[i]) for i in positions)
    (output / "README.md").write_text(
        "# Movement Study features\n\n"
        "Import `train.csv`; target = `label`; predictors = all remaining columns.\n"
        "Use statistics/PCA to explore training data. Model scores are exploratory.\n\n"
        + LIMITATION + "\n\n"
        "Features summarize each prepared window: per-axis and vector-magnitude "
        "mean, population std, min, max, RMS, median, IQR and mean absolute adjacent "
        "sample difference. No normalization is fitted. Acceleration is in g and "
        "gyroscope values in deg/s; adjacent differences are not per-second derivatives.\n\n"
        "Predictor files exclude IDs, timestamps, split membership, model predictions "
        "and auxiliary sensors. Companion index files retain source-window lineage. "
        "Do not use index columns as model inputs or combine the reserved splits.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1, "created_utc": utc_now(), "target": "label",
        "synthetic": synthetic,
        "features": names, "statistics": list(STATISTICS),
        "source_axes": axes, "source_units": units,
        "window_samples": report["window_samples"], "stride_samples": report["stride_samples"],
        "sample_rate_hz": report["sample_rate_hz"],
        "group_by": report["group_by"], "group_splits": report["group_splits"],
        "class_window_counts": report["class_window_counts"], "labels": labels,
        "source_dataset_sha256": sha256(dataset / "dataset.json"),
        "source_artifacts_sha256": {name: report["artifacts_sha256"][name]
                                    for name in ("windows.npz", "windows.csv")},
        "limitation": LIMITATION,
        "artifacts_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    write_json(output / "export.json", manifest)
    return manifest
