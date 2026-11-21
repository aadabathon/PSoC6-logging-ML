"""Auditable IMU sessions and leakage-safe datasets. No board required for import.

Capture uses only the standard library (plus pyserial for UART); NumPy is
loaded only when preparing a dataset. Physical source units are never changed
in samples.csv. The Deepcraft export converts acceleration to g explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LEGACY_COLUMNS = "ts_ms,baro_pa,baro_t,ax,ay,az,gx,gy,gz,imu_t,mx,my,mz,mag_t,radar_present,radar_bin,mic_rms,mic_peak,label".split(",")
CSV_COLUMNS = LEGACY_COLUMNS[:-1] + ["ml_class", "ml_conf", "label"]
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
SAMPLE_COLUMNS = CSV_COLUMNS + ["segment_id"]
LABEL_RE = re.compile(r"[A-Za-z0-9_-]{1,15}\Z")
SCHEMA_VERSION = 1
GRAVITY = 9.80665


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows indexers/scanners can briefly hold a newly written manifest.
    # Retry only sharing/lock violations; permission and storage errors surface.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == 5:
                raise
            time.sleep(0.01 * (2 ** attempt))


class ProtocolError(ValueError):
    pass


class RowParser:
    """Accept both shipped wire schemas; reject corrupt data without relabeling."""

    def __init__(self, expected_label=None):
        self.header = None
        self.expected_label = expected_label

    def parse(self, line):
        try:
            fields = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            raise ProtocolError("invalid CSV") from exc
        if not fields:
            return None
        if fields[0] == "ts_ms":
            if fields not in (LEGACY_COLUMNS, CSV_COLUMNS):
                raise ProtocolError("unsupported wire header")
            self.header = fields
            return None
        if not re.match(r"^[+-]?\d", fields[0]):
            return None  # CLI acknowledgements / diagnostics retained in raw.log
        header = self.header
        if header is None:  # a TCP client can attach after the header was sent
            header = {len(LEGACY_COLUMNS): LEGACY_COLUMNS, len(CSV_COLUMNS): CSV_COLUMNS}.get(len(fields))
        if header is None or len(fields) != len(header):
            raise ProtocolError("wrong column count")
        row = dict(zip(header, fields))
        try:
            timestamp = int(row["ts_ms"])
            if not 0 <= timestamp < 2**32:
                raise ValueError()
            for name in IMU_COLUMNS:
                if not math.isfinite(float(row[name])):
                    raise ValueError()
            for name, value in row.items():
                if name not in ("ts_ms", "label", "ml_class") and value:
                    if not math.isfinite(float(value)):
                        raise ValueError()
        except ValueError as exc:
            raise ProtocolError("invalid timestamp or nonfinite/missing sensor value") from exc
        if not LABEL_RE.fullmatch(row["label"]):
            raise ProtocolError("invalid label")
        if self.expected_label is not None and row["label"] != self.expected_label:
            raise ProtocolError(f"label mismatch: expected {self.expected_label}, received {row['label']}")
        return {name: row.get(name, "") for name in CSV_COLUMNS}


class SessionWriter:
    """Append raw bytes before parsing; checkpoint metadata and fsync every 100 rows.

    A terminated process leaves status=recording, which prepare refuses. A
    transport error leaves status=failed and preserves all evidence for import.
    """

    def __init__(self, root, metadata, expected_label=None):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
        self.path = Path(root) / self.session_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.raw = (self.path / "raw.log").open("xb")
        self.samples = (self.path / "samples.csv").open("x", encoding="utf-8", newline="")
        self.rejected = (self.path / "rejected.jsonl").open("x", encoding="utf-8")
        self.events = (self.path / "events.csv").open("x", encoding="utf-8", newline="")
        self.event_writer = csv.writer(self.events)
        self.event_writer.writerow(["state", "ts_ms", "source_ts_ms", "confidence"])
        self.writer = csv.DictWriter(self.samples, fieldnames=SAMPLE_COLUMNS)
        self.writer.writeheader()
        self.parser = RowParser(expected_label)
        self.counts = Counter()
        self.segment = 0
        self.previous_ts = None
        self.buffer = bytearray()
        self.discarding = False
        self.closed = False
        self.metadata = dict(metadata, schema_version=SCHEMA_VERSION, session_id=self.session_id,
                             started_utc=utc_now(), status="recording", expected_label=expected_label,
                             source_units={"acceleration": "m/s^2", "gyroscope": "deg/s"})
        self.checkpoint()

    def checkpoint(self):
        for stream in (self.raw, self.samples, self.rejected, self.events):
            stream.flush()
            os.fsync(stream.fileno())
        self.metadata["counts"] = dict(self.counts)
        write_json(self.path / "session.json", self.metadata)

    def reject(self, reason, line):
        self.counts["rejected"] += 1
        self.segment += 1
        self.previous_ts = None
        self.rejected.write(json.dumps({"reason": reason, "line": line}) + "\n")

    def feed(self, data):
        self.raw.write(data)
        # Bounded framing handles partial serial reads and arbitrary TCP chunks.
        for byte in data:
            if byte == 10:
                if not self.discarding:
                    self.line(bytes(self.buffer).rstrip(b"\r"))
                self.buffer.clear()
                self.discarding = False
            elif not self.discarding:
                self.buffer.append(byte)
                if len(self.buffer) > 4096:
                    self.reject("line exceeds 4096 bytes", "<see raw.log>")
                    self.buffer.clear()
                    self.discarding = True

    def line(self, data):
        self.counts["lines"] += 1
        try:
            line = data.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            self.reject("invalid UTF-8", data.hex())
            return
        # Repeated headers and flash session markers must split windows even
        # when the timestamp and label happen to remain continuous.
        if line.startswith("ts_ms,") or line.startswith("[flash] session"):
            self.segment += 1
            self.previous_ts = None
        event = re.fullmatch(r"\[fall\] state=(ALERT|CANCEL|TEST) ts_ms=(\d+) source_ts_ms=(\d+) confidence=([0-9.]+)", line)
        if event:
            state, timestamp, source_timestamp, confidence = event.groups()
            try:
                valid = (0 <= float(confidence) <= 1 and int(timestamp) < 2**32 and int(source_timestamp) < 2**32)
            except ValueError:
                valid = False
            if not valid:
                self.reject("invalid fall event", line)
                return
            self.event_writer.writerow([state, timestamp, source_timestamp, confidence])
            self.counts["fall_events"] += 1
            self.checkpoint()  # preserve rare events without waiting for 100 samples
            return
        try:
            row = self.parser.parse(line)
        except ProtocolError as exc:
            self.reject(str(exc), line)
            if "header" in str(exc) or "label mismatch" in str(exc):
                raise  # configuration errors must never look like successful capture
            return
        if row is None:
            self.counts["diagnostics"] += 1
            return
        timestamp = int(row["ts_ms"])
        if self.previous_ts is not None:
            dt = timestamp - self.previous_ts
            if dt <= 0:
                self.counts["clock_resets_or_duplicates"] += 1
                self.segment += 1
            elif not 10 <= dt <= 30:
                self.counts["timing_breaks"] += 1
                self.segment += 1
        self.previous_ts = timestamp
        row["segment_id"] = self.segment
        self.writer.writerow(row)
        self.counts["rows"] += 1
        if self.counts["rows"] % 100 == 0:
            self.checkpoint()

    def close(self, status="complete", error=None):
        if self.closed:
            return
        if self.buffer or self.discarding:
            self.reject("incomplete final line", "<see raw.log>")
        if not self.counts["rows"]:
            status = "failed"
            error = error or "no valid IMU rows received"
        self.metadata.update(status=status, ended_utc=utc_now(), error=error)
        try:
            self.checkpoint()
        finally:
            for stream in (self.raw, self.samples, self.rejected, self.events):
                stream.close()
            self.closed = True
        self.metadata["sha256"] = {name: sha256(self.path / name) for name in
                                  ("raw.log", "samples.csv", "rejected.jsonl", "events.csv")}
        write_json(self.path / "session.json", self.metadata)


def import_recording(source, root, metadata, expected_label=None):
    session = SessionWriter(root, dict(metadata, transport="file", source_name=Path(source).name), expected_label)
    try:
        with Path(source).open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                session.feed(chunk)
        # A text export may omit a newline on its last complete row.
        if session.buffer and not session.discarding:
            session.line(bytes(session.buffer).rstrip(b"\r"))
            session.buffer.clear()
        session.close("imported")
    except BaseException as exc:
        session.close("failed", str(exc))
        raise
    if session.metadata["status"] == "failed":
        raise ValueError(session.metadata["error"])
    return session.path


def load_session(path):
    path = Path(path)
    metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema: {path}")
    if metadata.get("status") not in ("complete", "interrupted", "imported"):
        raise ValueError(f"session is not finalized: {path}")
    for name in ("raw.log", "samples.csv", "rejected.jsonl", "events.csv"):
        if sha256(path / name) != metadata.get("sha256", {}).get(name):
            raise ValueError(f"checksum mismatch: {path / name}")
    with (path / "samples.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SAMPLE_COLUMNS:
            raise ValueError(f"invalid normalized schema: {path}")
        rows = list(reader)
    if not rows or len(rows) != metadata.get("counts", {}).get("rows"):
        raise ValueError(f"session row count mismatch: {path}")
    return metadata, rows


def assign_splits(groups, seed):
    """Split whole groups before windowing. Never silently fall back to rows."""
    ordered = sorted(set(groups), key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    if len(ordered) < 3:
        raise ValueError("need at least 3 distinct groups for train/validation/test; collect more subjects or explicitly use --group-by session_id")
    held_out = max(1, int(len(ordered) * 0.15))
    return {group: "test" if i < held_out else "validation" if i < 2 * held_out else "train"
            for i, group in enumerate(ordered)}


def prepare_dataset(paths, output, *, group_by="subject", seed=42, window=50, stride=30, channels=3, profile=None, exclude_failed=False):
    import numpy as np

    if window < 2 or stride < 1 or stride > window or channels not in (3, 6):
        raise ValueError("require window >= 2, 1 <= stride <= window, channels in {3, 6}")
    if group_by not in ("subject", "session_id"):
        raise ValueError("group_by must be subject or session_id")
    sessions, excluded = [], []
    for path in sorted(set(map(Path, paths))):
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if exclude_failed and meta.get("status") in ("failed","recording"):
            excluded.append({"session_id":meta.get("session_id"),"status":meta.get("status"),
                             "error":meta.get("error"),"manifest_sha256":sha256(path / "session.json")})
            continue
        sessions.append(load_session(path))
    if not sessions:
        raise ValueError("no sessions found")
    markers = [meta.get("synthetic", False) for meta, _ in sessions]
    if any(type(marker) is not bool for marker in markers):
        raise ValueError("synthetic must be a boolean in every session")
    if len(set(markers)) != 1:
        raise ValueError("synthetic and real sessions must be prepared as separate studies")
    if profile:
        if channels != profile["channels"]:
            raise ValueError("channel count must match the selected study profile")
        for meta, rows in sessions:
            if not meta.get("placement", "").startswith(profile["placement_prefix"]):
                raise ValueError("belt profile requires placement such as belt_front, belt_left, or belt_right")
            unexpected = {row["label"] for row in rows} - set(profile["allowed_labels"])
            if unexpected:
                raise ValueError(f"labels outside the study profile: {sorted(unexpected)}")
    ids = [meta["session_id"] for meta, _ in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate session IDs (copied session?)")
    source_hashes = [meta["sha256"]["raw.log"] for meta, _ in sessions]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("duplicate raw recordings; importing a recording twice would leak data")
    groups = [meta.get(group_by) for meta, _ in sessions]
    if any(not group or group == "unknown" for group in groups):
        raise ValueError(f"every session needs a known {group_by}")
    splits = assign_splits(groups, seed)
    features = IMU_COLUMNS[:channels]
    labels = sorted({row["label"] for _, rows in sessions for row in rows} - {"unlabeled", "unknown"})
    if not labels:
        raise ValueError("no ground-truth labels; predictions are never used as labels")
    label_ids = {label: i for i, label in enumerate(labels)}
    windows, indexes, tracks = [], [], []
    quality = []
    for meta, rows in sessions:
        if meta.get("sample_rate_hz") != 50:
            raise ValueError("this pipeline requires 50 Hz recordings; resample explicitly in a separate, versioned step")
        split = splits[meta[group_by]]
        runs = []
        run = []
        for row in rows:
            if run and (row["segment_id"] != run[-1]["segment_id"] or row["label"] != run[-1]["label"]
                        or not 10 <= int(row["ts_ms"]) - int(run[-1]["ts_ms"]) <= 30):
                runs.append(run)
                run = []
            run.append(row)
        if run:
            runs.append(run)
        accepted = 0
        for run_number, run in enumerate(runs):
            label = run[0]["label"]
            if label not in label_ids or len(run) < window:
                continue
            timestamps = np.asarray([int(row["ts_ms"]) for row in run], dtype=np.int64)
            values = np.asarray([[float(row[name]) for name in features] for row in run], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite data in normalized session")
            values[:, :3] /= GRAVITY
            starts = list(range(0, len(run) - window + 1, stride))
            # Reject a drifted run in BOTH exports. Otherwise Studio could
            # train on a run whose Python windows were all rejected.
            if any(abs(int(timestamps[start + window - 1] - timestamps[start]) - (window - 1) * 20) > 20
                   for start in starts):
                continue
            tracks.append((meta["session_id"], run_number, split, label, timestamps, values))
            for start in starts:
                stop = start + window
                windows.append(values[start:stop])
                indexes.append({"session_id": meta["session_id"], "subject": meta.get("subject", "unknown"),
                                "split": split, "label": label, "run": run_number,
                                "start_ts_ms": int(timestamps[start]), "end_ts_ms": int(timestamps[stop - 1])})
                accepted += 1
        raw_imu = np.asarray([[float(row[name]) for name in IMU_COLUMNS] for row in rows])
        deltas = np.diff([int(row["ts_ms"]) for row in rows])
        positive_deltas = deltas[deltas > 0]
        quality.append({"session_id": meta["session_id"], "counts": meta["counts"],
                        "runs": len(runs), "windows": accepted, "split": split,
                        "median_period_ms": float(np.median(positive_deltas)) if len(positive_deltas) else None,
                        "accel_peak_g": float(np.abs(raw_imu[:, :3]).max() / GRAVITY),
                        "gyro_peak_dps": float(np.abs(raw_imu[:, 3:]).max()),
                        "near_default_accel_limit_rows": int(np.any(np.abs(raw_imu[:, :3]) >= 0.99 * 4 * GRAVITY, axis=1).sum()),
                        "near_default_gyro_limit_rows": int(np.any(np.abs(raw_imu[:, 3:]) >= 0.99 * 2000, axis=1).sum())})
    if not windows:
        raise ValueError("no complete, continuously sampled labeled windows")
    counts = {split: Counter(item["label"] for item in indexes if item["split"] == split)
              for split in ("train", "validation", "test")}
    for split, count in counts.items():
        missing = set(labels) - set(count)
        if missing:
            raise ValueError(f"{split} lacks windows for {sorted(missing)}; collect those classes in more groups")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    X = np.stack(windows)
    y = np.asarray([label_ids[item["label"]] for item in indexes], dtype=np.int64)
    split_array = np.asarray([item["split"] for item in indexes])
    np.savez_compressed(output / "windows.npz", X=X, y=y, split=split_array,
                        session_id=np.asarray([item["session_id"] for item in indexes]),
                        labels=np.asarray(labels), features=np.asarray(features))
    with (output / "windows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(indexes[0]))
        writer.writeheader()
        writer.writerows(indexes)
    # Studio owns sliding windows. Export continuous runs, never pre-windowed
    # overlapping records, and keep the split assignment alongside the tracks.
    for session_id, run_number, split, label, timestamps, values in tracks:
        directory = output / "deepcraft" / split / session_id / f"run_{run_number:04d}"
        directory.mkdir(parents=True)
        times = (timestamps - timestamps[0]) / 1000.0
        with (directory / "data.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Time (seconds)"] + [f"CH{i}" for i in range(channels)])
            writer.writerows(zip(times, *values.T))
        with (directory / "label.label").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Time(Seconds)", "Length(Seconds)", "Label(string)", "Confidence(double)", "Comment(string)"])
            writer.writerow([0, float(times[-1]) + 0.02, label, 1, ""])
    training = X[split_array == "train"]
    report = {"schema_version": 1, "created_utc": utc_now(), "sample_rate_hz": 50,
              "synthetic": markers[0],
              "study_profile": profile,
              "window_samples": window, "stride_samples": stride, "features": features,
              "units": ["g"] * 3 + (["deg/s"] * 3 if channels == 6 else []),
              "labels": labels, "group_by": group_by, "seed": seed, "group_splits": splits,
              "excluded_sessions": excluded,
              "class_window_counts": counts, "quality": quality,
              "normalization": {"applied": False, "training_mean": training.mean(axis=(0, 1)).tolist(),
                                "training_std": training.std(axis=(0, 1)).tolist()},
              "sessions": [{"session_id": meta["session_id"], "raw_sha256": meta["sha256"]["raw.log"],
                            "synthetic": meta.get("synthetic", False),
                            "samples_sha256": meta["sha256"]["samples.csv"],
                            "placement": meta.get("placement"), "orientation": meta.get("orientation"),
                            "enclosure": meta.get("enclosure"), "firmware": meta.get("firmware")}
                           for meta, _ in sessions]}
    report["artifacts_sha256"] = {str(path.relative_to(output)).replace("\\", "/"): sha256(path)
                                  for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "dataset.json", report)
    return report
