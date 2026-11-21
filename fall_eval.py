#!/usr/bin/env python3
"""Score fall alerts against annotated events and explicit monitored exposure.

Truth CSV: session_id,event_id,start_ms,end_ms
Alerts CSV: session_id,ts_ms,confidence
Coverage CSV: session_id,start_ms,end_ms (fully annotated monitored intervals)
Times are in the same session clock. Intervals are half-open [start, end).
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from wearable_pipeline import sha256, write_json


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("times and confidence must be finite and nonnegative")
    return result


def evaluate(truth, alerts, coverage, threshold=0.7, grace_ms=3000):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or not math.isfinite(grace_ms) or grace_ms < 0:
        raise ValueError("threshold must be in [0,1]; grace_ms must be finite and nonnegative")
    intervals = defaultdict(list)
    for row in coverage:
        start, end = number(row["start_ms"]), number(row["end_ms"])
        if not row["session_id"] or end <= start:
            raise ValueError("coverage needs a session and positive duration")
        intervals[row["session_id"]].append((start, end))
    for group in intervals.values():
        group.sort()
        if any(left[1] > right[0] for left, right in zip(group, group[1:])):
            raise ValueError("coverage intervals overlap")
    monitored_ms = sum(end - start for group in intervals.values() for start, end in group)
    if monitored_ms <= 0:
        raise ValueError("positive monitored coverage is required")

    def covering(session, start, end=None):
        for i, (low, high) in enumerate(intervals[session]):
            if low <= start < high and (end is None or end <= high):
                return i
        raise ValueError(f"event or alert outside monitored coverage: {session}")

    events = []
    identities = set()
    for row in truth:
        session, event_id = row["session_id"], row["event_id"]
        start, end = number(row["start_ms"]), number(row["end_ms"])
        if end <= start or not event_id or (session, event_id) in identities:
            raise ValueError("events need positive duration and unique session/event IDs")
        identities.add((session, event_id))
        events.append({"session_id": session, "event_id": event_id, "start_ms": start,
                       "end_ms": end, "coverage_index": covering(session, start, end)})
    events.sort(key=lambda row: (row["session_id"], row["start_ms"]))
    for left, right in zip(events, events[1:]):
        if left["session_id"] == right["session_id"] and left["end_ms"] > right["start_ms"]:
            raise ValueError("ground-truth fall intervals overlap")
    predictions = []
    for row in alerts:
        session, timestamp = row["session_id"], number(row["ts_ms"])
        confidence = number(row["confidence"])
        if confidence > 1:
            raise ValueError("confidence must be <= 1")
        index = covering(session, timestamp)
        if confidence >= threshold:
            predictions.append({"session_id": session, "ts_ms": timestamp, "coverage_index": index})
    predictions.sort(key=lambda row: (row["session_id"], row["ts_ms"]))
    matched, detections, false_alerts = set(), [], []
    for alert in predictions:
        candidates = [i for i, event in enumerate(events) if i not in matched
                      and event["session_id"] == alert["session_id"]
                      and event["coverage_index"] == alert["coverage_index"]
                      and event["start_ms"] <= alert["ts_ms"] < event["end_ms"] + grace_ms]
        if candidates:
            i = candidates[0]  # earliest unmatched event, deterministic one-to-one match
            matched.add(i)
            detections.append(dict(events[i], alert_ts_ms=alert["ts_ms"],
                                   latency_ms=alert["ts_ms"] - events[i]["start_ms"]))
        else:
            false_alerts.append(alert)
    nonfall_hours = (monitored_ms - sum(event["end_ms"] - event["start_ms"] for event in events)) / 3600000
    latencies = [item["latency_ms"] for item in detections]
    return {"threshold": threshold, "grace_ms": grace_ms, "true_fall_events": len(events),
            "detected_events": len(detections), "missed_events": len(events) - len(detections),
            "false_alerts": len(false_alerts), "monitored_hours": monitored_ms / 3600000,
            "nonfall_hours": nonfall_hours,
            "event_recall": len(detections) / len(events) if events else None,
            "alert_precision": len(detections) / len(predictions) if predictions else None,
            "false_alerts_per_nonfall_hour": len(false_alerts) / nonfall_hours if nonfall_hours else None,
            "median_onset_to_alert_ms": statistics.median(latencies) if latencies else None,
            "max_onset_to_alert_ms": max(latencies) if latencies else None,
            "detections": detections, "unmatched_alerts": false_alerts,
            "missed": [event for i, event in enumerate(events) if i not in matched]}


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required:
            raise ValueError(f"{path}: expected header {','.join(required)}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"{path}: malformed row")
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--alerts", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--grace-ms", type=float, default=3000)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError("report path already exists; choose a new version")
        result = evaluate(read_csv(args.truth, ["session_id", "event_id", "start_ms", "end_ms"]),
                          read_csv(args.alerts, ["session_id", "ts_ms", "confidence"]),
                          read_csv(args.coverage, ["session_id", "start_ms", "end_ms"]),
                          args.threshold, args.grace_ms)
        result["input_sha256"] = {key: sha256(getattr(args, key)) for key in ("truth", "alerts", "coverage")}
        write_json(args.out, result)
        print(f"Detected {result['detected_events']}/{result['true_fall_events']} events; "
              f"{result['false_alerts']} false alerts. Report: {args.out}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
