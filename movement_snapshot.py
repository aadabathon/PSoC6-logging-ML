"""Save and reopen Movement Study snapshots without loading their source files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from movement_baselines import META, check_wearable_study, runtime_versions, validate_settings


@dataclass(frozen=True)
class Settings:
    cv_folds: int = 5
    final_test: bool = False
    random_state: int = 42

    def __post_init__(self):
        validate_settings(self.cv_folds, self.final_test, self.random_state)


@dataclass(frozen=True)
class Study:
    name: str
    frame: pd.DataFrame
    imported: dict
    settings: Settings = field(default_factory=Settings)
    report: dict | None = None
    run_settings: Settings | None = None
    version_notes: tuple[str, ...] = ()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _atomic_write(destination, write):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_evaluation(path, report):
    _atomic_write(Path(path), lambda stream: stream.write(_json_bytes(report)))


def save_study(path, study):
    snapshot = study.frame.to_parquet(index=True, engine="pyarrow")
    manifest = {
        "format": "movement-study", "version": 1, "name": study.name, "imported": study.imported,
        "settings": asdict(study.settings), "report": study.report,
        "run_settings": asdict(study.run_settings) if study.run_settings is not None else None,
        "versions": runtime_versions(), "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
    }

    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("experiment.json", _json_bytes(manifest))
            archive.writestr("data.parquet", snapshot)

    _atomic_write(Path(path), write)


def _legacy_fields(manifest, frame):
    if manifest.get("recipe") != "wearable-study":
        raise ValueError("Select a Movement Study snapshot.")
    expected_roles = {
        col: "unsupported" if col in META.values() else "categorical" if col == "label" else "numeric"
        for col in frame.columns
    }
    if manifest.get("roles") != expected_roles or manifest["source"].get("kind") != "wearable_study":
        raise ValueError("Legacy study column roles or source metadata are invalid.")
    state = manifest["state"]
    report = state["report"].get("wearable_evaluation")
    settings = Settings(**state["request"]["params"])
    run_settings = Settings(**state["report_request"]["params"]) if report is not None else None
    return manifest["source"]["wearable_import"], settings, report, run_settings


def _check_report(report, run_settings, frame):
    if report is None and run_settings is None:
        return
    if not isinstance(report, dict) or run_settings is None:
        raise ValueError("The saved evaluation is missing its run settings.")
    labels = sorted(frame["label"].unique().tolist())
    splits = {"validation", "test"} if run_settings.final_test else {"validation"}
    if (report["seed"] != run_settings.random_state or report["final_test_evaluated"] is not run_settings.final_test
            or report["source_export_sha256"] != frame[META["export_sha256"]].iloc[0]
            or report["labels"] != labels or set(report["predictions"]) != splits
            or report["selected_model"] not in report["candidates"]):
        raise ValueError("The saved evaluation does not match the snapshot or run settings.")
    for candidate in report["candidates"].values():
        metrics = candidate["validation"]
        for key in ("macro_f1", "balanced_accuracy", "accuracy"):
            float(metrics[key])
        matrix = metrics["confusion_matrix"]
        if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
            raise ValueError("The saved confusion matrix does not match its labels.")
        if not isinstance(candidate["grouped_cv"], list):
            raise ValueError("The saved grouped CV results must be a list.")
    if run_settings.final_test:
        metrics = report["candidates"][report["selected_model"]]["test"]
        for key in ("macro_f1", "balanced_accuracy", "accuracy"):
            float(metrics[key])
    for key in ("warnings", "limitations"):
        if not isinstance(report[key], list) or not all(isinstance(item, str) for item in report[key]):
            raise ValueError("The saved evaluation notes must be a list of strings.")


def open_study(path):
    """Open a standalone or legacy MLCookbook snapshot without evaluating it."""
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["data.parquet", "experiment.json"]:
                raise ValueError("Expected a study manifest and one dataset snapshot.")
            manifest = json.loads(archive.read("experiment.json"))
            if (not isinstance(manifest, dict) or manifest.get("version") != 1
                    or manifest.get("format") not in ("movement-study", "mlcookbook-experiment")):
                raise ValueError("Unsupported Movement Study snapshot format.")
            snapshot = archive.read("data.parquet")
        if hashlib.sha256(snapshot).hexdigest() != manifest["snapshot_sha256"]:
            raise ValueError("The dataset snapshot checksum does not match.")
        frame = pd.read_parquet(io.BytesIO(snapshot), engine="pyarrow")
        check_wearable_study(frame)
        if manifest["format"] == "mlcookbook-experiment":
            imported, settings, report, run_settings = _legacy_fields(manifest, frame)
        else:
            imported = manifest["imported"]
            settings = Settings(**manifest["settings"])
            report = manifest["report"]
            run_settings = Settings(**manifest["run_settings"]) if manifest["run_settings"] is not None else None
        name = manifest["name"]
        if (not isinstance(name, str) or not name or not isinstance(imported, dict)
                or imported.get("source_export_sha256") != frame[META["export_sha256"]].iloc[0]
                or imported.get("group_by") != frame[META["group_by"]].iloc[0]
                or not isinstance(imported.get("features"), list)
                or not isinstance(imported.get("labels"), list)
                or not isinstance(imported.get("group_splits"), dict)):
            raise ValueError("The saved study provenance is invalid.")
        _check_report(report, run_settings, frame)
        installed = runtime_versions()
        recorded = manifest["versions"] if report is None else {**manifest["versions"], **report["versions"]}
        notes = tuple(f"Saved with {name} {version}. Installed version is {installed[name]}."
                      for name, version in recorded.items() if name in installed and version != installed[name])
        return Study(name, frame, imported, settings, report, run_settings, notes)
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, OSError) as error:
        raise ValueError(f"Could not open the study snapshot: {error}") from error
