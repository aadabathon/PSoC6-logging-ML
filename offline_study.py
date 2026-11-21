"""Assemble and verify a relocatable offline study using the shared movement evaluator."""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from mlcookbook_export import export_mlcookbook
from wearable_pipeline import prepare_dataset, sha256, utc_now, write_json

ARTIFACTS = {"prepared": "prepared/dataset.json", "movement_study": "mlcookbook/export.json",
             "baseline": "baseline/evaluation.json", "instructions": "START_HERE.md"}


def session_paths(inputs):
    """Discover finalized manifests without silently accepting misspelled inputs."""
    found = set()
    for item in inputs:
        path = Path(item).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"sessions input must be a directory: {path}")
        matches = [path / "session.json"] if (path / "session.json").is_file() else path.rglob("session.json")
        found.update(manifest.parent.resolve() for manifest in matches)
    if not found:
        raise ValueError("no session manifests found")
    return sorted(found)


def _hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "study.json"}


def build_study(inputs, output, *, group_by="subject", seed=42,
                profile=None, exclude_failed=False):
    """Prepare, export and evaluate once; never evaluate the reserved test groups."""
    from movement_evaluate import evaluate_export

    paths = session_paths(inputs)
    sources = [{"session_id": json.loads((p / "session.json").read_text(encoding="utf-8")).get("session_id"),
                "manifest_path": str(p / "session.json"), "manifest_sha256": sha256(p / "session.json")}
               for p in paths]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": utc_now(), "status": "building", "stage": "prepare",
              "sources": sources,
              "artifacts": dict(ARTIFACTS), "final_test_evaluated": False}
    write_json(output / "study.json", report)
    try:
        defaults = profile or {"window_samples": 100, "stride_samples": 25, "channels": 6}
        prepared = prepare_dataset(paths, output / "prepared", group_by=group_by, seed=seed,
                                   window=defaults["window_samples"], stride=defaults["stride_samples"],
                                   channels=defaults["channels"], profile=profile, exclude_failed=exclude_failed)
        report.update(synthetic=prepared["synthetic"], group_by=group_by,
                      group_splits=prepared["group_splits"], excluded_sessions=prepared["excluded_sessions"])
        report["stage"] = "export"
        write_json(output / "study.json", report)
        export_mlcookbook(output / "prepared", output / "mlcookbook")
        report["stage"] = "baseline"
        write_json(output / "study.json", report)
        result = evaluate_export(output / "mlcookbook", output / "baseline", seed=seed)
        report["stage"] = "source recheck"
        for source in sources:
            if sha256(source["manifest_path"]) != source["manifest_sha256"]:
                raise ValueError("source session metadata changed while building the study; use a new output")
        report["selected_model"] = result["selected_model"]
        warning = "**SYNTHETIC SOFTWARE FIXTURE. These scores are not movement-recognition evidence.**\n\n" if prepared["synthetic"] else ""
        (output / "START_HERE.md").write_text(
            "# Resume this offline movement study\n\n" + warning +
            "1. Verify with `python wearable.py verify-study --study PATH_TO_THIS_FOLDER`.\n"
            "2. From the analysis repository, run `python movement_study.py --study PATH_TO_THIS_FOLDER`. "
            "Opening does not train anything; keep the reserved test groups sealed.\n"
            "3. `baseline/evaluation.json` and `baseline/predictions_validation.csv` contain the "
            "native grouped baseline, exact window lineage, source hashes and runtime versions. "
            "Run in the desktop app to review interactively, then Save snapshot as `.mlcs`. "
            "Reopen that snapshot without the original CSVs; it contains settings and completed results.\n"
            "4. For statistics/PCA in ordinary recipes, import only `mlcookbook/train.csv`. "
            "Their row-wise scores remain exploratory.\n"
            "5. Use `prepared/deepcraft/train` and `prepared/deepcraft/validation` for CNN development. "
            "Preserve `prepared/dataset.json` group assignments. Freeze model/threshold choices before "
            "explicit final-test evaluation in a new output folder.\n\n"
            "`study.json` ties these artifacts together with SHA-256 hashes. The whole folder can move; "
            "source-session paths are provenance only, not required to reopen. Keep the immutable raw "
            "sessions separately: this folder is derived data, not a raw-data backup. "
            "Do not edit a completed study; create a new output for each revision. "
            "Classical window scores do not measure fall-event recall, false alerts/hour or latency.\n",
            encoding="utf-8")
        report.update(status="complete", stage="complete", artifacts_sha256=_hashes(output))
        write_json(output / "study.json", report)
    except Exception as exc:
        report.update(status="failed", error=str(exc), artifacts_sha256=_hashes(output))
        write_json(output / "study.json", report)
        raise
    return report


def _artifact(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("invalid study artifact path")
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or any(part in (".", "..") for part in name.split("/")):
        raise ValueError("study artifact path must stay inside its folder")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("study artifact path escapes its folder or is not a file")
    return path


def verify_study(path):
    """Verify a completed study without fitting a model or needing raw sessions."""
    path = Path(path)
    manifest = path / "study.json" if path.is_dir() else path
    root = manifest.resolve(strict=True).parent
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("study is incomplete or failed; inspect study.json and create a new output")
    if report.get("artifacts") != ARTIFACTS or report.get("final_test_evaluated") is not False:
        raise ValueError("invalid study artifact contract or test-holdout status")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not set(ARTIFACTS.values()).issubset(hashes):
        raise ValueError("study is missing required artifact hashes")
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("invalid study SHA-256")
        if sha256(_artifact(root, name)) != digest:
            raise ValueError(f"study checksum mismatch: {name}")
    prepared = json.loads(_artifact(root, ARTIFACTS["prepared"]).read_text(encoding="utf-8"))
    exported = json.loads(_artifact(root, ARTIFACTS["movement_study"]).read_text(encoding="utf-8"))
    baseline = json.loads(_artifact(root, ARTIFACTS["baseline"]).read_text(encoding="utf-8"))
    if (exported.get("source_dataset_sha256") != hashes[ARTIFACTS["prepared"]]
            or baseline.get("source_export_sha256") != hashes[ARTIFACTS["movement_study"]]
            or baseline.get("final_test_evaluated") is not False
            or "test" in baseline.get("predictions", {})):
        raise ValueError("study lineage or sealed test holdout does not match")
    for document, directory in ((prepared, "prepared"), (exported, "mlcookbook"), (baseline, "baseline")):
        for name, digest in document.get("artifacts_sha256", {}).items():
            if hashes.get(f"{directory}/{name}") != digest:
                raise ValueError(f"nested artifact hash is missing or inconsistent: {directory}/{name}")
    markers = [document.get("synthetic", False) for document in (report, prepared, exported)]
    if any(type(marker) is not bool for marker in markers) or len(set(markers)) != 1:
        raise ValueError("inconsistent study synthetic markers")
    return {"status": "verified", "synthetic": markers[0], "artifact_count": len(hashes),
            "movement_study": str(_artifact(root, ARTIFACTS["movement_study"])),
            "final_test_evaluated": False}
