"""Write CLI artifacts using the standalone desktop app's evaluator."""

from __future__ import annotations

import csv
from pathlib import Path

from movement_baselines import evaluate_wearable, load_wearable_study
from wearable_pipeline import sha256, write_json


def evaluate_export(directory, output, *, final_test=False, seed=42, folds=5):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output must be new.")
    frame, imported = load_wearable_study(directory)
    result = evaluate_wearable(frame, cv_folds=folds, final_test=final_test, random_state=seed)
    report = result.to_dict()
    report["native_evaluator"] = "movement_baselines.evaluate_wearable"
    report["import_report"] = imported.to_dict()
    report["evaluator_source_sha256"] = sha256(Path(__file__).with_name("movement_baselines.py"))

    output.mkdir(parents=True, exist_ok=False)
    for split, rows in report["predictions"].items():
        with (output / f"predictions_{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "README.md").write_text(
        "# Grouped movement baseline\n\n"
        f"Selected {report['selected_model']} using the fixed validation groups.\n\n"
        "The CLI and Movement Study desktop app share this evaluator. "
        "evaluation.json includes exact groups, folds, metrics, predictions and lineage. "
        "These classical baselines do not provide CNN weights or fall-event performance.\n",
        encoding="utf-8",
    )
    report["artifacts_sha256"] = {path.name: sha256(path) for path in sorted(output.iterdir())}
    write_json(output / "evaluation.json", report)
    return report
