import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from sklearn.preprocessing import StandardScaler

from movement_baselines import META, evaluate_wearable, load_wearable_study
from movement_evaluate import evaluate_export
from movement_snapshot import Settings, Study, export_evaluation, open_study, save_study
from mlcookbook_export import export_mlcookbook
from test_wearable import row
from wearable_pipeline import SessionWriter, prepare_dataset, sha256

class GroupedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        paths=[]
        for subject in range(6):
            writer=SessionWriter(self.root/"sessions",{"subject":f"p{subject}","sample_rate_hz":50})
            for index,label in enumerate(("walking","standing")):
                writer.feed("".join(row(subject*10000+index*4000+i*20,label=label,
                                       acceleration=9.80665+index*3+(i%7)*0.02) for i in range(125)).encode())
            writer.close()
            paths.append(writer.path)
        prepare_dataset(paths,self.root/"prepared",window=100,stride=25,channels=6)
        self.export=self.root/"export"
        export_mlcookbook(self.root/"prepared",self.export)

    def tearDown(self):self.temp.cleanup()

    def test_grouped_cv_and_final_test_have_lineage(self):
        report=evaluate_export(self.export,self.root/"result",final_test=True)
        self.assertTrue(report["final_test_evaluated"])
        json.dumps(report,allow_nan=False)
        for candidate in report["candidates"].values():
            self.assertEqual(len(candidate["grouped_cv"]),4)
            for fold in candidate["grouped_cv"]:
                self.assertFalse(set(fold["train_groups"])&set(fold["validation_groups"]))
                self.assertFalse(set(fold["train_source_windows"])&set(fold["validation_source_windows"]))
        self.assertIn("test",report["candidates"][report["selected_model"]])
        with (self.root/"result"/"predictions_test.csv").open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r["split"]=="test" and "source_window" in r for r in rows))
        with self.assertRaises(FileExistsError):
            evaluate_export(self.export,self.root/"result")

    def test_test_labels_do_not_choose_model_or_change_validation_predictions(self):
        first=evaluate_export(self.export,self.root/"first",folds=0)
        self.assertFalse(first["final_test_evaluated"])
        manifest=json.loads((self.export/"export.json").read_text())
        # Swap only test labels and their provenance; holdout values must not
        # influence model selection or validation predictions.
        for name in ("test.csv","test_index.csv"):
            path=self.export/name
            text=path.read_text().replace("walking","temporary").replace("standing","walking").replace("temporary","standing")
            path.write_text(text,encoding="utf-8")
            manifest["artifacts_sha256"][name]=sha256(path)
        (self.export/"export.json").write_text(json.dumps(manifest))
        second=evaluate_export(self.export,self.root/"second",folds=0)
        self.assertEqual(first["selected_model"],second["selected_model"])
        self.assertEqual((self.root/"first"/"predictions_validation.csv").read_bytes(),
                         (self.root/"second"/"predictions_validation.csv").read_bytes())

    def test_modified_export_is_rejected(self):
        with (self.export/"train.csv").open("a") as stream:stream.write("modified")
        with self.assertRaisesRegex(ValueError,"checksum"):
            load_wearable_study(self.export)

    def test_cli_and_native_desktop_core_share_predictions(self):
        report=evaluate_export(self.export,self.root/"native",folds=0)
        dataset,_=load_wearable_study(self.export)
        native=evaluate_wearable(dataset,cv_folds=0).to_dict()
        self.assertEqual(report["native_evaluator"],"movement_baselines.evaluate_wearable")
        self.assertEqual(report["selected_model"],native["selected_model"])
        self.assertEqual(report["predictions"],native["predictions"])
        self.assertEqual(report["dataset_fingerprint"],native["dataset_fingerprint"])

    def test_scaler_fits_only_training_windows(self):
        frame, imported = load_wearable_study(self.export)
        frame.loc[frame[META["split"]] != "train", imported.features] += 10000
        original = StandardScaler.fit
        seen = []

        def observe(scaler, values, *args, **kwargs):
            seen.append(np.asarray(values))
            return original(scaler, values, *args, **kwargs)

        with patch.object(StandardScaler, "fit", observe):
            evaluate_wearable(frame)
        self.assertEqual(sorted(len(values) for values in seen), [12, 12, 12, 12, 16])
        self.assertTrue(all(values.max() < 100 for values in seen))

    def test_snapshot_preserves_unrun_settings_and_reopens_offline(self):
        frame, imported = load_wearable_study(self.export)
        run = Settings(cv_folds=0)
        edited = Settings(cv_folds=5, final_test=True, random_state=99)
        report = evaluate_wearable(frame, **asdict(run)).to_dict()
        path = self.root / "saved.mlcs"
        save_study(path, Study("Pilot", frame, imported.to_dict(), edited, report, run))
        shutil.rmtree(self.export)

        with patch("movement_baselines.evaluate_wearable", side_effect=AssertionError("unexpected fit")):
            reopened = open_study(path)
        self.assertEqual(reopened.settings, edited)
        self.assertEqual(reopened.run_settings, run)
        self.assertEqual(reopened.report, report)
        self.assertNotIn("test", reopened.report["predictions"])
        rerun = evaluate_wearable(reopened.frame, **asdict(run)).to_dict()
        self.assertEqual(rerun["predictions"], report["predictions"])
        self.assertEqual(rerun["dataset_fingerprint"], report["dataset_fingerprint"])
        exported = self.root / "evaluation.json"
        export_evaluation(exported, reopened.report)
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8")), report)

    def test_legacy_movement_snapshot_reopens_and_checksum_is_verified(self):
        frame, imported = load_wearable_study(self.export)
        snapshot = frame.to_parquet(index=True, engine="pyarrow")
        roles = dict.fromkeys(imported.features, "numeric")
        roles.update({name: "unsupported" for name in META.values()}, label="categorical")
        manifest = {
            "format": "mlcookbook-experiment", "version": 1, "recipe": "wearable-study",
            "name": "Legacy pilot", "roles": roles, "versions": {},
            "source": {"kind": "wearable_study", "wearable_import": imported.to_dict()},
            "state": {"request": {"params": asdict(Settings())}, "report": {}},
            "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        }
        path = self.root / "legacy.mlcb"
        with ZipFile(path, "w") as archive:
            archive.writestr("experiment.json", json.dumps(manifest))
            archive.writestr("data.parquet", snapshot)
        self.assertIsNone(open_study(path).report)
        with ZipFile(path, "w") as archive:
            archive.writestr("experiment.json", json.dumps(manifest))
            archive.writestr("data.parquet", snapshot + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            open_study(path)


if __name__=="__main__":unittest.main()
