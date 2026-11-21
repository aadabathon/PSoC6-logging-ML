import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_study import build_study, session_paths, verify_study
from mlcookbook_export import export_mlcookbook
from test_wearable import row
from wearable import main
from wearable_pipeline import SessionWriter, prepare_dataset, write_json

class OfflineStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for person in range(6):
            writer = SessionWriter(self.root / "sessions", {"subject": f"fixture-{person}",
                "sample_rate_hz": 50, "placement": "belt_fixture", "synthetic": True})
            for index, label in enumerate(("walking", "standing")):
                writer.feed("".join(row(person * 10000 + index * 4000 + sample * 20, label=label,
                                       acceleration=9.80665 + index * 3 + sample % 7 * 0.02)
                                    for sample in range(125)).encode())
            writer.close()
            self.paths.append(writer.path)
        self.output = self.root / "study with spaces"

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_study([self.root / "sessions"], self.output)

    def test_end_to_end_native_lineage_synthetic_and_sealed_test_after_relocation(self):
        report = self.build()
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["synthetic"])
        baseline = json.loads((self.output / "baseline/evaluation.json").read_text())
        self.assertFalse(baseline["final_test_evaluated"])
        self.assertNotIn("test", baseline["predictions"])
        self.assertTrue(any("Synthetic" in warning for warning in baseline["warnings"]))
        self.assertEqual(baseline["source_export_sha256"], report["artifacts_sha256"]["mlcookbook/export.json"])
        self.assertEqual(baseline["native_evaluator"], "movement_baselines.evaluate_wearable")
        self.assertTrue((self.output / "prepared/deepcraft/train").is_dir())
        relocated = self.root / "relocated"
        shutil.copytree(self.output, relocated)
        (self.root / "sessions").rename(self.root / "raw archive")
        verified = verify_study(relocated)
        json.dumps(verified, allow_nan=False)
        self.assertEqual(verified["movement_study"], str(relocated / "mlcookbook/export.json"))
        self.assertFalse(verified["final_test_evaluated"])
        with self.assertRaises(FileExistsError):
            build_study([self.root / "raw archive"], self.output)

    def test_changed_artifact_and_unsafe_path_fail_verification(self):
        report = self.build()
        csv = self.output / "mlcookbook/train.csv"
        original = csv.read_bytes()
        csv.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_study(self.output / "study.json")
        csv.write_bytes(original)
        for unsafe in ("../outside.txt", "C:/outside.txt", "/outside.txt", "mlcookbook\\train.csv"):
            tampered = dict(report, artifacts_sha256={**report["artifacts_sha256"], unsafe: "0" * 64})
            write_json(self.output / "study.json", tampered)
            with self.assertRaisesRegex(ValueError, "path"):
                verify_study(self.output)

    def test_nested_hash_removal_and_holdout_changes_fail(self):
        report = self.build()
        report["artifacts_sha256"].pop("prepared/windows.npz")
        write_json(self.output / "study.json", report)
        with self.assertRaisesRegex(ValueError, "nested artifact"):
            verify_study(self.output)
        report["final_test_evaluated"] = True
        write_json(self.output / "study.json", report)
        with self.assertRaisesRegex(ValueError, "holdout"):
            verify_study(self.output)

    def test_failure_manifest_preserves_prepared_data_and_cannot_be_verified(self):
        with patch("offline_study.export_mlcookbook", side_effect=ValueError("injected export failure")):
            with self.assertRaisesRegex(ValueError, "injected"):
                self.build()
        report = json.loads((self.output / "study.json").read_text())
        self.assertEqual((report["status"], report["stage"]), ("failed", "export"))
        self.assertIn("prepared/windows.npz", report["artifacts_sha256"])
        with self.assertRaisesRegex(ValueError, "incomplete or failed"):
            verify_study(self.output)
        self.assertTrue(all((p / "raw.log").is_file() for p in self.paths))

    def test_mixed_or_invalid_synthetic_metadata_is_rejected_before_export(self):
        manifest = self.paths[0] / "session.json"
        metadata = json.loads(manifest.read_text())
        for marker, error in ((False, "separate studies"), ("true", "boolean")):
            metadata["synthetic"] = marker
            write_json(manifest, metadata)
            with self.assertRaisesRegex(ValueError, error):
                prepare_dataset(self.paths, self.root / "mixed", window=100, stride=25, channels=6)
            self.assertFalse((self.root / "mixed").exists())

    def test_source_manifest_change_during_build_is_not_acknowledged_as_complete(self):
        def changed_source(*args):
            result = export_mlcookbook(*args)
            manifest = self.paths[0] / "session.json"
            metadata = json.loads(manifest.read_text())
            metadata["notes"] = "changed during build"
            write_json(manifest, metadata)
            return result

        with patch("offline_study.export_mlcookbook", side_effect=changed_source):
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                self.build()
        report = json.loads((self.output / "study.json").read_text())
        self.assertEqual((report["status"], report["stage"]), ("failed", "source recheck"))
        self.assertTrue((self.output / "baseline/evaluation.json").is_file())

    def test_cli_assembly_verification_and_missing_inputs(self):
        main(["study", "--sessions", str(self.root / "sessions"), "--out", str(self.output)])
        main(["verify-study", "--study", str(self.output)])
        self.assertTrue(verify_study(self.output)["synthetic"])
        self.assertEqual(session_paths([self.root / "sessions", self.paths[0]]), sorted(self.paths))
        with self.assertRaises(FileNotFoundError):
            session_paths([self.root / "typo"])
        with self.assertRaisesRegex(ValueError, "no session"):
            session_paths([self.root / "study with spaces/baseline"])


if __name__ == "__main__":
    unittest.main()
