import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlcookbook_export import export_mlcookbook, window_features
from test_wearable import row
from wearable import main
from wearable_pipeline import SessionWriter, prepare_dataset, sha256


class CookbookExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        paths = []
        for i in range(3):
            writer = SessionWriter(self.root / "sessions", {"subject": f"p{i}", "sample_rate_hz": 50})
            for j, label in enumerate(("walking", "standing")):
                writer.feed("".join(row(i * 10000 + j * 4000 + k * 20, label=label)
                                    for k in range(125)).encode())
            writer.close()
            paths.append(writer.path)
        self.dataset = self.root / "dataset"
        self.report = prepare_dataset(paths, self.dataset, window=100, stride=25, channels=6)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite_manifest(self):
        (self.dataset / "dataset.json").write_text(json.dumps(self.report), encoding="utf-8")

    def test_feature_values_and_three_axis_support(self):
        values = np.ones((2, 4, 3))
        names, features = window_features(values, ["ax", "ay", "az"])
        self.assertEqual(features.shape, (2, 32))
        np.testing.assert_allclose(features[:, names.index("ax_mean")], 1)
        np.testing.assert_allclose(features[:, names.index("ax_std")], 0)
        np.testing.assert_allclose(features[:, names.index("accel_mag_rms")], np.sqrt(3))
        np.testing.assert_allclose(features[:, names.index("ax_mean_abs_delta")], 0)

    def test_preserves_splits_units_lineage_and_excludes_leaking_columns(self):
        output = self.root / "export"
        report = export_mlcookbook(self.dataset, output)
        self.assertEqual(len(report["features"]), 64)
        self.assertEqual(report["group_splits"], self.report["group_splits"])
        subjects = {}
        for split in ("train", "validation", "test"):
            with (output / f"{split}.csv").open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, report["features"] + ["label"])
            with (output / f"{split}_index.csv").open() as stream:
                index = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(index), len(rows))
            for values, entry in zip(rows, index):
                self.assertEqual(values["label"], entry["label"])
                self.assertEqual(entry["split"], split)
                self.assertEqual(subjects.setdefault(entry["subject"], split), split)
                self.assertAlmostEqual(float(values["ax_mean"]), 1, places=5)
                self.assertAlmostEqual(float(values["gx_mean"]), 1, places=5)
                self.assertNotIn("ml_conf", values)
                self.assertNotIn("subject", values)
                self.assertNotIn("source_window", values)
                self.assertNotIn("start_ts_ms", values)
        for name, digest in report["artifacts_sha256"].items():
            self.assertEqual(sha256(output / name), digest)
        self.assertIn("row-wise", report["limitation"])
        with self.assertRaises(FileExistsError):
            export_mlcookbook(self.dataset, output)

    def test_rejects_modified_source_before_creating_output(self):
        with (self.dataset / "windows.csv").open("a") as stream:
            stream.write("modified")
        output = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "checksum"):
            export_mlcookbook(self.dataset, output)
        self.assertFalse(output.exists())

    def test_rejects_index_array_mismatch_even_with_updated_hash(self):
        path = self.dataset / "windows.csv"
        path.write_text(path.read_text().replace(",walking,", ",jumping,", 1), encoding="utf-8")
        self.report["artifacts_sha256"]["windows.csv"] = sha256(path)
        self.rewrite_manifest()
        with self.assertRaisesRegex(ValueError, "index"):
            export_mlcookbook(self.dataset, self.root / "bad")

    def test_rejects_group_reassignment(self):
        group = next(iter(self.report["group_splits"]))
        self.report["group_splits"][group] = "train" if self.report["group_splits"][group] != "train" else "test"
        self.rewrite_manifest()
        with self.assertRaisesRegex(ValueError, "group"):
            export_mlcookbook(self.dataset, self.root / "bad")

    def test_rejects_nonfinite_tensor_even_with_updated_hash(self):
        path = self.dataset / "windows.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["X"][0, 0, 0] = np.nan
        np.savez_compressed(path, **arrays)
        self.report["artifacts_sha256"]["windows.npz"] = sha256(path)
        self.rewrite_manifest()
        with self.assertRaisesRegex(ValueError, "tensor"):
            export_mlcookbook(self.dataset, self.root / "bad")

    def test_cli_export(self):
        main(["export-mlcookbook", "--dataset", str(self.dataset), "--out", str(self.root / "cli")])
        self.assertTrue((self.root / "cli" / "train.csv").is_file())


if __name__ == "__main__":
    unittest.main()
