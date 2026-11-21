import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wearable_pipeline import (CSV_COLUMNS, LEGACY_COLUMNS, ProtocolError, RowParser,
                               SessionWriter, import_recording, load_session, prepare_dataset, write_json)
from wearable import capture


def row(ts=0, label="walking", legacy=False, acceleration=9.80665):
    values = dict.fromkeys(CSV_COLUMNS, "")
    values.update(ts_ms=str(ts), ax=str(acceleration), ay="0", az="0", gx="1", gy="2", gz="3", label=label)
    columns = LEGACY_COLUMNS if legacy else CSV_COLUMNS
    return ",".join(values[key] for key in columns) + "\n"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def session(self, subject="p1", offset=0, extra="", samples=110):
        writer = SessionWriter(self.root / "sessions", {"subject": subject, "sample_rate_hz": 50})
        writer.feed((",".join(CSV_COLUMNS) + "\n" + extra + "".join(row(offset + i * 20) for i in range(samples))).encode())
        writer.close()
        return writer.path

    def test_legacy_and_current_wire_formats(self):
        for legacy in (False, True):
            parsed = RowParser().parse(row(20, legacy=legacy))
            self.assertEqual(parsed["label"], "walking")
            self.assertEqual(list(parsed), CSV_COLUMNS)

    def test_header_and_label_mismatch_fail_loudly(self):
        with self.assertRaisesRegex(ProtocolError, "header"):
            RowParser().parse("ts_ms,ax,label")
        with self.assertRaisesRegex(ProtocolError, "mismatch"):
            RowParser("standing").parse(row())

    def test_nonfinite_missing_and_invalid_timestamp_rejected(self):
        for line in (row(acceleration="nan"), row(acceleration="inf"), row(acceleration=""), row(-1), row(2**32)):
            with self.assertRaises(ProtocolError):
                RowParser().parse(line)

    def test_chunk_framing_raw_preservation_and_rejections(self):
        writer = SessionWriter(self.root, {"subject": "p1"})
        payload = (row(0) + "20,broken\n" + row(40)).encode() + b"\xff\n"
        for index in range(0, len(payload), 7):
            writer.feed(payload[index:index + 7])
        writer.close()
        meta, rows = load_session(writer.path)
        self.assertEqual((writer.path / "raw.log").read_bytes(), payload)
        self.assertEqual(meta["counts"]["rows"], 2)
        self.assertEqual(meta["counts"]["rejected"], 2)
        self.assertNotEqual(rows[0]["segment_id"], rows[1]["segment_id"])

    def test_empty_capture_is_failed(self):
        writer = SessionWriter(self.root, {})
        writer.close()
        self.assertEqual(writer.metadata["status"], "failed")
        with self.assertRaisesRegex(ValueError, "finalized"):
            load_session(writer.path)

    def test_partial_and_oversized_lines_are_bounded(self):
        writer = SessionWriter(self.root, {})
        writer.feed(b"7" * 20000 + b"\n" + row().encode() + b"partial")
        self.assertLess(len(writer.buffer), 4097)
        writer.close()
        self.assertEqual(writer.counts["rejected"], 2)

    def test_clock_reset_gap_and_header_split_segments(self):
        writer = SessionWriter(self.root, {})
        writer.feed((row(100) + row(100) + row(300) + row(0) +
                     "[flash] session ts_ms=20\n" + row(20) + ",".join(CSV_COLUMNS) + "\n" + row(40)).encode())
        writer.close()
        _, rows = load_session(writer.path)
        self.assertEqual(len({item["segment_id"] for item in rows}), 6)

    def test_hash_detects_edit(self):
        path = self.session()
        with (path / "samples.csv").open("a") as stream:
            stream.write("modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_session(path)

    def test_manifest_retries_only_transient_windows_sharing_errors(self):
        destination = self.root / "manifest.json"
        original_replace = Path.replace
        calls = []

        def transient_replace(source, target):
            calls.append(target)
            if len(calls) < 3:
                error = PermissionError("temporarily locked")
                error.winerror = 32
                raise error
            return original_replace(source, target)

        with patch.object(Path, "replace", transient_replace), patch("wearable_pipeline.time.sleep") as delay:
            write_json(destination, {"valid": True})
        self.assertEqual(json.loads(destination.read_text()), {"valid": True})
        self.assertEqual(delay.call_count, 2)
        with patch.object(Path, "replace", side_effect=PermissionError("access denied")), \
                patch("wearable_pipeline.time.sleep") as delay:
            with self.assertRaises(PermissionError):
                write_json(destination, {"valid": False})
        delay.assert_not_called()
        self.assertEqual(json.loads(destination.read_text()), {"valid": True})

    def test_failed_checkpoint_closes_streams_and_does_not_finalize_manifest(self):
        writer = SessionWriter(self.root, {})
        writer.feed(row().encode())
        error = PermissionError("persistent file lock")
        error.winerror = 32
        with patch.object(Path, "replace", side_effect=error), patch("wearable_pipeline.time.sleep") as delay:
            with self.assertRaises(PermissionError):
                writer.close()
        self.assertEqual(delay.call_count, 5)
        self.assertTrue(writer.closed)
        self.assertTrue(all(stream.closed for stream in (writer.raw, writer.samples, writer.rejected, writer.events)))
        self.assertEqual(json.loads((writer.path / "session.json").read_text())["status"], "recording")

    def test_import_last_line_without_newline(self):
        source = self.root / "dump.txt"
        source.write_text(row(0).rstrip(), encoding="utf-8")
        path = import_recording(source, self.root / "imported", {"subject": "p1"})
        self.assertEqual(load_session(path)[0]["counts"]["rows"], 1)
        self.assertEqual(source.read_text(), row(0).rstrip())

    def test_prepare_group_isolation_units_and_reproducibility(self):
        paths = [self.session(f"p{i}", i * 10000) for i in range(6)]
        # Second recording by the same subject stays in the same split.
        paths.append(self.session("p0", 99900))
        output = self.root / "dataset"
        report = prepare_dataset(paths, output)
        again = prepare_dataset(paths[::-1], self.root / "again")
        self.assertEqual(report["group_splits"], again["group_splits"])
        with np.load(output / "windows.npz", allow_pickle=False) as data:
            self.assertEqual(data["X"].shape, (21, 50, 3))
            np.testing.assert_allclose(data["X"][:, :, 0], 1)
            self.assertEqual(set(data["split"]), {"train", "validation", "test"})
        with (output / "windows.csv").open() as stream:
            index = list(csv.DictReader(stream))
        for subject in {item["subject"] for item in index}:
            self.assertEqual(len({item["split"] for item in index if item["subject"] == subject}), 1)
        self.assertFalse(report["normalization"]["applied"])
        self.assertTrue(list((output / "deepcraft").rglob("label.label")))
        self.assertIn("windows.npz", report["artifacts_sha256"])

    def test_no_windows_across_gap_or_label_boundary(self):
        paths = []
        for subject in range(3):
            writer = SessionWriter(self.root / "sessions", {"subject": f"p{subject}", "sample_rate_hz": 50})
            # Two 30-sample runs must not form a 50-sample window.
            writer.feed(("".join(row(subject * 10000 + i * 20) for i in range(30)) +
                         "20,bad\n" + "".join(row(subject * 10000 + 1000 + i * 20) for i in range(30))).encode())
            writer.close()
            paths.append(writer.path)
        with self.assertRaisesRegex(ValueError, "no complete"):
            prepare_dataset(paths, self.root / "bad")

    def test_missing_groups_and_duplicate_sources_rejected(self):
        first = self.session("p0")
        with self.assertRaisesRegex(ValueError, "3 distinct"):
            prepare_dataset([first], self.root / "few")
        copied = self.session("p1")
        with self.assertRaisesRegex(ValueError, "duplicate raw"):
            prepare_dataset([first, copied], self.root / "duplicate")

    def test_unknown_subject_not_silently_random_split(self):
        paths = [self.session("unknown", 0), self.session("p2", 10000), self.session("p3", 20000)]
        with self.assertRaisesRegex(ValueError, "known subject"):
            prepare_dataset(paths, self.root / "missing")
        prepare_dataset(paths, self.root / "explicit", group_by="session_id")

    def test_six_axis_export_preserves_gyro_units(self):
        paths = [self.session(f"p{i}", i * 10000) for i in range(3)]
        prepare_dataset(paths, self.root / "six", channels=6)
        with np.load(self.root / "six" / "windows.npz") as data:
            np.testing.assert_allclose(data["X"][0, 0], [1, 0, 0, 1, 2, 3])

    def test_label_transition_does_not_create_training_windows(self):
        paths = []
        for i in range(3):
            writer = SessionWriter(self.root, {"subject": f"p{i}", "sample_rate_hz": 50})
            writer.feed("".join(row(i * 10000 + j * 20, "walking" if j < 30 else "standing")
                                for j in range(60)).encode())
            writer.close()
            paths.append(writer.path)
        with self.assertRaisesRegex(ValueError, "no complete"):
            prepare_dataset(paths, self.root / "labels")

    def test_cumulative_cadence_drift_rejected(self):
        paths = []
        for i in range(3):
            writer = SessionWriter(self.root, {"subject": f"p{i}", "sample_rate_hz": 50})
            writer.feed("".join(row(i * 10000 + j * 25) for j in range(100)).encode())
            writer.close()
            paths.append(writer.path)
        with self.assertRaisesRegex(ValueError, "no complete"):
            prepare_dataset(paths, self.root / "drift")

    def test_transport_disconnect_finalizes_failed_session(self):
        class Connection:
            def __init__(self):
                self.chunks = iter([row().encode()[:15], row().encode()[15:], b""])
                self.closed = False

            def settimeout(self, _):
                pass

            def recv(self, _):
                return next(self.chunks)

            def close(self):
                self.closed = True

        connection = Connection()
        args = Namespace(transport="tcp", label="walking", outdir=self.root,
                         host="mock-board", port=50007, duration=None, idle_timeout=15,
                         subject="p1", placement="wrist", orientation="test", enclosure="v1",
                         device="test", firmware="test", notes="")
        with patch("wearable.socket.create_connection", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                capture(args)
        self.assertTrue(connection.closed)
        manifest = json.loads(next(self.root.rglob("session.json")).read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["counts"]["rows"], 1)
        self.assertIn("raw.log", manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
