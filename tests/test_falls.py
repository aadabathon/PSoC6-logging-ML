import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fall_eval import evaluate
from wearable_pipeline import SessionWriter, load_session, prepare_dataset
from test_wearable import row


def event(name, start, end):
    return {"session_id": "s1", "event_id": name, "start_ms": start, "end_ms": end}


def alert(timestamp, confidence=0.9):
    return {"session_id": "s1", "ts_ms": timestamp, "confidence": confidence}


class FallTests(unittest.TestCase):
    def test_event_metrics_and_duplicate_alarm_penalty(self):
        report = evaluate([event("f1", 1000, 2000), event("f2", 5000, 6000)],
                          [alert(1500), alert(1700), alert(5200, 0.2)],
                          [{"session_id": "s1", "start_ms": 0, "end_ms": 3600000}])
        self.assertEqual(report["event_recall"], 0.5)
        self.assertEqual(report["alert_precision"], 0.5)
        self.assertEqual(report["false_alerts"], 1)
        self.assertEqual(report["median_onset_to_alert_ms"], 500)
        self.assertEqual(report["missed"][0]["event_id"], "f2")
        self.assertAlmostEqual(report["false_alerts_per_nonfall_hour"], 3600000 / 3598000)

    def test_no_falls_does_not_claim_perfect_recall(self):
        report = evaluate([], [], [{"session_id": "s1", "start_ms": 0, "end_ms": 1000}])
        self.assertIsNone(report["event_recall"])
        self.assertIsNone(report["alert_precision"])
        self.assertEqual(report["false_alerts_per_nonfall_hour"], 0)

    def test_outside_exposure_and_overlap_rejected(self):
        coverage = [{"session_id": "s1", "start_ms": 0, "end_ms": 1000}]
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate([], [alert(1000)], coverage)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate([], [], coverage * 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate([], [], [])

    def test_gap_does_not_match_alert_to_earlier_fall(self):
        report = evaluate([event("f1", 100, 200)], [alert(1100)],
                          [{"session_id": "s1", "start_ms": 0, "end_ms": 1000},
                           {"session_id": "s1", "start_ms": 1100, "end_ms": 4000}])
        self.assertEqual(report["missed_events"], 1)
        self.assertEqual(report["false_alerts"], 1)

    def test_invalid_confidence_and_truth(self):
        coverage = [{"session_id": "s1", "start_ms": 0, "end_ms": 1000}]
        for value in (float("nan"), 1.2, -0.1):
            with self.assertRaises(ValueError):
                evaluate([], [alert(100, value)], coverage)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate([event("f1", 100, 200)] * 2, [], coverage)

    def test_alert_cancel_and_test_preserved_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = SessionWriter(temp, {})
            writer.feed(row().encode())
            for state in ("TEST", "CANCEL", "ALERT"):
                writer.feed(f"[fall] state={state} ts_ms=100 source_ts_ms=80 confidence=0.900\r\n".encode())
            writer.close()
            meta, rows = load_session(writer.path)
            self.assertEqual(meta["counts"]["fall_events"], 3)
            self.assertEqual(rows[0]["label"], "walking")
            with (writer.path / "events.csv").open() as stream:
                self.assertEqual([item["state"] for item in csv.DictReader(stream)], ["TEST", "CANCEL", "ALERT"])

    def test_belt_profile_guards_placement_and_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i in range(3):
                writer = SessionWriter(temp, {"subject": f"p{i}", "placement": "belt_right", "sample_rate_hz": 50})
                writer.feed("".join(row(i * 10000 + j * 20) for j in range(125)).encode())
                writer.close()
                paths.append(writer.path)
            profile = {"channels": 6, "placement_prefix": "belt", "allowed_labels": ["walking"]}
            report = prepare_dataset(paths, Path(temp) / "good", channels=6, window=100, stride=25, profile=profile)
            self.assertEqual(report["features"], ["ax", "ay", "az", "gx", "gy", "gz"])
            with self.assertRaisesRegex(ValueError, "channel count"):
                prepare_dataset(paths, Path(temp) / "bad", profile=profile)
            profile["allowed_labels"] = ["falling"]
            with self.assertRaisesRegex(ValueError, "labels outside"):
                prepare_dataset(paths, Path(temp) / "bad", channels=6, profile=profile)


if __name__ == "__main__":
    unittest.main()
