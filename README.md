# Wearable movement data pipeline

Capture 50 Hz IMU recordings from the CY8CKIT-062S2-AI over UART or TCP, retain
raw evidence, and prepare reproducible offline datasets for DEEPCRAFT and Python.
Run the following commands from `analysis/`. Firmware instructions are in
`../workingDir/documentation/wearable.md` (a separate Git repository).

The target is **belt mounting**, with walking, sprinting, jumping, biking,
driving and falling, plus standing/sitting/transitions/device-drop negatives.
Use the belt profile below for new training; the wrist starter model does not
recognize these new classes without retraining.

Start with [the complete demo build guide](../workingDir/DEMO_BUILD.md) for print
parts, power, traceable firmware images and the hands-on acceptance sequence.

## Flash v2 and grouped offline evaluation

New serial commands wait for board acknowledgments and retain all received
evidence. From this repository, with the analysis environment active:

```powershell
# Optional first format DISCARDs previous flash contents; omit --format thereafter.
python wearable.py configure-flash --port COM3 --label walking --transcript data/setup-01.log --format
# Keep battery power connected while disconnecting USB; USER BTN1 starts/stops.
python wearable.py download-flash --port COM3 --dest data/download-01.log --subject p01 --placement belt_right --orientation "USB upward, PCB outward" --enclosure belt-v1 --firmware ACTUAL_BUILD_ID
# A previously downloaded complete v2 dump can also be imported:
python wearable.py import-flash data/download-01.log --outdir data/reimport --subject p01 --placement belt_right

python wearable.py prepare --sessions data/sessions --profile belt-v1 --out data/processed/study-01
python wearable.py export-features --dataset data/processed/study-01 --out data/processed/features-01
python wearable.py evaluate --dataset data/processed/features-01 --out data/processed/evaluation-01
```

V2 downloads verify a complete begin/end, every sequence and wire CRC before
finalizing sessions. Each physical START/STOP is recovered separately; power-cut,
gap and queue-loss sessions stay failed with their source evidence. Preparation
rejects failed recordings by default. Review them first, then explicitly use
`prepare --exclude-failed` to retain an exclusion audit in `dataset.json`.
Downloads never erase. Metadata applies to every recording in that batch, so
download between changes in subject, placement or enclosure. Board label/sink
configuration is volatile across reset. Existing v1 imports remain unverified.

`evaluate` runs whole-group CV within training and selects logistic regression
or random forest by macro F1 on the fixed validation groups. It exports
probabilities and source-window/session/clock lineage. The desktop app uses
the same evaluator. The reserved test is evaluated only with an explicit
desktop choice or `--final-test` in a new CLI output directory.

For real-device bench checks:

```powershell
python hardware_check.py --port COM3 --out data/acceptance/bench-01 --expected-build-id ACTUAL_BUILD_ID --exercise-alert
```

This retains evidence and checks the device contract, sample timing, logger
losses, minimum free heap and TEST/CANCEL events. Physical control visibility,
flash recovery, battery behavior and CNN accuracy remain hands-on checks.
Keep acceptance recordings outside the training sessions directory.

## Setup and capture

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python collect.py --port COM3 --label walking --duration 45 --subject p01 --placement belt_right --orientation "USB upward, board facing outward" --enclosure v1 --firmware YOUR_FLASHED_REVISION
```

Replace `YOUR_FLASHED_REVISION` with the actual flashed revision.
The serial collector sends stop, mode csv, sink uart, label, start, and stop on
exit. Close other programs holding the COM port. `Ctrl-C` finalizes the capture.
Labels must be 1-15 characters in `[A-Za-z0-9_-]`.

For TCP, configure the default firmware's Wi-Fi credentials locally and use its
serial CLI to set `mode csv`, `sink tcp`, `label walking`, and `start`. Then:

```powershell
python collect_tcp.py --host 192.168.1.42 --label walking --duration 45 --subject p01 --placement belt_right --enclosure v1
```

TCP is push-only: the collector validates the board's label but cannot change it
or stop the board when finished. A disconnect fails and finalizes that session;
reconnect as a new recording. No-data timeout defaults to 15 seconds. Current
transport has no authentication/encryption; use your trusted local network.

The existing `--port`, `--baud` (serial), `--duration`, `--label` and `--outdir`
options remain supported. **Output is now a unique session directory**, not a
single flat CSV. Three additional metadata fields are especially useful for
wearables: `--placement`, `--orientation`, and `--enclosure`. Use pseudonymous
subject IDs. Unknown metadata is recorded honestly; unknown subjects are rejected
for subject-based training splits.

Each recording contains:

| File | Purpose |
| --- | --- |
| `raw.log` | Original received bytes, including diagnostics and corrupt rows |
| `samples.csv` | Validated 21-column schema plus `segment_id`; source physical units |
| `rejected.jsonl` | Rejection reason and offending row/byte reference |
| `events.csv` | Model fall alerts, cancellations and explicit indicator tests |
| `session.json` | Metadata, capture status, counters, timestamps and SHA-256 hashes |

Both the old 19-column schema (including flash dumps) and current 21-column
schema are accepted. Predictions remain separate from ground-truth labels.
All six IMU values must be finite; other missing sensors remain blank. Unknown
headers and mismatched expected labels fail loudly. Gaps, resets, duplicates,
rejected rows, repeated headers and flash boundaries split contiguous segments.
Raw data is never silently repaired, relabeled or interpolated. Live partial
last lines are quarantined. Buffers are bounded and files checkpoint every 100
valid rows; process/power failure can lose the unflushed tail. Such sessions stay
`recording` and cannot enter training without review/reimport.

`prestart_raw_bytes` marks the UART setup prefix in `raw.log`; those bytes are
retained for diagnosis but excluded from this session's normalized samples.
When recovering raw captures, omit that prefix before importing and preserve
the original separately. A failed transport session remains available for
inspection but `prepare` refuses it. Do not edit manifests to bypass checks.

## Import and inspect offline

```powershell
python wearable.py import old_capture.csv --subject p01 --placement belt_right --enclosure v1
python wearable.py audit data/sessions/<session-id>
```

Import never modifies the source. It supports saved CSV/UART dumps and preserves
labels from the data. An optional `--label` asserts the expected label. A dump
may include several activities but must belong to the subject/mounting described
by its metadata. Legacy dumps without session markers cannot recover hidden
recording boundaries; split them from your collection notes before import.
See firmware documentation before relying on its experimental flash storage.

## Build training data

Record every intended class across at least three independent participants
(preferably more, with multiple sessions/days). Then:

```powershell
python wearable.py prepare --sessions data/sessions --profile belt-v1 --out data/processed/belt-v1
```

The **belt-v1 profile** uses six IMU channels, 100 samples per window (2 seconds),
and stride 25 (0.5 seconds), with belt placement and label validation. Allowed
labels live in `profiles/belt-v1.json`. It trains on the classes actually
collected; the manifest does not invent data for missing activities. `--window`
and `--stride` may override these provisional research defaults.

Without a profile, the default is 50 samples per window, stride 30, three accelerometer channels in
g. This matches the shape/stride of the existing IMUexample CNN. `--channels 6`
adds gyro in deg/s and requires a correspondingly trained/exported six-input
model. `--window` and `--stride` must match the model's preprocessing.

Split membership is deterministic by seed (default 42), grouped by participant
before windowing, roughly 70/15/15 with at least one group per held-out split.
Three groups become one train, one validation and one test group. There must be
usable windows for every class in every split; otherwise preparation fails with
the missing coverage. It does not search seeds to optimize results. Never select
a seed based on test accuracy. Exact duplicate raw recordings are rejected.

For a personal-only exploratory dataset, explicitly choose `--group-by session_id`.
This separates recordings, but cannot measure generalization to unseen people or
guarantee different days. Record day information in notes and design your held-out
recordings accordingly. `unlabeled` and `unknown` rows are excluded from supervised
windows; deliberately label background behavior if you want a background class.

| Output | Purpose |
| --- | --- |
| `windows.npz` | `X` float32 `[window, time, channel]`, integer `y`, split, session IDs, labels, feature names |
| `windows.csv` | Source session and timestamp range for each window |
| `deepcraft/{split}/{session}/run_*/` | Continuous `data.csv` and `label.label` pairs for Studio |
| `dataset.json` | Input hashes, parameters, class coverage, group assignments, quality and artifact hashes |

The label IDs in `windows.npz` are alphabetical and saved in `labels`. They are
**not** assumed to match any generated CNN's output order. Check the exported
header and configure the firmware mapping after each training run.

Only contiguous, single-label runs form windows. Cadence checks allow individual
10-30 ms deltas and at most 20 ms cumulative deviation per window. These gates
expose acquisition defects rather than hiding them with resampling. Check timing
statistics before Studio import as well; the exported continuous tracks retain
their original timestamps. Quality reports include peaks and near-saturation
counts for the current driver's +/-4 g and +/-2000 deg/s configuration. Update
those thresholds if you change sensor ranges.

Output directories must be new, so reruns cannot overwrite earlier dataset
versions. Means/stds are computed only from training windows and recorded for
analysis, but **normalization is not applied**. If you add normalization to the
CNN, export that same preprocessing to firmware; never fit it on held-out data.

## Offline Python analysis

### Fixed-group feature export

After preparing a dataset, export window summary features without changing the
assigned splits:

```powershell
python wearable.py export-features --dataset data/processed/belt-v1 --out data/processed/belt-v1-mlcookbook
```

For exploratory tabular statistics or PCA, use the exported `train.csv` with
`label` as the categorical target. Six-axis windows produce 64 physical
unit features (per-axis and acceleration/gyro magnitude mean, population std,
min, max, RMS, median, IQR and mean absolute adjacent-sample difference).
Three-axis windows produce 32 features. The export verifies hashes, row/index
alignment, units, group assignments and class coverage. Output must be new.

`validation.csv` and `test.csv` remain separate reserved holdouts. Companion
`*_index.csv` files preserve lineage; IDs/timestamps/model predictions are
excluded from predictor tables. `export.json` records source/artifact hashes,
feature definitions and the original group assignments.

Use Movement Study for grouped baseline scores. Row-wise splits on overlapping
movement windows are exploratory. Use DEEPCRAFT for temporal CNN training and
MCU C export. Existing scripts can still use the `export-mlcookbook` and
`evaluate-mlcookbook` command aliases. Remove `--cookbook` from older commands.

### NumPy access

```python
import json
import numpy as np
from pathlib import Path

root = Path("data/processed/belt-v1")
report = json.loads((root / "dataset.json").read_text())
with np.load(root / "windows.npz", allow_pickle=False) as data:
    X, y, split = data["X"], data["y"], data["split"]
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_validation, y_validation = X[split == "validation"], y[split == "validation"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    print(data["labels"], X.shape)
print(report["class_window_counts"])
```

Use validation data for model/threshold choices and inspect the held-out test set
once for final per-class precision/recall and confusion matrix. Evaluate new
enclosure revisions and mounting locations as new conditions. Do not infer real
accuracy from the synthetic fixtures in the tests.

## Evaluate fall events separately

The firmware's local LED alarm latches once per event, supports button/CLI
cancellation, and suppresses retriggering for 30 seconds after cancellation.
Evaluate the resulting alerts as well as the CNN's window scores: that cooldown
can hide a second fall and must be represented in continuous evaluation.

Create human ground truth and monitored coverage files in the same device clock:

```text
# truth.csv (omit this comment line from the file)
session_id,event_id,start_ms,end_ms
session_a,fall_001,1000,2500

# alerts.csv -- emitted alerts, not every overlapping CNN window
session_id,ts_ms,confidence
session_a,2400,0.93

# coverage.csv -- fully annotated periods where monitoring was operational
session_id,start_ms,end_ms
session_a,0,3600000
```

Only label a fall when it actually occurs; the interval spans its annotated
onset/end. Long clips containing a brief fall must not have the entire clip
labeled `falling`. Preserve surrounding activity for continuous testing. The
window builder does not automatically make event-centered training clips;
curate those separately while preserving their source recording and timestamps.

Times are half-open intervals `[start_ms,end_ms)`; split recording/coverage at
gaps and reboots. If a session clock resets, use a distinct evaluation session
key for each clock segment. The tool rejects intervals that overlap or events
outside observed coverage. Coverage must be genuinely annotated, including
negative daily activity, rather than inferred from the first/last sample alone.

To derive alert rows from a verified session, filter `events.csv` for
`state == "ALERT"`, use its `ts_ms` and `confidence`, and add its `session_id`.
Keep `TEST` and `CANCEL` as audit evidence; neither is a detector prediction.
Use alert time for latency; `source_ts_ms` is retained to investigate processing
delay. Events are preserved by connected host capture and by the selected
flash v2 sink, even while sensor recording is stopped. Old flash v1 recordings
did not store them. Associate offline alerts only with explicitly monitored
coverage; orphan events are not a substitute for continuous exposure records.

```powershell
python fall_eval.py --truth truth.csv --alerts alerts.csv --coverage coverage.csv --threshold 0.85 --grace-ms 3000 --out fall-report-v1.json
```

The report includes event recall, alert precision, false alerts per non-fall
monitored hour, onset-to-alert latency, each matched/missed event, and input
hashes. One alert matches at most one event. Repeated alerts count as false
alerts; this makes a broken latch/refractory policy visible. Predictions after
the event end can match within `--grace-ms`, but cannot match across a recording
gap. Choose grace/threshold on validation data and freeze them for the test run.
If there are no positive events, recall is undefined (`null`), not 100%.

Do not induce falls while wearing the prototype. See the firmware's belt design
guide for public fall data, controlled fixture tests, sensor-range limits and
the required on-device validation. No fall accuracy has been established yet.

## Assemble an offline Movement Study

After collecting and auditing real sessions, run from this directory:

```powershell
.\venv\Scripts\python.exe wearable.py study --sessions data/sessions --profile belt-v1 --out data/processed/belt-study-01
.\venv\Scripts\python.exe wearable.py verify-study --study data/processed/belt-study-01
.\Open-OfflineWorkflow.ps1 -Study data/processed/belt-study-01
```

`study` performs preparation, feature export and native grouped baseline
evaluation in one **new** directory. It preserves the existing subject/session
split policy, profile validation, quality gates and explicit `--exclude-failed`
audit. For a personal session-disjoint demo, choose `--group-by session_id` and
retain that narrower claim. The command never evaluates the reserved test set.

The output includes `prepared/` (NumPy and continuous DEEPCRAFT tracks),
`mlcookbook/` (the desktop import), `baseline/` (evaluation JSON and validation
predictions), `study.json` (linked SHA-256 hashes and source manifest provenance),
and `START_HERE.md`. The complete folder can move; verification and desktop
reopening do not require the original raw-session paths. Keep raw sessions
separately, because this derived folder does not contain their raw evidence.
Save a self-contained `.mlcs` snapshot in the desktop app to keep the dataset,
settings and last completed evaluation together.

Verification checks every recorded output hash and nested artifact link without
fitting a model. A failed build keeps partial outputs and writes its failing
stage/error to `study.json`; verification refuses it. Correct the source issue
and choose a new output path rather than overwriting evidence.

Session `synthetic: true` now propagates automatically through preparation,
export, native evaluation and saved experiments. Mixing synthetic and real
sessions in one prepared study is rejected. Invented signals are integration
fixtures, never accuracy evidence. Existing lower-level commands remain usable.

## Movement Study desktop app

Use the analysis venv from [Setup and capture](#setup-and-capture). The app
uses Tkinter from the standard Windows Python install and the analysis
requirements. It runs offline without a MLCookbook installation.

```powershell
python movement_study.py
python movement_study.py --study data/processed/belt-study-01
python movement_study.py --experiment path/to/saved.mlcs
```

Opening an export or a snapshot never fits a model. New studies default to
validation-only. **Run grouped baselines** evaluates the chosen settings,
and **Export evaluation…** saves metrics, predictions and source lineage as
JSON. **Open snapshot…** also accepts legacy Movement Study `.mlcb` files.
Completed reports and unrun settings reopen from the embedded dataset even
when the original export has moved. Save an updated copy as `.mlcs`.

The Windows launcher in this directory accepts `-Study` for a complete
study folder or export and `-Experiment` for a saved snapshot.
`-Check` prints the launch command without opening a window.

## Tests

```powershell
python -m unittest discover -s tests -v
python wearable.py --help
```

The MCU model preparation integration tests require the sibling `workingDir/`
checkout and a host C compiler. The offline workflow tests run independently.

Capture/import require only Python's standard library plus pyserial for UART.
Dataset preparation and feature export require NumPy; the other requirements support notebooks and
further offline analysis. No cloud service is used by these Python commands.
