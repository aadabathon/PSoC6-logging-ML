#!/usr/bin/env python3
"""
collect.py -- capture CSV sensor data streaming from the PSoC6 over serial.

Usage:
    python collect.py --label walking --duration 30
    python collect.py --label still          # runs until Ctrl-C
    python collect.py --port COM7 --label walking --outdir C:/tmp/data

Expects the firmware in CSV mode (source/logging/logger.c, LOG_MODE_CSV),
emitting a header row once at boot then data rows:
    ts_ms,baro_pa,baro_t,ax,ay,az,gx,gy,gz,imu_t,mx,my,mz,mag_t,radar_present,radar_bin,mic_rms,mic_peak

NOTE: on this pipeline only the imu_* columns are real right now -- baro/mag/
radar/mic tasks aren't started (main.c only brings up i2c_bus + imu_task), so
those columns will just read 0. That's expected, not a capture bug.

NOTE: the firmware only prints the CSV header once, right after boot
(logger_task's header_printed latch never resets since nothing switches log
modes at runtime anymore). If you start this script after the board's
already been streaming for a while, you will never see that header line on
the wire. So we don't wait for it -- the column layout is hardcoded below to
match logger.c, and we sniff data rows by shape instead of waiting for a
literal header to arrive.
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

# Must match logger.c's print_csv_header() column order exactly.
CSV_COLUMNS = [
    "ts_ms", "baro_pa", "baro_t", "ax", "ay", "az", "gx", "gy", "gz", "imu_t",
    "mx", "my", "mz", "mag_t", "radar_present", "radar_bin", "mic_rms", "mic_peak",
]

# This file lives at analysis/collect.py -- data/ is right next to it.
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "data"


def is_data_row(fields):
    """A real CSV row: right column count, and ts_ms (first field) is an int."""
    if len(fields) != len(CSV_COLUMNS):
        return False
    try:
        int(fields[0])
    except ValueError:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="serial port (default: COM3)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True, help="activity label, e.g. walking")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (omit = until Ctrl-C)")
    ap.add_argument("--outdir", default=None,
                    help=f"default: {DEFAULT_OUTDIR}")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = outdir / f"{stamp}_{args.label}.csv"

    print(f"Opening {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()   # drop any partial line in the buffer

    header = CSV_COLUMNS + ["label"]
    rows = 0
    start = time.time()
    print(f"Recording to {fname}  (label='{args.label}')  Ctrl-C to stop")

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while True:
                if args.duration and (time.time() - start) >= args.duration:
                    break
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # skip human-mode log lines if any slip through, e.g. "[imu] a=..."
                if line.startswith("["):
                    continue
                # skip the wire header if we happen to catch it (fresh boot) --
                # we already wrote our own hardcoded header above.
                if "ts_ms" in line:
                    print("Header seen on wire:", line)
                    continue

                fields = line.split(",")
                if is_data_row(fields):
                    writer.writerow(fields + [args.label])
                    rows += 1
                    if rows % 100 == 0:
                        print(f"  {rows} rows", end="\r")
                # else: boot banner / garbled / partial line -- ignore
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()

    elapsed = time.time() - start
    print(f"\nSaved {rows} rows to {fname} ({elapsed:.1f}s, "
          f"{rows/elapsed:.0f} rows/s)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()
