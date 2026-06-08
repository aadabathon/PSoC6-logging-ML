#!/usr/bin/env python3
"""
collect.py -- capture CSV sensor data streaming from the PSoC6 over serial.

Usage:
    python collect.py --port COM5 --label walking --duration 30
    python collect.py --port COM5 --label still          # runs until Ctrl-C

Expects the firmware in CSV mode, emitting a header row then data rows:
    ts_ms,baro_pa,baro_t,ax,ay,az,gx,gy,gz,imu_t,mx,my,mz,mag_t,radar_present,radar_bin,mic_rms,mic_peak
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True, help="activity label, e.g. walking")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (omit = until Ctrl-C)")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = outdir / f"{stamp}_{args.label}.csv"

    print(f"Opening {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()   # drop any partial line in the buffer

    header = None
    rows = 0
    start = time.time()
    print(f"Recording to {fname}  (label='{args.label}')  Ctrl-C to stop")

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        try:
            while True:
                if args.duration and (time.time() - start) >= args.duration:
                    break
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # skip human-mode log lines if any slipped through
                if line.startswith("["):
                    continue
                fields = line.split(",")
                # first CSV-looking line is the header
                if header is None:
                    if "ts_ms" in line:
                        header = fields + ["label"]
                        writer.writerow(header)
                        print("Header:", ",".join(header))
                    continue
                # data row -- validate column count, append label
                if len(fields) == len(header) - 1:
                    writer.writerow(fields + [args.label])
                    rows += 1
                    if rows % 100 == 0:
                        print(f"  {rows} rows", end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()

    elapsed = time.time() - start
    print(f"\nSaved {rows} rows to {fname} ({elapsed:.1f}s, "
          f"{rows/elapsed:.0f} rows/s)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()