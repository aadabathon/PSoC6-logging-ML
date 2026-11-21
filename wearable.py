#!/usr/bin/env python3
"""Capture, import, audit, and prepare wearable movement recordings."""
import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

from wearable_pipeline import (LABEL_RE, SessionWriter, import_recording,
                               load_session, prepare_dataset)

DEFAULT_OUTDIR = Path(__file__).resolve().parent / "data" / "sessions"


def add_metadata(parser):
    parser.add_argument("--subject", default="unknown", help="pseudonymous participant ID")
    parser.add_argument("--placement", default="unknown", help="e.g. left_wrist, waist")
    parser.add_argument("--orientation", default="unknown", help="board axes relative to body")
    parser.add_argument("--enclosure", default="unhoused", help="printed enclosure revision")
    parser.add_argument("--device", default="CY8CKIT-062S2-AI")
    parser.add_argument("--firmware", default="unknown", help="flashed firmware revision, not host git HEAD")
    parser.add_argument("--notes", default="")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)


def metadata(args):
    result = {key: getattr(args, key) for key in
              ("subject", "placement", "orientation", "enclosure", "device", "firmware", "notes")}
    result["sample_rate_hz"] = 50
    return result


def capture(args):
    if args.transport == "serial":
        import serial
    session = SessionWriter(args.outdir, dict(metadata(args), transport=args.transport), args.label)
    connection = None
    status, error = "complete", None
    print(f"Recording session: {session.path}")
    try:
        if args.transport == "serial":
            from serial_control import command
            connection = serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=2)
            time.sleep(0.5)
            # Retain pre-start bytes only in raw.log, outside this session's data.
            command(connection,"stop","[cli] logging stopped",session.raw.write)
            for text, expected in (("mode csv","[cli] mode = csv"),("sink uart","[cli] sink = uart"),
                                   (f"label {args.label}",f"[cli] label = {args.label}")):
                command(connection,text,expected,session.raw.write)
            session.metadata["prestart_raw_bytes"] = session.raw.tell()
            command(connection,"start","[cli] logging started",session.feed)
        else:
            connection = socket.create_connection((args.host, args.port), timeout=10)
            connection.settimeout(0.2)
            print(f"TCP is receive-only: set `mode csv`, `sink tcp`, `label {args.label}`, `start` on the board.")
        started = time.monotonic()
        last_sample = started
        previous_rows = 0
        while args.duration is None or time.monotonic() - started < args.duration:
            try:
                chunk = (connection.read(max(1, min(connection.in_waiting, 4096)))
                         if args.transport == "serial" else connection.recv(4096))
            except socket.timeout:
                chunk = None
            if args.transport == "tcp" and chunk == b"":
                raise ConnectionError("board disconnected; session retained, reconnect as a new session")
            if chunk:
                session.feed(chunk)
            rows = session.counts["rows"]
            if rows != previous_rows:
                last_sample = time.monotonic()
                previous_rows = rows
            if time.monotonic() - last_sample > args.idle_timeout:
                raise TimeoutError("no valid IMU data before idle timeout; check board mode, sink, and label")
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "failed", str(exc)
    finally:
        if connection is not None:
            try:
                if args.transport == "serial":
                    from serial_control import command
                    command(connection,"stop","[cli] logging stopped",session.feed)
            except Exception as exc:
                status, error = "failed", error or f"could not stop recording: {exc}"
            finally:
                connection.close()
        session.close(status, error)
    print(json.dumps(session.metadata["counts"], indent=2))
    if session.metadata["status"] == "failed":
        raise RuntimeError(session.metadata["error"])
    return session.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("capture", help="record UART or TCP to an auditable session")
    record.add_argument("--transport", choices=("serial", "tcp"), default="serial")
    record.add_argument("--port", help="COM port for UART or integer TCP port")
    record.add_argument("--host", help="board IP for TCP")
    record.add_argument("--baud", type=int, default=115200)
    record.add_argument("--label", required=True)
    record.add_argument("--duration", type=float)
    record.add_argument("--idle-timeout", type=float, default=15)
    add_metadata(record)
    ingest = commands.add_parser("import", help="ingest saved CSV or UART flash dump; never changes source")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--label", help="optional expected ground truth; rejects mismatches")
    add_metadata(ingest)
    audit = commands.add_parser("audit", help="verify finalized session hashes and print metadata")
    audit.add_argument("session", type=Path)
    prepare = commands.add_parser("prepare", help="make NumPy windows and Deepcraft tracks")
    prepare.add_argument("--sessions", nargs="+", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--group-by", choices=("subject", "session_id"), default="subject")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--profile", choices=("belt-v1",), help="six-axis belt study defaults and label/placement validation")
    prepare.add_argument("--window", type=int)
    prepare.add_argument("--stride", type=int)
    prepare.add_argument("--channels", type=int, choices=(3, 6))
    prepare.add_argument("--exclude-failed",action="store_true",help="explicitly exclude failed/unfinalized sessions and record them in the dataset manifest")
    export = commands.add_parser("export-features", aliases=["export-mlcookbook"],
                                 help="export verified windows as split-preserving feature CSVs")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    flash = commands.add_parser("import-flash",help="verify a complete v2 dump and recover separate physical sessions")
    flash.add_argument("source",type=Path)
    add_metadata(flash)
    download = commands.add_parser("download-flash",help="stop, download and verify flash; never erases")
    download.add_argument("--port",required=True)
    download.add_argument("--baud",type=int,default=115200)
    download.add_argument("--dest",type=Path,required=True,help="new raw evidence file")
    add_metadata(download)
    setup = commands.add_parser("configure-flash",help="configure stopped board for battery/button collection")
    setup.add_argument("--port",required=True)
    setup.add_argument("--baud",type=int,default=115200)
    setup.add_argument("--label",required=True)
    setup.add_argument("--transcript",type=Path,required=True,help="new setup evidence file")
    setup.add_argument("--format",action="store_true",help="explicitly DISCARD old flash data and create v2 format")
    evaluation = commands.add_parser("evaluate", aliases=["evaluate-mlcookbook"],
                                     help="evaluate fixed holdouts with group-disjoint training CV")
    evaluation.add_argument("--dataset", type=Path, required=True, help="export-features output")
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--folds", type=int, choices=(0, 2, 3, 5, 10), default=5)
    evaluation.add_argument("--final-test", action="store_true", help="explicitly evaluate the frozen test holdout")
    study = commands.add_parser("study", help="prepare, export and run native baselines in one new offline study")
    study.add_argument("--sessions", nargs="+", type=Path, required=True)
    study.add_argument("--out", type=Path, required=True)
    study.add_argument("--profile", choices=("belt-v1",), default="belt-v1")
    study.add_argument("--group-by", choices=("subject", "session_id"), default="subject")
    study.add_argument("--seed", type=int, default=42)
    study.add_argument("--exclude-failed", action="store_true", help="explicit audited exclusion of failed/unfinalized sessions")
    verify = commands.add_parser("verify-study", help="verify a completed offline study without training")
    verify.add_argument("--study", type=Path, required=True, help="study folder or study.json")
    args = parser.parse_args(argv)
    if getattr(args, "label", None) and not LABEL_RE.fullmatch(args.label):
        parser.error("label must be 1-15 characters in [A-Za-z0-9_-]")
    try:
        if args.command == "capture":
            if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
                parser.error("duration must be finite and positive")
            if not math.isfinite(args.idle_timeout) or args.idle_timeout <= 0:
                parser.error("idle-timeout must be finite and positive")
            if args.transport == "tcp":
                if not args.host:
                    parser.error("--host is required for TCP")
                args.port = int(args.port or 50007)
            else:
                args.port = args.port or "COM3"
            print(capture(args))
        elif args.command == "import":
            print(import_recording(args.source, args.outdir, metadata(args), args.label))
        elif args.command == "audit":
            meta, _ = load_session(args.session)
            print(json.dumps(meta, indent=2))
        elif args.command in ("import-flash","download-flash"):
            from flash_transfer import download_flash, import_flash
            source = (download_flash(args.port,args.baud,args.dest)
                      if args.command == "download-flash" else args.source)
            batch, report = import_flash(source,args.outdir,metadata(args))
            print(json.dumps(report,indent=2))
            print(f"Recovered batch: {batch}")
        elif args.command == "configure-flash":
            import serial
            from serial_control import command
            args.transcript.parent.mkdir(parents=True,exist_ok=True)
            with args.transcript.open("xb") as transcript, serial.Serial(args.port,args.baud,timeout=0.2,write_timeout=2) as board:
                command(board,"stop","[cli] logging stopped",transcript.write)
                if args.format:
                    command(board,"erase confirm","[cli] flash formatted version=2",transcript.write)
                for text, expected in (("mode csv","[cli] mode = csv"),
                                       (f"label {args.label}",f"[cli] label = {args.label}"),
                                       ("sink flash","[cli] sink = flash"),("info","[device]"),("status","[flash] status")):
                    command(board,text,expected,transcript.write)
            print("Configured. Keep battery power connected; USER BTN1 starts/stops. Wait for LED1 to turn off before power removal.")
        elif args.command in ("export-features", "export-mlcookbook"):
            from mlcookbook_export import LIMITATION, export_mlcookbook
            report = export_mlcookbook(args.dataset, args.out)
            print(f"Exported {len(report['features'])} features to {args.out}")
            print(LIMITATION)
        elif args.command in ("evaluate", "evaluate-mlcookbook"):
            from movement_evaluate import evaluate_export
            result = evaluate_export(args.dataset, args.out, final_test=args.final_test,
                                     seed=args.seed, folds=args.folds)
            print(json.dumps(result, indent=2))
        elif args.command == "study":
            from offline_study import build_study
            profile = json.loads((Path(__file__).parent / "profiles" / (args.profile + ".json")).read_text())
            result = build_study(args.sessions, args.out, group_by=args.group_by,
                                 seed=args.seed, profile=profile, exclude_failed=args.exclude_failed)
            print(json.dumps({"status": result["status"], "synthetic": result["synthetic"],
                              "selected_model": result["selected_model"], "final_test_evaluated": False,
                              "start_here": str((args.out / "START_HERE.md").resolve())}, indent=2))
        elif args.command == "verify-study":
            from offline_study import verify_study
            print(json.dumps(verify_study(args.study), indent=2))
        else:
            paths = []
            for path in args.sessions:
                paths.extend([path] if (path / "session.json").is_file()
                             else [item.parent for item in path.rglob("session.json")])
            profile = None
            if args.profile:
                profile = json.loads((Path(__file__).parent / "profiles" / (args.profile + ".json")).read_text())
            defaults = profile or {"window_samples": 50, "stride_samples": 30, "channels": 3}
            result = prepare_dataset(paths, args.out, group_by=args.group_by, seed=args.seed,
                                     window=args.window if args.window is not None else defaults["window_samples"],
                                     stride=args.stride if args.stride is not None else defaults["stride_samples"],
                                     channels=args.channels if args.channels is not None else defaults["channels"],
                                     profile=profile,exclude_failed=args.exclude_failed)
            if result["excluded_sessions"]:
                print(f"Excluded {len(result['excluded_sessions'])} failed/unfinalized sessions; reasons retained in dataset.json")
            print(json.dumps(result["class_window_counts"], indent=2))
            print(f"Dataset: {args.out}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
