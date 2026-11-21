"""Record a bench acceptance run and verify the serial/device contract.

Leaves evidence on failure. This check never formats flash. The alert option
tests the local visual/cancel controls; it does not simulate a model detection.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import time
from pathlib import Path
import numpy as np
from serial_control import command
from wearable_pipeline import SessionWriter, load_session, write_json


def assess(rows,counts,duration):
    timestamps=np.asarray([int(row["ts_ms"]) for row in rows],dtype=np.int64)
    if len(timestamps)<2:
        return {"rows":len(rows),"passed":False,"error":"at least two samples required"}
    deltas=np.diff(timestamps)
    rate=(len(timestamps)-1)*1000/(timestamps[-1]-timestamps[0]) if timestamps[-1]>timestamps[0] else 0
    result={"rows":len(rows),"observed_hz":float(rate),"largest_interval_ms":int(deltas.max()),
            "minimum_interval_ms":int(deltas.min()),"nonmonotonic_intervals":int((deltas<=0).sum()),
            "intervals_outside_10_30_ms":int(((deltas<10)|(deltas>30)).sum()),
            "rejected_rows":counts.get("rejected",0)}
    result["passed"]=bool(48<=rate<=52 and result["intervals_outside_10_30_ms"]==0
                           and result["rejected_rows"]==0 and len(rows)>=duration*45)
    return result


def run(args):
    import serial
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    session=SessionWriter(root/"sessions",{"subject":"bench","placement":"bench","orientation":"bench",
        "enclosure":"acceptance","firmware":args.expected_build_id or "unknown","sample_rate_hz":50,
        "transport":"serial","purpose":"hardware acceptance; exclude from training"},"standing")
    error=None
    device={}
    try:
        with serial.Serial(args.port,args.baud,timeout=0.2,write_timeout=2) as board:
            try:
                command(board,"stop","[cli] logging stopped",session.raw.write)
                line=command(board,"info","[device]",session.raw.write)
                device=dict(re.findall(r"([A-Za-z_]+)=([^ ]+)",line))
                if device.get("board")!="CY8CKIT-062S2-AI" or device.get("sample_rate_hz")!="50":
                    raise ValueError("device acquisition contract mismatch")
                if args.expected_build_id and device.get("build_id")!=args.expected_build_id:
                    raise ValueError("device build ID differs from the selected image")
                session.metadata["device_contract"]=device
                session.metadata["firmware"]=device.get("build_id","unknown")
                for text,expected in (("mode csv","[cli] mode = csv"),("sink uart","[cli] sink = uart"),
                                      ("label standing","[cli] label = standing")):
                    command(board,text,expected,session.raw.write)
                command(board,"status","[memory]",session.raw.write)
                session.metadata["prestart_raw_bytes"]=session.raw.tell()
                command(board,"start","[cli] logging started",session.feed)
                if args.exercise_alert:
                    command(board,"alert test","[cli] alert test started",session.feed)
                    command(board,"alert status","[cli] alert active=1",session.feed)
                    # Keep the visual indicator observable, then cancel it.
                    until=time.monotonic()+2
                    while time.monotonic()<until:
                        session.feed(board.read(max(1,min(board.in_waiting,4096))))
                    command(board,"alert cancel","[cli] alert cancelled",session.feed)
                    command(board,"alert status","[cli] alert active=0",session.feed)
                until=time.monotonic()+args.duration
                while time.monotonic()<until:
                    chunk=board.read(max(1,min(board.in_waiting,4096)))
                    if chunk:session.feed(chunk)
            finally:
                command(board,"stop","[cli] logging stopped",session.feed)
                command(board,"status","[memory]",session.feed)
    except (Exception,KeyboardInterrupt) as exc:
        error=str(exc) or "interrupted"
    finally:
        session.close("failed" if error else "complete",error)
    report={"device":device,"session":str(session.path),"error":error,"passed":False,
            "physical_checks_pending":["LED visibility and button cancellation while worn",
                "flash stop/download, power interruption and sector crossing on actual NOR",
                "current, charging temperature and battery runtime","model accuracy and inference timing"]}
    if not error:
        metadata,rows=load_session(session.path)
        report["sampling"]=assess(rows,metadata["counts"],args.duration)
        raw=(session.path/"raw.log").read_text(errors="replace")
        snapshots=re.findall(r"\[flash\] status [^\r\n]*queue_dropped=(\d+) logger_dropped=(\d+)",raw)
        heaps=re.findall(r"minimum_free_heap=(\d+)",raw)
        report["no_new_logger_drops"]=bool(len(snapshots)>=2 and snapshots[0][1]==snapshots[-1][1])
        report["minimum_free_heap_bytes"]=min(map(int,heaps)) if heaps else None
        report["alert_events_verified"]=bool("[fall] state=TEST" in raw and "[fall] state=CANCEL" in raw) if args.exercise_alert else None
        report["passed"]=bool(report["sampling"]["passed"] and report["no_new_logger_drops"]
            and report["minimum_free_heap_bytes"] is not None and report["minimum_free_heap_bytes"]>=8192
            and (not args.exercise_alert or report["alert_events_verified"]))
    write_json(root/"acceptance.json",report)
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",required=True)
    p.add_argument("--baud",type=int,default=115200)
    p.add_argument("--out",type=Path,required=True,help="new evidence directory")
    p.add_argument("--duration",type=float,default=30)
    p.add_argument("--expected-build-id")
    p.add_argument("--exercise-alert",action="store_true")
    args=p.parse_args()
    if not math.isfinite(args.duration) or not 5<=args.duration<=600:p.error("duration must be 5..600 seconds")
    try:report=run(args)
    except (OSError,ValueError,RuntimeError) as exc:p.exit(1,f"Acceptance failed: {exc}\n")
    if not report["passed"]:p.exit(1,"Acceptance did not pass; retain the report and raw evidence.\n")


if __name__=="__main__":main()
