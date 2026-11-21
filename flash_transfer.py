"""CRC-checked flash-v2 download/import. Never erase the board automatically."""
from __future__ import annotations

import re
import shutil
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from serial_control import command, stop_for_inspection
from wearable_pipeline import LABEL_RE, SessionWriter, sha256, utc_now, write_json

BEGIN = re.compile(r"\[flash\] dump begin version=2 generation=(\d+) used=(\d+) valid=(\d+) damaged=(\d+)")
END = re.compile(r"\[flash\] dump complete version=2 generation=(\d+) used=(\d+)")
FRAME = re.compile(r"\[flash\] v2 crc32=([0-9a-fA-F]{8}) payload=(.*)")
GAP = re.compile(r"\[flash\] gap sequence=(\d+)")


@dataclass(frozen=True)
class Record:
    sequence: int
    session: int
    dropped: int
    kind: int
    text: str
    raw: bytes


class DumpReader:
    def __init__(self, source):
        self.source = Path(source)
        self.info = None

    def __iter__(self):
        begin = None
        ended = False
        expected, valid, damaged = 1, 0, 0
        with self.source.open("rb") as stream:
            for raw in stream:
                if len(raw) > 8192:
                    raise ValueError("oversized flash line")
                line = raw.decode("utf-8", errors="strict").strip()
                if match := BEGIN.fullmatch(line):
                    if begin is not None:
                        raise ValueError("duplicate flash dump start")
                    begin = tuple(map(int, match.groups()))
                    if not 0 < begin[0] < 2**32 or not 0 <= begin[1] <= 261120 or begin[2] + begin[3] != begin[1]:
                        raise ValueError("invalid flash extent")
                    continue
                if not begin:
                    continue
                if ended:
                    if line.startswith("[flash] v2") or BEGIN.fullmatch(line):
                        raise ValueError("records after flash dump end")
                    continue
                if match := END.fullmatch(line):
                    if tuple(map(int, match.groups())) != begin[:2] or expected != begin[1] + 1:
                        raise ValueError("flash dump end/sequence mismatch")
                    if (valid, damaged) != begin[2:]:
                        raise ValueError("flash dump record counts mismatch")
                    ended = True
                    continue
                if match := GAP.fullmatch(line):
                    sequence = int(match.group(1))
                    if sequence != expected:
                        raise ValueError("flash gap sequence mismatch")
                    expected += 1
                    damaged += 1
                    yield Record(sequence, 0, 0, 0, "damaged flash slot", raw)
                    continue
                if match := FRAME.fullmatch(line):
                    checksum, payload = match.groups()
                    if zlib.crc32(payload.encode("utf-8")) != int(checksum, 16):
                        raise ValueError("flash wire CRC mismatch")
                    try:
                        fields, text = payload.split("|", 1)
                        sequence, session, dropped, kind = map(int, fields.split(","))
                    except ValueError as exc:
                        raise ValueError("invalid flash record metadata") from exc
                    if (sequence != expected or not 0 <= session < 2**32
                            or not 0 <= dropped < 2**32 or kind not in range(1, 6)):
                        raise ValueError("invalid flash sequence/session/kind")
                    expected += 1
                    valid += 1
                    yield Record(sequence, session, dropped, kind, text, raw)
                elif line.startswith("[flash]"):
                    raise ValueError(f"unexpected flash diagnostic: {line}")
        if not begin or not ended:
            raise ValueError("incomplete flash dump; committed end marker required")
        self.info = dict(generation=begin[0], used=begin[1], valid=valid, damaged=damaged)


def import_flash(source, root, metadata):
    source = Path(source)
    reader = DumpReader(source)
    for _ in reader:
        pass  # validate the entire transfer before finalizing any session
    source_hash = sha256(source)
    batch = Path(root) / ("flash_" + uuid4().hex[:12])
    batch.mkdir(parents=True, exist_ok=False)
    evidence = batch / "download.log"
    shutil.copyfile(source, evidence)
    if sha256(evidence) != source_hash:
        raise ValueError("flash source changed during import")
    current = None
    stopped = False
    faults = []
    initial_dropped = 0
    physical_id = None
    completed_ids = set()
    results = []
    orphan_records = 0
    start_ts = last_sample_ts = None

    def finish():
        nonlocal current
        if current is None:
            return
        reason = "; ".join(faults) or (None if stopped else "recording has no committed STOP; recovered prefix retained")
        current.close("imported" if reason is None else "failed", reason)
        results.append({"path": str(current.path.relative_to(batch)),
                        "status": current.metadata["status"], "error": current.metadata.get("error"),
                        "flash_session": current.metadata["flash_session"]})
        current = None

    try:
        for record in DumpReader(evidence):
            if record.kind == 4:  # boot separates device-clock segments
                finish()
                physical_id = None
                continue
            if record.kind == 2:
                finish()
                if record.session == 0 or record.session in completed_ids:
                    raise ValueError("invalid or duplicate physical flash session")
                completed_ids.add(record.session)
                match = re.fullmatch(r"\[flash\] session ts_ms=(\d+) label=([A-Za-z0-9_-]{1,15})", record.text)
                if not match or not LABEL_RE.fullmatch(match.group(2)):
                    raise ValueError("invalid flash session boundary")
                start_ts=int(match.group(1))
                if not 0<=start_ts<2**32:raise ValueError("invalid boundary timestamp")
                last_sample_ts=None
                physical_id = record.session
                initial_dropped = record.dropped
                stopped = False
                faults = []
                current = SessionWriter(batch / "sessions", dict(metadata, transport="flash-v2",
                    flash_generation=reader.info["generation"], flash_session=physical_id,
                    source_download_sha256=source_hash, flash_start_sequence=record.sequence,
                    flash_start_ts_ms=start_ts,flash_wire_crc_verified=True), match.group(2))
                current.raw.write(record.raw)
                continue
            if record.kind == 0:
                if current:
                    current.raw.write(record.raw)
                    faults.append(f"damaged flash slot {record.sequence}")
                continue
            if current is None:
                orphan_records += 1
                continue
            current.raw.write(record.raw)
            if record.session != physical_id:
                faults.append("physical session mismatch")
                continue
            if not stopped and record.dropped != initial_dropped:
                faults.append("device reported dropped records during capture")
            if record.kind == 3:
                stop=re.fullmatch(r"\[flash\] stop ts_ms=(\d+) label=", record.text)
                if stopped or not stop:
                    faults.append("invalid STOP boundary")
                else:
                    stop_ts=int(stop.group(1))
                    if not 0<=stop_ts<2**32 or (stop_ts-(last_sample_ts if last_sample_ts is not None else start_ts))%2**32>=2**31:
                        faults.append("STOP timestamp precedes capture")
                    current.metadata["flash_stop_ts_ms"]=stop_ts
                stopped = True
                current.metadata["flash_stop_sequence"] = record.sequence
                continue
            if record.kind == 1:
                if stopped:
                    faults.append("sample after STOP")
                    continue
                before=current.counts["rows"]
                current.line(record.text.encode("utf-8"))
                if current.counts["rows"]!=before+1:
                    faults.append("invalid sample payload")
                else:
                    timestamp=current.previous_ts
                    if ((timestamp-start_ts)%2**32>=2**31 or
                            (last_sample_ts is not None and not 0<(timestamp-last_sample_ts)%2**32<2**31)):
                        faults.append("sample timestamp outside ordered capture")
                    last_sample_ts=timestamp
            elif record.kind == 5:
                if not record.text.startswith("[fall] state="):
                    faults.append("invalid alert payload")
                else:
                    before=current.counts["fall_events"]
                    current.line(record.text.encode("utf-8"))
                    if current.counts["fall_events"]!=before+1:faults.append("invalid alert payload")
        finish()
    except BaseException as exc:
        if current:
            faults.append(str(exc))
            finish()
        raise
    summary = dict(schema_version=1, created_utc=utc_now(), source_sha256=source_hash,
                   flash=reader.info, sessions=results, orphan_records=orphan_records)
    write_json(batch / "import.json", summary)
    return batch, summary


def download_flash(port, baud, destination, timeout=120):
    import serial
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw, serial.Serial(port, baud, timeout=0.2, write_timeout=2) as connection:
        stop_for_inspection(connection,raw.write)
        connection.write(b"dump\r\n")
        buffer = bytearray()
        last_data = time.monotonic()
        while time.monotonic() - last_data < timeout:
            chunk = connection.read(max(1, min(connection.in_waiting, 4096)))
            if not chunk:
                continue
            raw.write(chunk)
            buffer.extend(chunk)
            last_data = time.monotonic()
            if len(buffer) > 16384:
                raise ValueError("oversized flash transfer line")
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                text = line.decode("utf-8", errors="strict").strip()
                if "[cli] ERROR" in text or "[flash] ERROR" in text:
                    raise RuntimeError(text)
                if END.fullmatch(text):
                    raw.flush()
                    for _ in DumpReader(destination):
                        pass
                    return destination
        raise TimeoutError("flash download stalled; partial evidence retained")
