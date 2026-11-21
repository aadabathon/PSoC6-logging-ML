"""Acknowledged board commands. Incoming bytes are always retained by caller."""
import time
import re


def command(connection, text, expected, retain, timeout=60):
    connection.write((text + "\r\n").encode("ascii"))
    buffer = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = connection.read(max(1, min(connection.in_waiting, 4096)))
        if not chunk:
            continue
        retain(chunk)
        buffer.extend(chunk)
        if len(buffer) > 16384:
            raise RuntimeError("oversized command acknowledgement")
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer = bytearray(remainder)
            decoded = line.decode("utf-8", errors="replace").strip()
            if "[cli] ERROR" in decoded:
                raise RuntimeError(decoded)
            if expected in decoded:
                return decoded
    raise TimeoutError(f"board did not acknowledge {text!r}")


def stop_for_inspection(connection,retain):
    """Allow a read-only recovery dump after an explicitly unclean stop.

    Never infer that recording stopped merely from an error or elapsed time.
    The serialized status acknowledgment must confirm the recording gate is off.
    """
    try:
        command(connection,"stop","[cli] logging stopped",retain)
        return True
    except RuntimeError as exc:
        if "stop not committed" not in str(exc):raise
        status=command(connection,"status","[cli] mode=",retain)
        if not re.search(r"\benabled=0\b",status):
            raise RuntimeError("recovery dump requires recording stopped") from exc
        return False
