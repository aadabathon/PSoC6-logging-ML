#!/usr/bin/env python3
"""UART capture with raw evidence, schema checks, and session metadata.

Existing --port/--baud/--label/--duration/--outdir options remain supported.
Each recording is now a session directory. See README.md.
"""
import sys
from wearable import main
from wearable_pipeline import CSV_COLUMNS, RowParser, ProtocolError


def is_data_row(fields):
    try:
        return RowParser().parse(",".join(fields)) is not None
    except ProtocolError:
        return False


if __name__ == "__main__":
    main(["capture", "--transport", "serial", *sys.argv[1:]])
