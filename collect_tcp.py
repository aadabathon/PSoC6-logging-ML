#!/usr/bin/env python3
"""TCP capture using the same validated session format as UART."""
import sys
from wearable import main
from wearable_pipeline import CSV_COLUMNS

WIFI_TCP_PORT = 50007

if __name__ == "__main__":
    main(["capture", "--transport", "tcp", *sys.argv[1:]])
