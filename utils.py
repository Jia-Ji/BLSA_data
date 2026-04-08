import sys
from contextlib import redirect_stdout
from pathlib import Path

class _Tee:
    """Write to multiple file-like objects (e.g. console + main_log.txt)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()