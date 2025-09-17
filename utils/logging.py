from __future__ import annotations
from datetime import datetime

def nice(msg: str) -> str:
    return f"[ {datetime.now().strftime('%H:%M:%S')} ] {msg}"
