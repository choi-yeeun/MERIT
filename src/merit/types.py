"""Shared value types for the memory module."""

from dataclasses import dataclass
from typing import Optional, Tuple


def transform_timestamp(ts_str: str) -> str:
    """``DHHMMSS00`` -> ``DAY<d> HH:MM:SS``. e.g. ``1111209300`` -> ``DAY1 11:12:09``."""
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


@dataclass
class CaptionEntry:
    """One 30-second clip: its caption text, time span and source video."""

    id: str
    text: str
    start_time: str
    end_time: str
    date: str
    video_path: Optional[str] = None

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        """Start/end as ``DHHMMSS00`` integers (day digit + zero-padded time)."""
        if isinstance(self.date, int):
            day = str(self.date)
        else:
            day = str(self.date).replace("DAY", "").replace("Day", "")
        start_ts = int(day + str(self.start_time).zfill(8))
        end_ts = int(day + str(self.end_time).zfill(8))
        return start_ts, end_ts
