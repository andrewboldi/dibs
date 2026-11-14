"""Sizes and durations as people and agents type them: 16G, 512M, 20m, 1h30m."""

import math
import re

KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30
TiB = 1 << 40

_SIZE_UNITS = {"K": KiB, "M": MiB, "G": GiB, "T": TiB}
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([KMGT])(?:I?B)?", re.IGNORECASE)

_DURATION_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_DURATION_PART = r"(\d+(?:\.\d+)?)([dhms])"
_DURATION_RE = re.compile(rf"(?:{_DURATION_PART})+")
_CLOCK_RE = re.compile(r"\d+(?::\d{1,2}){1,2}")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class UnitError(ValueError):
    """A size or duration that can't be understood."""


def parse_size(text: str) -> int:
    """Bytes in a size like 512M, 16G, 16GB, 16GiB or 1.5G. Units are binary: G means GiB."""
    s = text.strip()
    if s == "0":
        return 0
    if _NUMBER_RE.fullmatch(s):
        raise UnitError(f"size {text!r} needs a unit: add one, like {s}G or {s}M")
    m = _SIZE_RE.fullmatch(s)
    if not m:
        raise UnitError(f"can't read size {text!r}: use a number and a unit, like 16G or 512M")
    return round(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()])


def parse_duration(text: str) -> float:
    """Seconds in a duration like 90s, 20m, 1h30m, 1.5h, 2d, HH:MM:SS or MM:SS. A bare number is minutes."""
    s = text.strip()
    if _NUMBER_RE.fullmatch(s):
        seconds = float(s) * 60
    elif _CLOCK_RE.fullmatch(s):
        seconds = 0.0
        for part in s.split(":"):
            seconds = seconds * 60 + int(part)
    elif _DURATION_RE.fullmatch(s):
        seconds = sum(
            float(number) * _DURATION_UNITS[unit] for number, unit in re.findall(_DURATION_PART, s)
        )
    else:
        raise UnitError(f"can't read duration {text!r}: use something like 90s, 20m, 1h30m or 2d")
    if seconds <= 0:
        raise UnitError(f"duration {text!r} must be more than 0")
    return seconds


def format_size(n: int) -> str:
    """A compact size: 6G, 3.1G, 512M, 0."""
    for unit, size in (("T", TiB), ("G", GiB)):
        if n >= size:
            value = n / size
            text = f"{value:.0f}" if value >= 100 else f"{value:.1f}".removesuffix(".0")
            return text + unit
    if n >= MiB:
        return f"{n / MiB:.0f}M"
    if n >= KiB:
        return f"{n / KiB:.0f}K"
    return str(n)


def format_duration(seconds: float) -> str:
    """A compact duration: 45s, 8m, 1h30m, 2d3h. Below a day, minutes are rounded down."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        hours, minutes = s // 3600, (s % 3600) // 60
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    days, hours = s // 86400, (s % 86400) // 3600
    return f"{days}d{hours}h" if hours else f"{days}d"


def suggest_size(n: int) -> int:
    """A reservation to suggest for something seen using n bytes: n plus 10%, rounded up to a whole GiB."""
    return max(1, math.ceil(n * 1.1 / GiB)) * GiB
