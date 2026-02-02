from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

DATE_PATTERNS = [
    re.compile(r"^(?P<year>\d{4})$"),
    re.compile(r"^(?P<year>\d{4})[-_](?P<month>\d{2})$"),
    re.compile(r"^(?P<year>\d{4})[-_](?P<month>\d{2})[-_](?P<day>\d{2})$"),
    re.compile(r"^(?P<month>\d{1,2})[-_](?P<day>\d{1,2})[-_](?P<year>\d{2,4})$"),
    re.compile(r"^(?P<month>\d{1,2})[-_](?P<day>\d{1,2})[-_](?P<year>\d{2,4})\(\d+\)$"),
    re.compile(r"^Photos from (?P<year>\d{4})$", re.IGNORECASE),
]


def is_year_folder(name: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", name))


def is_date_folder(name: str, parent: Optional[str] = None) -> bool:
    if name.isdigit() and len(name) in {1, 2}:
        if parent and is_year_folder(parent):
            return True
    for pattern in DATE_PATTERNS:
        if pattern.match(name):
            return True
    return False


def parse_date_from_folder(name: str) -> Optional[datetime]:
    for pattern in DATE_PATTERNS:
        match = pattern.match(name)
        if not match:
            continue
        parts = match.groupdict()
        year = int(parts.get("year") or 0)
        if year and year < 100:
            year += 2000 if year <= 69 else 1900
        month = int(parts.get("month") or 1)
        day = int(parts.get("day") or 1)
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None
