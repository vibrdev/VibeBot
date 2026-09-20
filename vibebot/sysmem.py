"""How much memory can we actually still commit?

This module exists because of a failure that looked impossible: on a 16 GB
machine with 7 GB of RAM reported free, loading Laya killed the process with a
segfault, and torch refused an 8 MB tensor with "DefaultCPUAllocator: not
enough memory".

Measured on the machine in question (Windows 11, 16 GB):

    Committed : 22.5 GB
    Limit     : 25.2 GB
    HEADROOM  :  2.8 GB      <- what a new allocation must fit in
    Available physical: 7,009 MB

Free *physical* RAM is the wrong number. Windows charges every allocation
against the commit limit (RAM + pagefile), and refuses it when the charge does
not fit, however much RAM happens to be idle. A C extension like torch takes
that refusal as a fatal error rather than a Python exception, so the run dies
without a traceback.

So: gate on headroom, not on free RAM.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Memory:
    total_mb: float
    """Physical RAM installed."""
    available_mb: float
    """Physical RAM free right now. Informational — not what allocations check."""
    headroom_mb: float
    """Commit that is still grantable. This is the number that matters."""
    committed_mb: float
    limit_mb: float

    def short(self) -> str:
        return (
            f"{self.headroom_mb:,.0f} MB headroom "
            f"({self.committed_mb:,.0f}/{self.limit_mb:,.0f} MB committed, "
            f"{self.available_mb:,.0f} MB RAM free)"
        )


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


_MB = 1024 * 1024


def snapshot() -> Memory | None:
    """Current memory picture, or None on a platform we cannot read."""
    try:
        return _windows() if _is_windows() else _linux()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break a run
        log.debug("memory snapshot unavailable: %s", exc)
        return None


def _is_windows() -> bool:
    return hasattr(ctypes, "windll")


def _windows() -> Memory:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    # ullTotalPageFile / ullAvailPageFile are the *commit* limit and what is
    # left of it, despite the name — they are not the pagefile's own size.
    return Memory(
        total_mb=status.ullTotalPhys / _MB,
        available_mb=status.ullAvailPhys / _MB,
        headroom_mb=status.ullAvailPageFile / _MB,
        committed_mb=(status.ullTotalPageFile - status.ullAvailPageFile) / _MB,
        limit_mb=status.ullTotalPageFile / _MB,
    )


def _linux() -> Memory:
    fields: dict[str, float] = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            fields[key] = float(rest.strip().split()[0]) / 1024.0  # kB -> MB
    total = fields.get("MemTotal", 0.0)
    available = fields.get("MemAvailable", fields.get("MemFree", 0.0))
    limit = fields.get("CommitLimit", 0.0)
    committed = fields.get("Committed_AS", 0.0)
    # Overcommit is usually on, in which case the commit limit is advisory and
    # free memory is the honest constraint.
    headroom = max(limit - committed, 0.0) if limit else available
    return Memory(
        total_mb=total,
        available_mb=available,
        headroom_mb=min(headroom, available) if limit else available,
        committed_mb=committed,
        limit_mb=limit or total,
    )


def headroom_mb() -> float:
    """Grantable memory in MB, or `inf` when we cannot tell (never block blind)."""
    memory = snapshot()
    return memory.headroom_mb if memory else float("inf")
