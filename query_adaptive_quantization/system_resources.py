"""Container-aware memory reporting for model preparation."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import psutil

GIB = 1024**3
DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def _read_int(path: Path, default: int = 0) -> int:
    value = _read_text(path)
    if not value or value == "max":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in _read_text(path).splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _pressure_full_avg10(path: Path) -> float:
    for line in _read_text(path).splitlines():
        if not line.startswith("full "):
            continue
        for item in line.split()[1:]:
            key, _, value = item.partition("=")
            if key == "avg10":
                try:
                    return float(value)
                except ValueError:
                    return 0.0
    return 0.0


@dataclass(frozen=True)
class CgroupMemorySnapshot:
    limit_bytes: int | None
    current_bytes: int
    anon_bytes: int
    shmem_bytes: int
    kernel_bytes: int
    file_bytes: int
    inactive_file_bytes: int
    pressure_full_avg10: float
    events: dict[str, int]

    @property
    def non_reclaimable_bytes(self) -> int:
        return self.anon_bytes + self.shmem_bytes + self.kernel_bytes

    def to_gib_dict(self) -> dict[str, float | int | None | dict[str, int]]:
        return {
            "limit_gib": (
                self.limit_bytes / GIB if self.limit_bytes is not None else None
            ),
            "current_gib": self.current_bytes / GIB,
            "non_reclaimable_gib": self.non_reclaimable_bytes / GIB,
            "anon_gib": self.anon_bytes / GIB,
            "file_gib": self.file_bytes / GIB,
            "inactive_file_gib": self.inactive_file_bytes / GIB,
            "pressure_full_avg10": self.pressure_full_avg10,
            "events": dict(self.events),
        }


def read_cgroup_memory(
    root: Path = DEFAULT_CGROUP_ROOT,
) -> CgroupMemorySnapshot | None:
    """Read a cgroup-v2 memory snapshot, if one is available."""

    max_path = root / "memory.max"
    current_path = root / "memory.current"
    if not max_path.is_file() or not current_path.is_file():
        return None
    raw_limit = _read_text(max_path)
    limit = None if raw_limit == "max" else _read_int(max_path)
    stats = _read_key_values(root / "memory.stat")
    return CgroupMemorySnapshot(
        limit_bytes=limit,
        current_bytes=_read_int(current_path),
        anon_bytes=stats.get("anon", 0),
        shmem_bytes=stats.get("shmem", 0),
        kernel_bytes=stats.get("kernel", 0),
        file_bytes=stats.get("file", 0),
        inactive_file_bytes=stats.get("inactive_file", 0),
        pressure_full_avg10=_pressure_full_avg10(root / "memory.pressure"),
        events=_read_key_values(root / "memory.events"),
    )


def system_memory_info(
    root: Path = DEFAULT_CGROUP_ROOT,
) -> dict[str, float]:
    """Return memory information using the container limit when present."""

    snapshot = read_cgroup_memory(root)
    if snapshot is not None and snapshot.limit_bytes is not None:
        total = float(snapshot.limit_bytes)
        used = float(min(snapshot.current_bytes, snapshot.limit_bytes))
        available = max(total - used, 0.0)
        return {
            "used": used / GIB,
            "available": available / GIB,
            "percent": (used / total * 100.0) if total else 0.0,
            "total": total / GIB,
            "non_reclaimable": snapshot.non_reclaimable_bytes / GIB,
            "file_cache": snapshot.file_bytes / GIB,
        }
    memory = psutil.virtual_memory()
    return {
        "used": memory.used / GIB,
        "available": memory.available / GIB,
        "percent": float(memory.percent),
        "total": memory.total / GIB,
        "non_reclaimable": memory.used / GIB,
        "file_cache": float(getattr(memory, "cached", 0)) / GIB,
    }

