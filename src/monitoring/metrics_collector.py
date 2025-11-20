"""Runtime metrics collection utilities.

Provides lightweight collection for CPU/RAM and simple latency aggregation.
Designed for embedding in inference endpoints; can be extended for Prometheus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


@dataclass
class LatencyStats:
    samples: List[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        self.samples.append(value)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0}
        s = sorted(self.samples)

        def pct(p: float) -> float:
            idx = min(len(s) - 1, int(p * (len(s) - 1)))
            return s[idx]
        return {
            "count": len(s),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "max": s[-1],
            "min": s[0],
            "avg": sum(s) / len(s),
        }


def system_metrics() -> Dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        return {"psutil_available": 0}
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.0)
    return {
        "cpu_percent": cpu,
        "ram_used_mb": vm.used / (1024 * 1024),
        "ram_available_mb": vm.available / (1024 * 1024),
    }


def to_json(data: Dict) -> str:
    return json.dumps(data, indent=2)


__all__ = ["LatencyStats", "system_metrics", "to_json"]
