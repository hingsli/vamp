"""CPU-only validation of measured runtime costs; no model imports."""
import math

VERSION = "qaq-hbf-v1"


def validate_profiles(payload):
    if payload.get("schema") != VERSION or payload.get("status") != "complete":
        raise ValueError("Missing complete measured HBF profile")
    if payload.get("synthetic", True):
        raise ValueError("Synthetic costs cannot enter measured HBF")
    for phase in ("1", "64", "1024"):
        for bit in range(3, 9):
            row = payload["profiles"][phase][str(bit)]
            for key in ("comp_ms", "lut_io_ms", "qweight_io_ms"):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"Invalid HBF cost {phase}/{bit}/{key}")
    return payload
