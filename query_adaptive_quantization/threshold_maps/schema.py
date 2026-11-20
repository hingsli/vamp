"""Schema helpers for threshold-supervised precision-map artifacts."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = 2

LEGACY_FIELD_ALIASES = {
    "kl_to_int8": "kl_to_reference",
    "int8_nll_sum": "reference_nll_sum",
    "int8_token_count": "reference_token_count",
    "calibration_ppl_ratio_to_int8": "calibration_ppl_ratio_to_reference",
    "mean_kl_to_int8": "mean_kl_to_reference",
    "p95_kl_to_int8": "p95_kl_to_reference",
    "ppl_ratio_to_int8": "ppl_ratio_to_reference",
    "maximum_ppl_ratio_to_int8": "maximum_ppl_ratio_to_reference",
}


def normalize_legacy_fields(value: Any) -> Any:
    """Return an in-memory generic-schema view without rewriting source files."""
    if isinstance(value, list):
        return [normalize_legacy_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: normalize_legacy_fields(item) for key, item in value.items()
    }
    for legacy_name, generic_name in LEGACY_FIELD_ALIASES.items():
        if legacy_name in normalized and generic_name not in normalized:
            normalized[generic_name] = normalized[legacy_name]
        normalized.pop(legacy_name, None)
    return normalized


def normalize_record(
    record: Mapping[str, Any],
    *,
    default_reference_bit: int = 8,
) -> dict[str, Any]:
    normalized = normalize_legacy_fields(dict(record))
    normalized.setdefault("reference_bit", int(default_reference_bit))
    return normalized


def reference_bit_from_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    default: int = 8,
) -> int:
    if not manifest:
        return int(default)
    reference = manifest.get("reference")
    if isinstance(reference, Mapping) and "bit" in reference:
        return int(reference["bit"])
    quality = manifest.get("quality")
    if isinstance(quality, Mapping) and "reference_bit" in quality:
        return int(quality["reference_bit"])
    if "reference_bit" in manifest:
        return int(manifest["reference_bit"])
    if "reference_precision" in manifest:
        return int(manifest["reference_precision"])
    return int(default)


def metric_value(
    metrics: Mapping[str, Any],
    generic_name: str,
    *,
    legacy_name: str | None = None,
) -> Any:
    if generic_name in metrics:
        return metrics[generic_name]
    if legacy_name and legacy_name in metrics:
        return metrics[legacy_name]
    for candidate_legacy_name, candidate_generic_name in LEGACY_FIELD_ALIASES.items():
        if (
            candidate_generic_name == generic_name
            and candidate_legacy_name in metrics
        ):
            return metrics[candidate_legacy_name]
    raise KeyError(generic_name)
