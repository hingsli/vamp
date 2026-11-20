from pathlib import Path

import pytest
from datasets import Dataset

from query_adaptive_quantization.threshold_maps.labels import (
    index_records_by_sample,
    write_label_parquets,
)
from query_adaptive_quantization.threshold_maps.schema import (
    metric_value,
    normalize_record,
    reference_bit_from_manifest,
)
from query_adaptive_quantization.training.threshold_map.data import (
    load_precision_map_dataset,
)


def _record(sample_index: int, bit: int = 6) -> dict:
    return {
        "sample_index": sample_index,
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "precision_map": [[bit, bit], [bit, bit]],
        "threshold": 0.001,
    }


def test_parquet_round_trip_and_schema_validation(tmp_path: Path) -> None:
    write_label_parquets(
        tmp_path,
        [_record(0), _record(1, 7), _record(2, 8)],
        validation_samples=1,
        formal_split=True,
    )
    train, validation = load_precision_map_dataset(tmp_path)
    assert len(train) == 2
    assert len(validation) == 1
    assert set(train.column_names) == {
        "attention_mask",
        "input_ids",
        "precision_map",
    }


def test_schema_rejects_non_blockwise_map(tmp_path: Path) -> None:
    invalid = _record(0)
    invalid["precision_map"] = [[6]]
    Dataset.from_list([invalid]).to_parquet(str(tmp_path / "train.parquet"))
    Dataset.from_list([_record(1)]).to_parquet(
        str(tmp_path / "validation.parquet")
    )
    with pytest.raises(ValueError, match=r"shape \[L,2\]"):
        load_precision_map_dataset(tmp_path)


def test_resume_index_rejects_duplicate_prompts() -> None:
    with pytest.raises(ValueError, match="Duplicate sample_index"):
        index_records_by_sample([_record(3), _record(3)])


def test_legacy_int8_schema_is_normalized_in_memory() -> None:
    normalized = normalize_record(
        {
            "kl_to_int8": 0.001,
            "int8_nll_sum": 12.5,
            "int8_token_count": 3,
            "ppl_ratio_to_int8": 1.001,
        }
    )
    assert normalized == {
        "kl_to_reference": 0.001,
        "reference_nll_sum": 12.5,
        "reference_token_count": 3,
        "ppl_ratio_to_reference": 1.001,
        "reference_bit": 8,
    }
    assert reference_bit_from_manifest({"reference_precision": 8}) == 8
    assert metric_value(
        {"mean_kl_to_int8": 0.002}, "mean_kl_to_reference"
    ) == 0.002
