"""Precision-map Parquet loading and schema validation."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset


REQUIRED_COLUMNS = {"input_ids", "attention_mask", "precision_map"}
SUPPORTED_BITS = {3, 4, 5, 6, 7, 8}


def _validate_split(dataset: Dataset, split_name: str) -> Dataset:
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Precision-map {split_name} split is missing columns: {sorted(missing)}"
        )
    if len(dataset) == 0:
        raise ValueError(f"Precision-map {split_name} split is empty")
    for row_index, example in enumerate(dataset):
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        precision_map = example["precision_map"]
        if len(input_ids) < 2 or len(input_ids) != len(attention_mask):
            raise ValueError(
                f"Invalid token fields in {split_name} row {row_index}: "
                f"input_ids={len(input_ids)}, attention_mask={len(attention_mask)}"
            )
        if not precision_map or any(len(layer) != 2 for layer in precision_map):
            raise ValueError(
                f"Precision map in {split_name} row {row_index} must have shape [L,2]"
            )
        bits = {int(bit) for layer in precision_map for bit in layer}
        if not bits.issubset(SUPPORTED_BITS):
            raise ValueError(
                f"Unsupported bits in {split_name} row {row_index}: "
                f"{sorted(bits.difference(SUPPORTED_BITS))}"
            )
    return dataset.select_columns(sorted(REQUIRED_COLUMNS))


def load_precision_map_dataset(path: str | Path) -> tuple[Dataset, Dataset]:
    dataset_dir = Path(path).expanduser().resolve()
    train_path = dataset_dir / "train.parquet"
    validation_path = dataset_dir / "validation.parquet"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Precision-map dataset must contain train.parquet and "
            f"validation.parquet: {dataset_dir}"
        )
    data_files = {
        "train": str(train_path),
        "validation": str(validation_path),
    }
    train_dataset = load_dataset("parquet", data_files=data_files, split="train")
    validation_dataset = load_dataset(
        "parquet", data_files=data_files, split="validation"
    )
    return (
        _validate_split(train_dataset, "train"),
        _validate_split(validation_dataset, "validation"),
    )
