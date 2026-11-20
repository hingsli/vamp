"""Build verified layer/branch precision labels from a local text dataset."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from .schema import SCHEMA_VERSION
from .search import ThresholdPrecisionMapSearcher


def index_records_by_sample(
    records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_index = int(record["sample_index"])
        if sample_index in indexed:
            raise ValueError(f"Duplicate sample_index in labels: {sample_index}")
        indexed[sample_index] = record
    return indexed


def pack_examples(
    dataset,
    tokenizer,
    *,
    num_samples: int,
    skip_samples: int,
    max_length: int,
) -> list[dict[str, Any]]:
    eos_token_id = tokenizer.eos_token_id
    packed = []
    buffer: list[int] = []
    packed_index = 0
    for source_index, example in enumerate(dataset):
        text = example.get("text") or example.get("sentence")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer(
            text, add_special_tokens=False, truncation=False
        )["input_ids"]
        if not token_ids:
            continue
        buffer.extend(int(token_id) for token_id in token_ids)
        if eos_token_id is not None:
            buffer.append(int(eos_token_id))
        while len(buffer) >= max_length:
            chunk = buffer[:max_length]
            del buffer[:max_length]
            if packed_index >= skip_samples:
                packed.append(
                    {
                        "sample_index": packed_index,
                        "source_row_index": source_index,
                        "input_ids": chunk,
                        "attention_mask": [1] * max_length,
                    }
                )
                if len(packed) >= num_samples:
                    return packed
            packed_index += 1
    raise ValueError(
        f"Dataset produced {len(packed)} requested samples after skipping "
        f"{skip_samples}; expected {num_samples}."
    )


def write_label_parquets(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    validation_samples: int,
    formal_split: bool,
) -> None:
    index_records_by_sample(records)
    ordered = sorted(records, key=lambda record: int(record["sample_index"]))
    if not ordered or (formal_split and len(ordered) < 2):
        raise ValueError("A training/validation split requires at least two labels")
    Dataset.from_list(ordered).to_parquet(str(output_dir / "labels.parquet"))
    if formal_split:
        validation_count = min(max(1, validation_samples), len(ordered) - 1)
        train_records = ordered[:-validation_count]
        validation_records = ordered[-validation_count:]
        Dataset.from_list(train_records).to_parquet(str(output_dir / "train.parquet"))
        Dataset.from_list(validation_records).to_parquet(
            str(output_dir / "validation.parquet")
        )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-model", required=True)
    parser.add_argument("--data", required=True, help="Local JSONL/JSON/Parquet with a text column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--num-groups", type=int, default=6)
    parser.add_argument("--max-dp-candidates", type=int, default=8)
    parser.add_argument("--refinement-passes", type=int, default=2)
    parser.add_argument("--max-swap-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.num_samples < 2 or args.max_length < 2:
        raise ValueError("num-samples and max-length must be at least two")
    if not 1 <= args.validation_samples < args.num_samples:
        raise ValueError("validation-samples must leave at least one training sample")
    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Packed-model search requires CUDA and the compiled extension")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select one CUDA GPU with CUDA_VISIBLE_DEVICES")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    data = Path(args.data)
    loader = {".json": "json", ".jsonl": "json", ".parquet": "parquet"}.get(data.suffix.lower())
    if loader is None:
        raise ValueError("data must be JSON, JSONL or Parquet")
    dataset = load_dataset(loader, data_files=str(data), split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.packed_model)
    examples = pack_examples(
        dataset, tokenizer, num_samples=args.num_samples, skip_samples=0,
        max_length=args.max_length,
    )
    from query_adaptive_quantization.modules import QueryAdaptiveQuantizationForCausalLM

    torch.manual_seed(args.seed)
    model = QueryAdaptiveQuantizationForCausalLM.from_quantized(
        args.packed_model, trust_remote_code=False,
    ).to("cuda:0").eval()
    if not set(range(3, 9)).issubset(model.supported_bits):
        raise ValueError("The packed model must support all Int3–Int8 precisions")
    searcher = ThresholdPrecisionMapSearcher(
        model, reference_bit=8, precision_levels=range(3, 9),
        num_groups=args.num_groups, max_dp_candidates=args.max_dp_candidates,
        refinement_passes=args.refinement_passes,
        max_swap_candidates=args.max_swap_candidates,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "VAMP",
        "status": "running",
        "reference": {"type": "uniform_packed_precision", "bit": 8},
        "selected_threshold": args.threshold,
        "candidate_bits": list(range(3, 9)),
        "quality_metric": {"name": "mean_next_token_kl", "direction": "reference_to_candidate"},
        "data_sha256": sha256(data),
        "packed_config_sha256": sha256(Path(args.packed_model) / "config.json"),
        "arguments": vars(args),
        "cost_model": searcher.cost_model.manifest(),
        "map_shape": [searcher.num_layers, 2],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (output / "labels.jsonl").open("w") as handle:
        for example in examples:
            tokens = torch.tensor([example["input_ids"]], device="cuda:0")
            mask = torch.ones_like(tokens)
            result = searcher.search_selected_threshold(tokens, mask, args.threshold)
            kl = float(result["kl_to_reference"])
            if not math.isfinite(kl) or kl > args.threshold:
                raise RuntimeError(f"Search returned an infeasible label: KL={kl}")
            record = {**example, "reference_bit": 8, **result}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
            print(f"[{len(records)}/{len(examples)}] KL={kl:.6g} weighted bits={result['weighted_effective_bit']:.2f}", flush=True)
    write_label_parquets(output, records, validation_samples=args.validation_samples, formal_split=True)
    manifest["status"] = "complete"
    manifest["num_labels"] = len(records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()


