"""Train the precision Router using maps from vamp-search."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-model", required=True)
    parser.add_argument("--labels", required=True, help="Directory containing train/validation.parquet.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--online-ttt-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.grad_accum < 1 or args.learning_rate <= 0:
        raise ValueError("grad-accum and learning-rate must be positive")
    import torch
    from transformers import set_seed
    from query_adaptive_quantization.training.threshold_map import (
        ThresholdMapTrainingPipeline, build_training_config,
    )
    from query_adaptive_quantization.training.threshold_map.config import save_training_config
    from query_adaptive_quantization.training.threshold_map.data import load_precision_map_dataset

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires one visible CUDA GPU; set CUDA_VISIBLE_DEVICES to its index.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    train, _ = load_precision_map_dataset(args.labels)
    length = len(train[0]["input_ids"])
    config = build_training_config(
        student_model=args.packed_model, label_dataset=args.labels,
        output_dir=str(output), epochs=args.epochs, max_length=length,
        train_batch_size=1, grad_accum=args.grad_accum,
        learning_rate=args.learning_rate, online_ttt_lr=args.online_ttt_lr,
        gradient_checkpointing=False, report_to="none", seed=args.seed,
    )
    set_seed(args.seed)
    path = output / "router_training_config.yaml"
    save_training_config(path, config)
    ThresholdMapTrainingPipeline(path).run()


if __name__ == "__main__":
    main()
