from datasets import load_dataset
import random
import numpy as np
import logging
import torch


def _get_wikitext2(split):
    assert split in [
        "train",
        "validation",
        "test",
    ], f"Unknown split {split} for wikitext2"
    data = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True
    )
    return data["text"]


def _get_ptb(split, slice_unk=True):
    assert split in ["train", "validation", "test"], f"Unknown split {split} for ptb"
    data = load_dataset(
        "ptb_text_only", "penn_treebank", split=split, trust_remote_code=True
    )
    data_list = data["sentence"]
    if slice_unk:
        data_list = [s.replace("<unk>", "< u n k >") for s in data_list]
    return data_list


def _get_c4(split):
    assert split in ["train", "validation"], f"Unknown split {split} for c4"
    data = load_dataset("c4", "en", split=split, trust_remote_code=True)
    return data["text"]


def _get_pileval(split):
    if split != "validation":
        logging.warning(
            f"Pileval only has a validation split, but got split={split}. Using validation split."
        )
    data = load_dataset(
        "mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True
    )
    return data["text"]


def _get_alpaca(split):
    if split not in ["train", "validation", "test"]:
        raise ValueError(
            f"Unknown split {split} for alpaca. Available splits: train, validation, test"
        )
    data = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
    if split == "train":
        train_size = int(0.8 * len(data))
        data = data.select(range(train_size))
    elif split == "validation":
        train_size = int(0.8 * len(data))
        val_size = int(0.9 * len(data))
        data = data.select(range(train_size, val_size))
    elif split == "test":
        val_size = int(0.9 * len(data))
        data = data.select(range(val_size, len(data)))
    return data["text"]


def _sample_and_tokenize(
    texts,
    tokenizer,
    seq_len,
    num_samples,
    seed=None,
    force_concatenation=False,
):
    assert num_samples <= len(
        texts
    ), f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"
    logging.info(f"Total texts available: {len(texts)}")
    logging.info(
        f"Starting sampling process for {num_samples} samples with sequence length {seq_len}..."
    )
    random.seed(seed)
    np.random.seed(seed)
    selected_indices = set()
    samples = []
    attempts = 0
    max_attempts = min(len(texts) * 10, 50000)
    consecutive_failures = 0
    max_consecutive_failures = 1000
    logging.info("Checking dataset compatibility...")
    long_enough_count = 0
    sample_check_size = min(1000, len(texts))
    for i in range(sample_check_size):
        text = texts[i]
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(tokens) >= seq_len:
            long_enough_count += 1
    estimated_total_long_enough = long_enough_count / sample_check_size * len(texts)
    logging.info(
        f"Estimated {estimated_total_long_enough:.0f} texts ({estimated_total_long_enough / len(texts) * 100:.1f}%) satisfy the sequence length requirement of {seq_len} tokens"
    )
    if force_concatenation or estimated_total_long_enough < num_samples:
        logging.warning(
            f"Dataset may not have enough long texts. Estimated {estimated_total_long_enough:.0f} available vs {num_samples} requested. Will try to concatenate short texts."
        )
        use_concatenation = True
    else:
        use_concatenation = False
    while len(samples) < num_samples and attempts < max_attempts:
        idx = random.randint(0, len(texts) - 1)
        attempts += 1
        if idx in selected_indices:
            continue
        text = texts[idx]
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(tokens) >= seq_len:
            tokens = tokens[:seq_len]
            selected_indices.add(idx)
            samples.append(tokens)
            consecutive_failures = 0
        elif use_concatenation and len(tokens) > 10:
            combined_tokens = tokens
            used_indices = [idx]
            while len(combined_tokens) < seq_len and len(used_indices) < 10:
                next_idx = random.randint(0, len(texts) - 1)
                if next_idx not in selected_indices and next_idx not in used_indices:
                    next_text = texts[next_idx]
                    next_tokens = tokenizer(next_text, return_tensors="pt")[
                        "input_ids"
                    ][0]
                    if len(next_tokens) > 5:
                        separator_token = tokenizer(" ", return_tensors="pt")[
                            "input_ids"
                        ][0][-1:]
                        combined_tokens = torch.cat(
                            [combined_tokens, separator_token, next_tokens]
                        )
                        used_indices.append(next_idx)
            if len(combined_tokens) >= seq_len:
                combined_tokens = combined_tokens[:seq_len]
                for used_idx in used_indices:
                    selected_indices.add(used_idx)
                samples.append(combined_tokens)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        else:
            consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            logging.error(
                f"continuous{max_consecutive_failures}Unable to find suitable samples, the data set may not be suitable for the current sequence length requirements"
            )
            logging.error(
                f"Recommendations: 1) Reduce sequence length (currently{seq_len}) 2) Use other datasets like c4 or wikitext2"
            )
            raise ValueError(
                f"Unable to sample enough from the data set{seq_len}-token sample. Please consider lowering seq_len or using a different dataset."
            )
        if attempts % 1000 == 0 or len(samples) % 10 == 0:
            logging.info(
                f"Sampling progress: {len(samples)}/{num_samples} Sample obtained (tried {attempts} times, consecutive failures {consecutive_failures} Second-rate)"
            )
    if len(samples) < num_samples:
        logging.warning(
            f"Can only sample {len(samples)} samples, less than requested {num_samples} indivual"
        )
        if len(samples) == 0:
            raise ValueError(
                "Unable to sample any suitable samples, please check the data set and parameter settings"
            )
    logging.info(
        f"Sampling is complete! get {len(samples)} samples, tried in total {attempts} Second-rate"
    )
    return samples


def _get_dataset(dataset_name, split):
    if dataset_name == "wikitext2":
        return _get_wikitext2(split)
    elif dataset_name == "ptb":
        return _get_ptb(split)
    elif dataset_name == "c4":
        return _get_c4(split)
    elif dataset_name == "pileval":
        return _get_pileval(split)
    elif dataset_name == "alpaca":
        return _get_alpaca(split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(
    dataset_name,
    split,
    tokenizer,
    seq_len,
    num_samples,
    seed=None,
    force_concatenation=False,
):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(dataset_name, split)
    logging.info(
        f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}..."
    )
    return _sample_and_tokenize(
        texts,
        tokenizer,
        seq_len,
        num_samples,
        seed,
        force_concatenation=force_concatenation,
    )
