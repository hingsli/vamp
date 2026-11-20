import os
import os.path
import shutil
import logging
from .config import *
from ..analyzer import get_analyzer
from .gradients import get_gradients

try:
    from .gradients_optimized import (
        get_gradients_optimized,
        get_gradients_with_memory_limit,
    )

    OPTIMIZED_GRADIENTS_AVAILABLE = True
    logging.info(
        " The optimized version of the gradient calculation module has been successfully imported"
    )
except ImportError as e:
    OPTIMIZED_GRADIENTS_AVAILABLE = False
    logging.warning(
        f" The optimized version of the gradient calculation module is not available: {e}"
    )
    logging.warning(" Standard version will be used (enhanced memory monitoring)")
from .quantize import seed_and_upscale
from .pack import pack
from .gradient_store import (
    gradient_shard_dir,
    release_model_source_cache,
)
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def query_adaptive_quantize(
    model,
    seed_precision=DEFAULT_SEED_PRECISION,
    parent_precision=DEFAULT_PARENT_PRECISION,
    mode="pack",
    yaml_path=None,
    cache_dir=DEFAULT_CACHE_DIR,
    dataset=DEFAULT_DATASET,
    seq_len=DEFAULT_SEQ_LEN,
    num_examples=DEFAULT_NUM_EXAMPLES,
    cpu_count=os.cpu_count(),
    overwrite_gradients=False,
    overwrite_quantize=False,
    overwrite_pack=False,
    random_state=None,
    group_count=1,
    use_optimized_gradients=None,
    gradient_accumulation_steps=4,
    memory_check_interval=10,
    max_memory_gb=None,
    max_parent_relative_rmse=0.05,
    gradient_checkpointing=False,
):
    assert mode in [
        "gradients",
        "quantize",
        "pack",
    ], "mode must be one of 'gradients', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."
    if overwrite_gradients:
        if not overwrite_quantize:
            logging.warning(
                "Parent model needs to be recalculated if gradients are recalculated. Setting overwrite_quantize to True."
            )
            overwrite_quantize = True
    if overwrite_quantize:
        if not overwrite_pack:
            logging.warning(
                "Packed model needs to be recalculated if parent model is recalculated. Setting overwrite_pack to True."
            )
            overwrite_pack = True
    if mode == "gradients":
        logging.info("Running: [Gradients]")
    elif mode == "quantize":
        logging.info("Running: [Gradients -> Quantize]")
    else:
        logging.info("Running: [Gradients -> Quantize -> Pack]")
    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]
    logging.info(
        f"Running Query-Adaptive Quantization on {model_name} with seed precision {seed_precision} and parent precision {parent_precision} using {dataset} for gradient calculation"
    )
    logging.info(f"Select a gradient collector:")
    logging.info(f"   - User specified: {use_optimized_gradients}")
    logging.info(f"   - Optimized version available: {OPTIMIZED_GRADIENTS_AVAILABLE}")
    logging.info(f"   - Sample size: {num_examples}")
    if use_optimized_gradients is None:
        if OPTIMIZED_GRADIENTS_AVAILABLE:
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_memory_gb = device_props.total_memory / 1024**3
                use_optimized_gradients = num_examples > 50 or total_memory_gb < 50
                if use_optimized_gradients:
                    logging.info(
                        f" Automatically force the use of optimized version of gradient calculation (number of samples: {num_examples}, GPU memory: {total_memory_gb:.1f}GB)"
                    )
                    logging.info(
                        f" Reason: Memory risk detected - number of samples>{50} or GPU memory <{50}GB"
                    )
                else:
                    logging.info(
                        f" Automatically select the standard version of gradient calculation (number of samples: {num_examples}, GPU memory: {total_memory_gb:.1f}GB)"
                    )
            else:
                use_optimized_gradients = False
                logging.info(
                    "CPU mode, using standard version of gradient calculation"
                )
        else:
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_memory_gb = device_props.total_memory / 1024**3
                if num_examples > 50 or total_memory_gb < 50:
                    logging.error(
                        f" High OOM risk detected but optimized version is not available!"
                    )
                    logging.error(
                        f"Recommendation: Reduce the number of examples to 25 or less: --num_examples 25"
                    )
            use_optimized_gradients = False
            logging.warning(
                " The optimized version is not available, use the standard version of gradient calculation"
            )
    elif use_optimized_gradients and (not OPTIMIZED_GRADIENTS_AVAILABLE):
        logging.warning(
            " Request to use the optimized version but it is not available, fall back to the standard version"
        )
        use_optimized_gradients = False
    if max_memory_gb is None and torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        max_memory_gb = device_props.total_memory / 1024**3 * 0.9
        logging.info(f" Automatically set the memory limit to: {max_memory_gb:.1f}GB")
    analyzer = get_analyzer(
        model,
        yaml_path=yaml_path,
        include_tokenizer=True,
        dtype=torch.bfloat16 if gradient_checkpointing else torch.float16,
    )
    if isinstance(model, str):
        release_model_source_cache(model)
    gradients_cache_path = f"{cache_dir}/gradients/({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt"
    quantized_cache_path = f"{cache_dir}/quantized/({model_name})-w{parent_precision}_orig{seed_precision}-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}"
    model_output_path = f"{cache_dir}/packed/qaqprec-({model_name})-w{parent_precision}_orig{seed_precision}-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}"
    logging.info("------------------- Gradients -------------------")
    logging.info("Beginning gradient calculation...")
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        logging.info(
            f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate."
        )
        os.remove(gradients_cache_path)
    gradients_shard_path = gradient_shard_dir(gradients_cache_path)
    if overwrite_gradients and gradients_shard_path.exists():
        logging.info(
            "Detected cached layer-sharded gradients at "
            f"{gradients_shard_path}. Will delete and recalculate."
        )
        shutil.rmtree(gradients_shard_path)
    if use_optimized_gradients:
        logging.info(" Use optimized version of gradient calculation...")
        if max_memory_gb and max_memory_gb < 50:
            logging.info(
                f" Memory constrained environment ({max_memory_gb:.1f}GB), use the memory-limited version"
            )
            model_gradients = get_gradients_with_memory_limit(
                analyzer=analyzer,
                dataset=dataset,
                seq_len=seq_len,
                num_examples=num_examples,
                save_path=gradients_cache_path,
                random_state=random_state,
                max_memory_gb=max_memory_gb,
                gradient_accumulation_steps=gradient_accumulation_steps,
                memory_check_interval=memory_check_interval,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            model_gradients = get_gradients_optimized(
                analyzer=analyzer,
                dataset=dataset,
                seq_len=seq_len,
                num_examples=num_examples,
                save_path=gradients_cache_path,
                random_state=random_state,
                gradient_accumulation_steps=gradient_accumulation_steps,
                memory_check_interval=memory_check_interval,
            )
    else:
        logging.info(" Use standard version of gradient calculation...")
        model_gradients = get_gradients(
            analyzer=analyzer,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            save_path=gradients_cache_path,
            random_state=random_state,
        )
    logging.info(" Gradient calculation completed")
    if mode == "gradients":
        return
    logging.info("------------------- Quantize: Seed + Upscale -------------------")
    logging.info(
        f"Beginning {seed_precision}~{parent_precision}-bit Query-Adaptive Quantization..."
    )
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        logging.info(
            f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate."
        )
        shutil.rmtree(quantized_cache_path)
    seed_and_upscale(
        analyzer=analyzer,
        gradients=model_gradients,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        random_state=random_state,
        group_count=group_count,
        max_parent_relative_rmse=max_parent_relative_rmse,
    )
    if mode == "quantize":
        return
    del model_gradients
    analyzer.drop_original_weights()
    logging.info("Quantization(Seed + Upscale) complete.")
    logging.info("------------------- Pack -------------------")
    should_pack = True
    if (
        os.path.exists(model_output_path)
        and os.path.isdir(model_output_path)
        and os.listdir(model_output_path)
    ):
        if overwrite_pack:
            logging.info(
                f"Model output path {model_output_path} already exists and is not empty. Will delete and re-pack."
            )
            shutil.rmtree(model_output_path)
        else:
            logging.info(
                f"Model output path {model_output_path} already exists and is not empty. Will skip packing."
            )
            should_pack = False
    if should_pack:
        pack(
            analyzer=analyzer,
            lut_path=quantized_cache_path,
            output_model_path=model_output_path,
            seed_precision=seed_precision,
            parent_precision=parent_precision,
            cpu_count=cpu_count,
            group_count=group_count,
        )
        logging.info("Packing complete.")
