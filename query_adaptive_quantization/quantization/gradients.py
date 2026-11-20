import torch
from tqdm import tqdm
import os
import logging
import gc
from .config import *
from query_adaptive_quantization.analyzer.analyzer import get_analyzer
from .datautils import get_tokens
from .salience import scaled_squared_gradient

try:
    from .memory_utils import MemoryTracker, cleanup_gpu_memory, monitor_gradient_memory

    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    logging.warning(
        " Memory tools module is not available, basic memory monitoring will be used"
    )


def get_gradients(
    analyzer,
    dataset=DEFAULT_DATASET,
    seq_len=DEFAULT_SEQ_LEN,
    num_examples=DEFAULT_NUM_EXAMPLES,
    save_path=None,
    random_state=None,
):
    if MEMORY_UTILS_AVAILABLE:
        memory_tracker = MemoryTracker(logger=logging.getLogger(__name__))
        memory_tracker.log_memory_status("Gradient calculation starts", "INFO")
    elif torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        logging.info(f" Initial GPU memory usage: {initial_memory:.2f}GB")
    if save_path is not None and os.path.isfile(save_path):
        logging.info(f" Gradient is calculated and saved in {save_path}")
        logging.info(f" Loading gradients...")
        return torch.load(save_path, weights_only=False)
    logging.info(f" Start gradient calculation:")
    logging.info(f"   - Dataset: {dataset}")
    logging.info(f"   - Sequence length: {seq_len}")
    logging.info(f"   - Sample size: {num_examples}")
    save_target = save_path if save_path else "Do not save"
    logging.info(f"   - Save path: {save_target}")
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / 1024**3
        logging.info(
            f"GPU information: {device_props.name}, total memory: {total_memory:.1f}GB"
        )
        if num_examples > 50 and total_memory < 50:
            logging.warning(
                f" A large number of samples were detected ({num_examples}) and limited GPU memory ({total_memory:.1f}GB)"
            )
            logging.warning(
                f"   It is recommended to use the get_gradients_optimized() function for better memory management"
            )
    logging.info(
        f" Calculating gradient, using dataset {dataset}, sequence length {seq_len}, number of samples {num_examples}..."
    )
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    input_tokens = get_tokens(
        dataset, "train", tokenizer, seq_len, num_examples, seed=random_state
    )
    logging.info(f"Successfully loaded {len(input_tokens)} input tokens")
    if analyzer is None:
        analyzer = get_analyzer(model)
    logging.info("Converting model to bfloat16 and moving to GPU...")
    model = model.bfloat16()
    model.eval()
    if model.device.type != "cuda":
        model.cuda()
    logging.info(f"Model is on device: {model.device}")
    layers = analyzer.get_layers()
    logging.info(f"Found {len(layers)} layers to analyze")

    def square_grad_hook(grad):
        return scaled_squared_gradient(grad)

    hooks = []
    logging.info("Registering gradient hooks...")
    for layer in layers:
        for module in analyzer.get_modules(layer).values():
            hooks.append(module.weight.register_hook(square_grad_hook))
    logging.info(f"Registered {len(hooks)} gradient hooks")
    logging.info(
        f" Start gradient calculation, total number of samples: {len(input_tokens)}"
    )
    if MEMORY_UTILS_AVAILABLE:
        memory_tracker.checkpoint("Gradient calculation loop starts")
    total_samples_processed = 0
    oom_count = 0
    try:
        for i, tokens in enumerate(tqdm(input_tokens, desc="Calculate gradient")):
            try:
                tokens = tokens.to(model.device)
                tokens = tokens.unsqueeze(0)
                outputs = model(input_ids=tokens, labels=tokens)
                loss = outputs.loss
                loss.backward()
                total_samples_processed += 1
                if (i + 1) % 10 == 0:
                    if MEMORY_UTILS_AVAILABLE:
                        memory_tracker.log_memory_status(f"sample {i + 1}", "DEBUG")
                        monitor_gradient_memory(
                            model, f"step_{i + 1}", logging.getLogger(__name__)
                        )
                    elif torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        memory_usage_pct = (
                            current_memory
                            / (
                                torch.cuda.get_device_properties(0).total_memory
                                / 1024**3
                            )
                            * 100
                        )
                        if memory_usage_pct > 90:
                            logging.warning(
                                f"GPU memory usage is too high: {memory_usage_pct:.1f}% ({current_memory:.2f}GB)"
                            )
                if (i + 1) % 20 == 0 or i + 1 == len(input_tokens):
                    current_memory_str = ""
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        current_memory_str = f", GPU memory: {current_memory:.2f}GB"
                    logging.info(
                        f" Processed {i + 1}/{len(input_tokens)} Sample, current loss: {loss.item():.4f}{current_memory_str}"
                    )
                    if (i + 1) % 50 == 0 and MEMORY_UTILS_AVAILABLE:
                        memory_tracker.log_memory_status(
                            f"progress checkpoint {i + 1}", "INFO"
                        )
            except torch.cuda.OutOfMemoryError as e:
                oom_count += 1
                logging.error(
                    f" Sample {i + 1} Out of GPU memory occurred while processing (section{oom_count}Second-rate): {e}"
                )
                if MEMORY_UTILS_AVAILABLE:
                    cleanup_gpu_memory()
                    memory_tracker.log_memory_status("OOM recovery attempt", "ERROR")
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                if oom_count >= 3:
                    logging.error(
                        f" Occurs continuously{oom_count}OOM error, it is recommended to use an optimized version of the function or reduce the number of samples"
                    )
                    break
                logging.warning(f" Skip samples {i + 1}")
                continue
            except Exception as e:
                logging.error(
                    f" Sample {i + 1} An unexpected error occurred while processing: {e}"
                )
                continue
    except KeyboardInterrupt:
        logging.warning(" User interrupts gradient calculation")
    except Exception as e:
        logging.error(f" A serious error occurred during gradient calculation: {e}")
        raise
    logging.info(
        f" Gradient calculation completed and successfully processed {total_samples_processed}/{len(input_tokens)} samples"
    )
    if oom_count > 0:
        logging.warning(
            f" Happens during the process {oom_count} out of memory error"
        )
        logging.warning(
            f" It is recommended to use the get_gradients_optimized() function for better memory management"
        )
    logging.info(" Clean up gradient hooks...")
    for hook in hooks:
        hook.remove()
    logging.info(f" Removed {len(hooks)} gradient hook")
    logging.info(" Move the model back to the CPU to free up GPU memory...")
    model.cpu()
    if MEMORY_UTILS_AVAILABLE:
        cleanup_gpu_memory()
        memory_tracker.log_memory_status("After the model is moved to the CPU", "INFO")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        final_memory = torch.cuda.memory_allocated() / 1024**3
        logging.info(f" GPU memory usage after cleaning: {final_memory:.2f}GB")
    logging.info(" Collect calculated gradients...")
    if MEMORY_UTILS_AVAILABLE:
        memory_tracker.checkpoint("Gradient collection starts")
    gradients = []
    for layer_idx, layer in enumerate(layers):
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer).items():
            if module.weight.grad is not None:
                gradients_per_layer[module_name] = module.weight.grad
                logging.debug(
                    f" Collection layer {layer_idx} module {module_name} Gradient of: shape={module.weight.grad.shape}"
                )
            else:
                logging.warning(
                    f" Layers {layer_idx} module {module_name} no gradient"
                )
                gradients_per_layer[module_name] = None
        gradients.append(gradients_per_layer)
    logging.info(f" Collected {len(gradients)} gradient of layer")
    if MEMORY_UTILS_AVAILABLE:
        memory_tracker.compare_checkpoint(
            "Gradient collection starts", "Gradient collection completed"
        )
    if save_path is not None:
        logging.info(f" Save gradient to {save_path}...")
        if MEMORY_UTILS_AVAILABLE:
            memory_tracker.checkpoint("Gradient saving starts")
        if not save_path.endswith(".pt"):
            save_path = save_path + ".pt"
        if os.path.exists(save_path):
            backup_path = save_path + ".backup"
            logging.warning(
                f"Documents {save_path} Already exists, create a backup: {backup_path}"
            )
            os.rename(save_path, backup_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            torch.save(gradients, save_path)
            logging.info(f" Gradient has been successfully saved to {save_path}")
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / 1024**3
                logging.info(f" Save file size: {file_size:.2f}GB")
        except Exception as e:
            logging.error(f" Error while saving gradient: {e}")
            raise
        if MEMORY_UTILS_AVAILABLE:
            memory_tracker.compare_checkpoint(
                "Gradient saving starts", "Gradient saving completed"
            )
    if MEMORY_UTILS_AVAILABLE:
        memory_tracker.log_memory_status("function completed", "INFO")
        memory_tracker.compare_checkpoint("Gradient calculation starts", "final state")
    logging.info(" Gradient calculation completed successfully!")
    return gradients
