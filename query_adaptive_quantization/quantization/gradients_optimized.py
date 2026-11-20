import torch
from tqdm import tqdm
import os
import logging
import gc
import math
from contextlib import contextmanager
from .config import *
from query_adaptive_quantization.analyzer.analyzer import get_analyzer
from .datautils import get_tokens
from .gradient_store import LayerGradientStore
from .memory_utils import (
    MemoryTracker,
    cleanup_gpu_memory,
    optimize_memory_allocation,
    monitor_gradient_memory,
    GradientAccumulator,
)
from .salience import scaled_squared_gradient


def _nonzero_dropout_settings(model) -> dict[str, float]:
    nonzero: dict[str, float] = {}
    config = getattr(model, "config", None)
    payload = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}

    def visit(value, prefix: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if "dropout" in str(key).lower() and isinstance(child, (int, float)):
                    if float(child) != 0.0:
                        nonzero[f"config.{path}"] = float(child)
                else:
                    visit(child, path)

    visit(payload)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0:
            nonzero[f"module.{name}"] = float(module.p)
    return nonzero


@contextmanager
def _gradient_collection_mode(model, *, gradient_checkpointing: bool):
    original_training = bool(model.training)
    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None)
    original_checkpointing = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    if config is not None and original_use_cache is not None:
        config.use_cache = False
    try:
        if gradient_checkpointing:
            nonzero_dropout = _nonzero_dropout_settings(model)
            if nonzero_dropout:
                raise ValueError(
                    "Gradient checkpointing requires deterministic training mode; "
                    f"non-zero dropout settings were found: {nonzero_dropout}"
                )
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.train()
        else:
            model.eval()
        yield
    finally:
        if gradient_checkpointing and not original_checkpointing:
            model.gradient_checkpointing_disable()
        model.train(original_training)
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache


def get_gradients_optimized(
    analyzer,
    dataset=DEFAULT_DATASET,
    seq_len=DEFAULT_SEQ_LEN,
    num_examples=DEFAULT_NUM_EXAMPLES,
    save_path=None,
    random_state=None,
    gradient_accumulation_steps=4,
    memory_check_interval=10,
    enable_checkpointing=True,
):
    memory_tracker = MemoryTracker(logger=logging.getLogger(__name__))
    memory_tracker.log_memory_status("function start", "INFO")
    if save_path is not None and os.path.isfile(save_path):
        logging.info(f" Gradient is calculated and saved in {save_path}")
        logging.info(f" Loading gradients...")
        return torch.load(save_path, weights_only=False)
    logging.info(f" Start the optimized version of gradient calculation:")
    logging.info(f"   - Dataset: {dataset}")
    logging.info(f"   - Sequence length: {seq_len}")
    logging.info(f"   - Sample size: {num_examples}")
    logging.info(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"   - Memory check interval: {memory_check_interval}")
    optimize_memory_allocation()
    memory_tracker.checkpoint("initialization")
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    logging.info(f" Get input tokens...")
    input_tokens = get_tokens(
        dataset, "train", tokenizer, seq_len, num_examples, seed=random_state
    )
    logging.info(f" Loaded successfully {len(input_tokens)} input tokens")
    memory_tracker.compare_checkpoint("initialization", "Tokens loading completed")
    if analyzer is None:
        analyzer = get_analyzer(model)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_name = "GPU" if use_cuda else "CPU"
    logging.info(f" Convert model to bfloat16 and move to{device_name}...")
    memory_tracker.checkpoint("Before model conversion")
    model = model.bfloat16()
    model.eval()
    if use_cuda and model.device.type != "cuda":
        model.cuda()
    elif not use_cuda and model.device.type != "cpu":
        model.cpu()
    logging.info(" Make sure the model parameters require gradients...")
    requires_grad_count_before = sum((1 for p in model.parameters() if p.requires_grad))
    logging.info(
        f" Number of parameters required for gradient before enabling: {requires_grad_count_before}"
    )
    for param in model.parameters():
        param.requires_grad_(True)
    requires_grad_count_after = sum((1 for p in model.parameters() if p.requires_grad))
    logging.info(
        f" Number of parameters that require gradients after enabling: {requires_grad_count_after}"
    )
    if use_cuda:
        cleanup_gpu_memory()
    else:
        import gc

        gc.collect()
    logging.info(f" Model deployed to device: {model.device}")
    memory_tracker.compare_checkpoint(
        "Before model conversion", "Model conversion completed"
    )
    try:
        analyzer.state_dict = model.state_dict()
    except Exception as e:
        logging.warning(f" Failed to refresh analyzer.state_dict: {e}")
    layers = analyzer.get_layers()
    logging.info(f" Discover {len(layers)} layers of analysis")

    def square_grad_hook(grad):
        try:
            return scaled_squared_gradient(grad)
        except RuntimeError as e:
            logging.error(f" Gradient hook error: {e}")
            return scaled_squared_gradient(grad.detach().cpu()).to(grad.device)

    hooks = []
    logging.info(" Register gradient hook...")
    memory_tracker.checkpoint("Before hook registration")
    for layer in layers:
        for module in analyzer.get_modules(layer).values():
            hooks.append(module.weight.register_hook(square_grad_hook))
    logging.info(f" Already registered {len(hooks)} gradient hook")
    memory_tracker.compare_checkpoint(
        "Before hook registration", "Hook registration completed"
    )
    accumulator = GradientAccumulator(target_steps=gradient_accumulation_steps)
    logging.info(
        f" Start gradient calculation, total number of samples: {len(input_tokens)}"
    )
    logging.info(
        f" Use gradient accumulation strategy, every {gradient_accumulation_steps} Clean once"
    )
    memory_tracker.checkpoint("Gradient calculation starts")
    total_samples_processed = 0
    model.zero_grad()
    logging.info(
        " The initial gradient has been cleaned and started to accumulate..."
    )
    try:
        for i, tokens in enumerate(tqdm(input_tokens, desc="Calculate gradient")):
            try:
                tokens = tokens.to(model.device)
                tokens = tokens.unsqueeze(0)
                if use_cuda:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=tokens, labels=tokens)
                        loss = outputs.loss
                else:
                    outputs = model(input_ids=tokens, labels=tokens)
                    loss = outputs.loss
                scaled_loss = accumulator.scale_loss(loss)
                scaled_loss.backward()
                total_samples_processed += 1
                if accumulator.should_step():
                    cleanup_gpu_memory()
                    if i % memory_check_interval == 0:
                        memory_tracker.log_memory_status(
                            f"sample {i + 1}/{len(input_tokens)}", "DEBUG"
                        )
                        monitor_gradient_memory(
                            model, f"step_{i + 1}", logging.getLogger(__name__)
                        )
                if (i + 1) % 20 == 0 or i + 1 == len(input_tokens):
                    logging.info(
                        f" Processed {i + 1}/{len(input_tokens)} Sample, current loss: {loss.item():.4f}, scaling loss: {scaled_loss.item():.4f}"
                    )
                    if (i + 1) % 50 == 0 or i + 1 == len(input_tokens):
                        memory_tracker.log_memory_status(
                            f"progress checkpoint {i + 1}", "INFO"
                        )
                if enable_checkpointing and (i + 1) % 50 == 0 and save_path:
                    checkpoint_path = f"{save_path}.checkpoint_{i + 1}"
                    logging.info(f" Save intermediate checkpoints: {checkpoint_path}")
            except torch.cuda.OutOfMemoryError as e:
                logging.error(
                    f" Sample {i + 1} Out of GPU memory occurred while processing: {e}"
                )
                model.zero_grad()
                cleanup_gpu_memory()
                memory_tracker.log_memory_status("OOM recovery attempt", "ERROR")
                logging.warning(f" Skip samples {i + 1} to avoid program crashes")
                continue
            except Exception as e:
                logging.error(
                    f" Sample {i + 1} An unexpected error occurred while processing: {e}"
                )
                continue
    except KeyboardInterrupt:
        logging.warning(" User interruption, saving calculated gradients...")
    except Exception as e:
        logging.error(f" A serious error occurred during gradient calculation: {e}")
        raise
    finally:
        logging.info(" Clean up resources...")
        for hook in hooks:
            hook.remove()
        logging.info(f" Removed {len(hooks)} gradient hook")
        cleanup_gpu_memory()
    memory_tracker.compare_checkpoint(
        "Gradient calculation starts", "Gradient calculation completed"
    )
    logging.info(
        f" Gradient calculation completed and processed together {total_samples_processed} samples"
    )
    logging.info(" Final check before gradient collection...")
    grad_count = sum((1 for p in model.parameters() if p.grad is not None))
    total_params = sum((1 for p in model.parameters()))
    logging.info(
        f" Final gradient state: {grad_count}/{total_params} Parameters have gradients"
    )
    logging.info(" Move the model back to the CPU to free up GPU memory...")
    model.cpu()
    cleanup_gpu_memory()
    memory_tracker.log_memory_status("After the model is moved to the CPU", "INFO")
    logging.info(" Collect calculated gradients...")
    memory_tracker.checkpoint("Gradient collection starts")
    gradients = []
    gradient_found_count = 0
    gradient_missing_count = 0
    for layer_idx, layer in enumerate(layers):
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer).items():
            if module.weight.grad is not None:
                gradients_per_layer[module_name] = module.weight.grad.cpu()
                gradient_found_count += 1
                logging.debug(
                    f" Collection layer {layer_idx} module {module_name} Gradient of: shape={module.weight.grad.shape}"
                )
            else:
                logging.warning(
                    f" Layers {layer_idx} module {module_name} no gradient"
                )
                gradients_per_layer[module_name] = None
                gradient_missing_count += 1
        gradients.append(gradients_per_layer)
    memory_tracker.compare_checkpoint(
        "Gradient collection starts", "Gradient collection completed"
    )
    logging.info(f" Collected {len(gradients)} gradient of layer")
    logging.info(
        f"Gradient collection statistics: {gradient_found_count} turn up, {gradient_missing_count} Missing"
    )
    if gradient_found_count == 0:
        logging.error(" Fatal error: No gradients collected!")
        logging.error(" Recommended checks:")
        logging.error(
            "   1. Are the model parameters correctly set with requires_grad=True?"
        )
        logging.error("   2. Whether the loss is calculated correctly")
        logging.error("   3. Whether backpropagation is executed successfully")
        logging.error(
            "   4. Whether to run in CPU mode (may require special processing)"
        )
    elif gradient_missing_count > 0:
        logging.warning(
            f" Some modules are missing gradients: {gradient_missing_count}/{gradient_found_count + gradient_missing_count}"
        )
    if save_path is not None:
        logging.info(f" Save gradient to {save_path}...")
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
            file_size = os.path.getsize(save_path) / 1024**3
            logging.info(f" Save file size: {file_size:.2f}GB")
        except Exception as e:
            logging.error(f" Error while saving gradient: {e}")
            raise
        memory_tracker.compare_checkpoint(
            "Gradient saving starts", "Gradient saving completed"
        )
    memory_tracker.log_memory_status("function completed", "INFO")
    memory_tracker.compare_checkpoint("initialization", "final state")
    logging.info(
        " The optimized version of gradient calculation is successfully completed!"
    )
    return gradients


def _get_gradients_chunked_by_layers(
    analyzer,
    dataset,
    seq_len,
    num_examples,
    save_path,
    random_state,
    gradient_accumulation_steps,
    memory_check_interval,
    chunk_layer_count,
    gradient_checkpointing=False,
    layer_limit=None,
    force_concatenation=False,
):
    memory_tracker = MemoryTracker(logger=logging.getLogger(__name__))
    memory_tracker.log_memory_status("chunked gradient calculation starts", "INFO")
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    logging.info(f" Get input tokens...")
    input_tokens = get_tokens(
        dataset,
        "train",
        tokenizer,
        seq_len,
        num_examples,
        seed=random_state,
        force_concatenation=force_concatenation,
    )
    logging.info(f" Loaded successfully {len(input_tokens)} input tokens")
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise RuntimeError(
            "Chunked gradient computation requires CUDA, but no GPU is available."
        )
    optimize_memory_allocation()
    model = model.bfloat16()
    try:
        if getattr(model, "device", None) is None or model.device.type != "cuda":
            model.cuda()
    except Exception as e:
        logging.warning(
            f" Attempts to move the model to the GPU failed and will continue to use the current device: {e}"
        )
    try:
        analyzer.state_dict = model.state_dict()
    except Exception as e:
        logging.warning(f" Failed to refresh analyzer.state_dict: {e}")
    cleanup_gpu_memory()
    memory_tracker.log_memory_status("After refreshing state_dict", "INFO")
    layers = analyzer.get_layers()
    num_layers = len(layers)
    layer_modules = [analyzer.get_modules(layer) for layer in layers]
    gradient_store = None
    if save_path is not None:
        gradient_store = LayerGradientStore.open_if_present(save_path)
        if gradient_store is None:
            gradient_store = LayerGradientStore(
                save_path,
                num_layers=num_layers,
                module_names=analyzer.module_names,
            )
        elif (
            gradient_store.num_layers != num_layers
            or gradient_store.module_names != tuple(analyzer.module_names)
        ):
            raise ValueError(
                "Existing gradient store does not match the current model: "
                f"layers={gradient_store.num_layers}/{num_layers}, "
                f"modules={gradient_store.module_names}/{tuple(analyzer.module_names)}"
            )
        gradient_store.initialize()
        completed_layers = set(gradient_store.completed_layers())
        if completed_layers:
            logging.info(
                "Resuming layer-sharded gradients with "
                f"{len(completed_layers)}/{num_layers} layers complete"
            )
        gradients = None
    else:
        completed_layers = set()
        gradients = [
            {name: None for name in analyzer.module_names}
            for _ in range(num_layers)
        ]
    model.requires_grad_(False)
    target_layer_count = (
        num_layers if layer_limit is None else min(int(layer_limit), num_layers)
    )
    total_chunks = math.ceil(target_layer_count / chunk_layer_count)
    logging.info(
        "Use chunked gradient calculation: "
        f"layers_per_chunk={chunk_layer_count}, chunks={total_chunks}, "
        f"target_layers={target_layer_count}/{num_layers}, "
        f"gradient_checkpointing={gradient_checkpointing}"
    )

    def square_grad_hook(grad):
        return scaled_squared_gradient(grad)

    with _gradient_collection_mode(
        model, gradient_checkpointing=gradient_checkpointing
    ):
        for chunk_idx, start in enumerate(
            range(0, target_layer_count, chunk_layer_count)
        ):
            end = min(start + chunk_layer_count, target_layer_count)
            active_layer_indices = [
                layer_index
                for layer_index in range(start, end)
                if layer_index not in completed_layers
            ]
            if not active_layer_indices:
                logging.info(
                    f"Skip completed gradient chunk [{start}:{end})"
                )
                continue
            hooks = []
            collected_gradients: dict[
                int, dict[str, torch.Tensor | None]
            ] = {layer_index: {} for layer_index in active_layer_indices}
            try:
                for layer_idx in active_layer_indices:
                    for module in layer_modules[layer_idx].values():
                        if not hasattr(module, "weight") or module.weight is None:
                            continue
                        module.weight.requires_grad_(True)
                        hooks.append(module.weight.register_hook(square_grad_hook))
                model.zero_grad(set_to_none=True)
                cleanup_gpu_memory()
                memory_tracker.log_memory_status(
                    f"chunk {chunk_idx + 1}/{total_chunks} start [{start}:{end})",
                    "INFO",
                )
                for i, tokens in enumerate(
                    tqdm(
                        input_tokens,
                        desc=(
                            "Calculate the gradient "
                            f"(chunk {chunk_idx + 1}/{total_chunks})"
                        ),
                    )
                ):
                    tokens = tokens.to(model.device).unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids=tokens, labels=tokens)
                        loss = outputs.loss
                    scaled_loss = loss / max(1, int(gradient_accumulation_steps))
                    scaled_loss.backward()
                    if (
                        memory_check_interval
                        and (i + 1) % memory_check_interval == 0
                    ):
                        cleanup_gpu_memory()
                for layer_idx in active_layer_indices:
                    for module_name, module in layer_modules[layer_idx].items():
                        if not hasattr(module, "weight") or module.weight is None:
                            collected_gradients[layer_idx][module_name] = None
                            continue
                        grad = module.weight.grad
                        if grad is None:
                            logging.warning(
                                f"Layer {layer_idx} module {module_name} has no gradient"
                            )
                            collected_gradients[layer_idx][module_name] = None
                        else:
                            collected_gradients[layer_idx][module_name] = (
                                grad.detach().cpu()
                            )
                        module.weight.grad = None
                        module.weight.requires_grad_(False)
                if gradient_store is not None:
                    for layer_idx in active_layer_indices:
                        gradient_store.save_layer(
                            layer_idx,
                            collected_gradients[layer_idx],
                        )
                        completed_layers.add(layer_idx)
                else:
                    for layer_idx in active_layer_indices:
                        gradients[layer_idx] = collected_gradients[layer_idx]
                del collected_gradients
                gc.collect()
                memory_tracker.log_memory_status(
                    f"chunk {chunk_idx + 1}/{total_chunks} done [{start}:{end})",
                    "INFO",
                )
                cleanup_gpu_memory()
            finally:
                for hook in hooks:
                    try:
                        hook.remove()
                    except Exception:
                        pass
                for layer_idx in active_layer_indices:
                    for module in layer_modules[layer_idx].values():
                        if hasattr(module, "weight") and module.weight is not None:
                            module.weight.requires_grad_(False)
                            module.weight.grad = None
    if gradient_store is not None:
        gradient_store.mark_complete()
        found = len(gradient_store.completed_layers()) * len(
            analyzer.module_names
        )
    else:
        found = sum(
            1
            for layer in gradients
            for gradient in layer.values()
            if gradient is not None
        )
    total = num_layers * len(analyzer.module_names)
    logging.info(
        f" Blocked gradient collection statistics: {found}/{total} modules get the gradient"
    )
    if found == 0:
        raise RuntimeError("No gradients were collected (chunked mode)")
    if save_path is not None and gradient_store is None:
        logging.info(f" Save gradient to {save_path}...")
        if not save_path.endswith(".pt"):
            save_path = save_path + ".pt"
        if os.path.exists(save_path):
            backup_path = save_path + ".backup"
            logging.warning(
                f"Documents {save_path} Already exists, create a backup: {backup_path}"
            )
            os.rename(save_path, backup_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)
        file_size = os.path.getsize(save_path) / 1024**3
        logging.info(
            f" Gradient has been successfully saved to {save_path} (size={file_size:.2f}GB)"
        )
    memory_tracker.log_memory_status("chunked gradient calculation completed", "INFO")
    return gradient_store if gradient_store is not None else gradients


def get_gradients_with_memory_limit(
    analyzer,
    dataset=DEFAULT_DATASET,
    seq_len=DEFAULT_SEQ_LEN,
    num_examples=DEFAULT_NUM_EXAMPLES,
    save_path=None,
    random_state=None,
    max_memory_gb=40.0,
    emergency_checkpoint_interval=25,
    gradient_accumulation_steps=4,
    memory_check_interval=10,
    gradient_checkpointing=False,
):
    memory_tracker = MemoryTracker(logger=logging.getLogger(__name__))
    memory_tracker.log_memory_status(
        "Gradient calculation starts with memory limit", "INFO"
    )
    if save_path is not None:
        gradient_store = LayerGradientStore.open_if_present(save_path)
        if gradient_store is not None and gradient_store.is_complete():
            logging.info(
                "Layer-sharded gradients are complete at "
                f"{gradient_store.root}"
            )
            return gradient_store
        if os.path.isfile(save_path):
            logging.info(f"Gradient cache found at {save_path}")
            try:
                return torch.load(save_path, weights_only=False)
            except Exception as exc:
                invalid_path = f"{save_path}.invalid"
                suffix = 1
                while os.path.exists(invalid_path):
                    invalid_path = f"{save_path}.invalid.{suffix}"
                    suffix += 1
                os.replace(save_path, invalid_path)
                logging.warning(
                    "Invalid monolithic gradient cache was quarantined at "
                    f"{invalid_path}: {exc}"
                )

    def check_memory_and_cleanup():
        gpu_info = memory_tracker.get_gpu_memory_info()
        if gpu_info and gpu_info["allocated"] > max_memory_gb * 0.9:
            logging.warning(
                f" Approaching memory limit ({gpu_info['allocated']:.1f}/{max_memory_gb}GB), perform an emergency cleanup"
            )
            cleanup_gpu_memory()
            return True
        return False

    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        logging.info(
            " No available GPU detected, CPU compatible optimized gradient calculation will be used"
        )
        return get_gradients_optimized(
            analyzer=analyzer,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            save_path=save_path,
            random_state=random_state,
            gradient_accumulation_steps=gradient_accumulation_steps,
            memory_check_interval=memory_check_interval,
            enable_checkpointing=True,
        )
    logging.info(
        " Detects GPU, uses tiled gradient calculation to control memory peaking"
    )
    memory_tracker.log_memory_status("Before chunked mode is enabled", "INFO")
    candidate_chunk_sizes = [1] if gradient_checkpointing else [4, 2, 1]
    for chunk_layer_count in candidate_chunk_sizes:
        if chunk_layer_count > analyzer.num_layers:
            continue
        try:
            return _get_gradients_chunked_by_layers(
                analyzer=analyzer,
                dataset=dataset,
                seq_len=seq_len,
                num_examples=num_examples,
                save_path=save_path,
                random_state=random_state,
                gradient_accumulation_steps=gradient_accumulation_steps,
                memory_check_interval=memory_check_interval,
                chunk_layer_count=chunk_layer_count,
                gradient_checkpointing=gradient_checkpointing,
            )
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(
                f" chunked gradient calculation OOM (layers_per_chunk={chunk_layer_count}): {e}"
            )
            cleanup_gpu_memory()
            continue
    raise RuntimeError(
        "Chunked gradient calculation failed: OOM occurs for all chunk sizes"
    )


def run_gradient_collection_smoke(
    analyzer,
    *,
    dataset=DEFAULT_DATASET,
    seq_len=DEFAULT_SEQ_LEN,
    random_state=None,
    max_peak_memory_gib=30.5,
    gradient_checkpointing=True,
):
    """Collect one layer/sample and report whether the 14B-safe path is viable."""

    if not torch.cuda.is_available():
        raise RuntimeError("Gradient resource smoke requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gradients = _get_gradients_chunked_by_layers(
        analyzer=analyzer,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=1,
        save_path=None,
        random_state=random_state,
        gradient_accumulation_steps=1,
        memory_check_interval=1,
        chunk_layer_count=1,
        gradient_checkpointing=gradient_checkpointing,
        layer_limit=1,
        force_concatenation=True,
    )
    present = [
        gradient
        for layer in gradients
        for gradient in layer.values()
        if gradient is not None
    ]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in present)
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    report = {
        "status": "passed" if present and finite and peak_gib < max_peak_memory_gib else "failed",
        "gradient_checkpointing": bool(gradient_checkpointing),
        "collected_modules": len(present),
        "finite": finite,
        "peak_allocated_gib": peak_gib,
        "max_peak_memory_gib": float(max_peak_memory_gib),
    }
    del gradients, present
    cleanup_gpu_memory()
    if report["status"] != "passed":
        raise RuntimeError(f"Gradient resource smoke failed: {report}")
    return report
