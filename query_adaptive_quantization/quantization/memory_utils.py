import torch
import gc
import logging
from typing import Optional, Dict, Any
import time

from query_adaptive_quantization.system_resources import system_memory_info

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning(
        " psutil is not available, CPU memory monitoring will be disabled"
    )


class MemoryTracker:

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoints = {}
        self.start_time = time.time()

    def get_gpu_memory_info(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(device) / 1024**3
        memory_free = (
            torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_reserved(device)
        ) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return {
            "allocated": memory_allocated,
            "cached": memory_cached,
            "free": memory_free,
            "total": memory_total,
            "utilization": memory_allocated / memory_total * 100,
        }

    def get_cpu_memory_info(self) -> Dict[str, float]:
        return system_memory_info()

    def log_memory_status(self, stage: str, level: str = "INFO"):
        gpu_info = self.get_gpu_memory_info()
        cpu_info = self.get_cpu_memory_info()
        elapsed = time.time() - self.start_time
        if gpu_info:
            msg = f"[{stage}] GPU Memory: Allocated={gpu_info['allocated']:.2f}GB, Cached={gpu_info['cached']:.2f}GB, Free={gpu_info['free']:.2f}GB, Utilization={gpu_info['utilization']:.1f}%"
        else:
            msg = f"[{stage}] GPU not available"
        msg += (
            f" | CPU Memory: Used={cpu_info['used']:.2f}GB, "
            f"Available={cpu_info['available']:.2f}GB, "
            f"Non-reclaimable={cpu_info.get('non_reclaimable', 0.0):.2f}GB, "
            f"Utilization={cpu_info['percent']:.1f}% | Elapsed={elapsed:.1f}s"
        )
        log_func = getattr(self.logger, level.lower())
        log_func(msg)
        if gpu_info and gpu_info["utilization"] > 90:
            self.logger.warning(
                f"GPU memory usage is too high: {gpu_info['utilization']:.1f}%"
            )
        if cpu_info["percent"] > 90:
            self.logger.warning(
                f"CPU memory usage is too high: {cpu_info['percent']:.1f}%"
            )

    def checkpoint(self, name: str):
        self.checkpoints[name] = {
            "gpu": self.get_gpu_memory_info(),
            "cpu": self.get_cpu_memory_info(),
            "time": time.time(),
        }
        self.logger.debug(f" Memory checkpoint settings: {name}")

    def compare_checkpoint(self, name: str, stage: str = "current"):
        if name not in self.checkpoints:
            self.logger.warning(f"checkpoint '{name}' does not exist")
            return
        checkpoint = self.checkpoints[name]
        current_gpu = self.get_gpu_memory_info()
        current_cpu = self.get_cpu_memory_info()
        time_diff = time.time() - checkpoint["time"]
        if current_gpu and checkpoint["gpu"]:
            gpu_diff = current_gpu["allocated"] - checkpoint["gpu"]["allocated"]
            self.logger.info(
                f" [{stage}] GPU memory changes: {gpu_diff:+.2f}GB ({checkpoint['gpu']['allocated']:.2f}→{current_gpu['allocated']:.2f}GB) Time taken: {time_diff:.1f}s"
            )
        cpu_diff = current_cpu["used"] - checkpoint["cpu"]["used"]
        self.logger.info(
            f" [{stage}] CPU memory changes: {cpu_diff:+.2f}GB ({checkpoint['cpu']['used']:.2f}→{current_cpu['used']:.2f}GB)"
        )


def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def optimize_memory_allocation():
    if torch.cuda.is_available():
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.empty_cache()
        import os

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_recommended_batch_size(
    model, sample_input_shape, max_memory_gb: float = None
) -> int:
    if not torch.cuda.is_available():
        return 1
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if max_memory_gb is None:
        max_memory_gb = total_memory * 0.8
    estimated_memory_per_sample = 0.5
    recommended_batch_size = max(1, int(max_memory_gb / estimated_memory_per_sample))
    logging.info(
        f" Memory optimization suggestions: Total memory ={total_memory:.1f}GB, Available ={max_memory_gb:.1f}GB, recommended batch size ={recommended_batch_size}"
    )
    return min(recommended_batch_size, 16)


class GradientAccumulator:

    def __init__(self, target_steps: int = 1):
        self.target_steps = target_steps
        self.current_step = 0

    def should_step(self) -> bool:
        self.current_step += 1
        return self.current_step % self.target_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss / self.target_steps


def monitor_gradient_memory(model, stage: str, logger: logging.Logger):
    total_grad_memory = 0
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_memory = param.grad.numel() * param.grad.element_size() / 1024**3
            total_grad_memory += grad_memory
            grad_count += 1
    logger.debug(
        f" [{stage}] Gradient memory: {total_grad_memory:.3f}GB (Total{grad_count}parameters have gradients)"
    )
    return total_grad_memory
