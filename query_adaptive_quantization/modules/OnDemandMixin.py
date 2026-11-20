import os
import torch
import torch.nn as nn
import gc
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Iterable
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PipelineTask:
    layer_idx: int
    module_idx: int
    precision: int
    frequency: int
    lut_bytes: int

class IOComputePipelinePlanner:

    def __init__(
        self,
        use_hebf: bool = True,
        bitwidth_profiles: Optional[Dict[int, Dict[str, float]]] = None,
        preserve_low_bits: bool = True,
    ) -> None:
        self.use_hebf = bool(use_hebf)
        self.bitwidth_profiles = bitwidth_profiles or {}
        self.preserve_low_bits = bool(preserve_low_bits)

    def _task_priority(self, task: PipelineTask) -> float:
        prof = self.bitwidth_profiles.get(int(task.precision), {})
        comp = float(prof.get("comp", 1.0))
        io = float(prof.get("io", 1.0))
        return float(task.frequency) * (comp / max(io, 1e-06))

    def build_schedule(self, tasks: Iterable[PipelineTask]) -> List[PipelineTask]:
        tasks = list(tasks)
        if not tasks:
            return []
        if not self.use_hebf:
            return sorted(tasks, key=lambda t: (t.layer_idx, t.module_idx, t.precision))
        return sorted(
            tasks,
            key=lambda t: (
                -self._task_priority(t),
                t.precision,
                t.layer_idx,
                t.module_idx,
            ),
        )

class OnDemandLinearLayer:

    def __init__(self, original_layer, layer_idx: int, module_idx: int):
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.module_idx = module_idx
        self.cpu_luts = {}
        self.supported_bits = original_layer.supported_bits
        saved_luts = []
        for bit in self.supported_bits:
            lut_name = f"lut{bit}"
            if hasattr(original_layer, lut_name):
                self.cpu_luts[bit] = getattr(original_layer, lut_name).cpu().clone()
                saved_luts.append(bit)
                delattr(original_layer, lut_name)
        missing_bits = set(self.supported_bits) - set(saved_luts)
        if missing_bits:
            logger.warning(
                f"layer{layer_idx}-module{module_idx}: LUTs with the following precisions have been removed by precision pruning: {missing_bits}"
            )
            logger.warning(
                f"layer{layer_idx}-module{module_idx}:Available precision: {saved_luts}"
            )
            self.supported_bits = saved_luts
        self.cpu_qweight = original_layer.qweight.cpu().clone()
        self._cpu_qweight_pinned = None
        self._cpu_luts_pinned = {}
        self.target_device = original_layer.qweight.device
        self.current_gpu_precision = None
        logger.debug(
            f"OnDemandLinearLayer initialization: layer{layer_idx}-module{module_idx}, saved{len(saved_luts)}precision LUT"
        )

    def get_target_device(self) -> torch.device:
        device = self.original_layer.qweight.device
        if device.type != "cuda" and torch.cuda.is_available():
            target = getattr(self, "target_device", None)
            if target is not None and target.type == "cuda":
                return target
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def clear_gpu_luts(
        self, specific_precision: Optional[int] = None, safe_mode: bool = False
    ):
        from typing import Optional

        cleared_count = 0
        cleared_memory_mb = 0
        if specific_precision is None and self.current_gpu_precision is not None:
            specific_precision = int(self.current_gpu_precision)
        if specific_precision is not None:
            lut_name = f"lut{specific_precision}"
            if hasattr(self.original_layer, lut_name):
                gpu_lut = getattr(self.original_layer, lut_name)
                if gpu_lut is not None:
                    cleared_memory_mb = (
                        gpu_lut.element_size() * gpu_lut.numel() / 1024 / 1024
                    )
                    if safe_mode and gpu_lut.device.type == "cuda":
                        torch.cuda.synchronize(gpu_lut.device)
                delattr(self.original_layer, lut_name)
                cleared_count += 1
                logger.debug(
                    f" Clean the specified GPU LUT: layer{self.layer_idx}-module{self.module_idx} precision={specific_precision}, release memory: {cleared_memory_mb:.2f}MB"
                )
                if self.current_gpu_precision == specific_precision:
                    self.current_gpu_precision = None
        else:
            if safe_mode and torch.cuda.is_available():
                torch.cuda.synchronize()
            for precision in self.supported_bits:
                lut_name = f"lut{precision}"
                if hasattr(self.original_layer, lut_name):
                    gpu_lut = getattr(self.original_layer, lut_name)
                    if gpu_lut is not None:
                        cleared_memory_mb += (
                            gpu_lut.element_size() * gpu_lut.numel() / 1024 / 1024
                        )
                        if safe_mode and gpu_lut.device.type == "cuda":
                            pass
                    delattr(self.original_layer, lut_name)
                    cleared_count += 1
                    logger.debug(
                        f" Clean GPU LUT: Layers{self.layer_idx}-module{self.module_idx} precision={precision}"
                    )
            self.current_gpu_precision = None
        if (
            safe_mode
            and torch.cuda.is_available()
            and bool(int(os.environ.get("QAQ_EMPTY_CACHE_ON_CLEAR", "0")))
        ):
            torch.cuda.empty_cache()
        if cleared_count > 0:
            logger.debug(
                f" GPU LUT cleanup completed: Layer{self.layer_idx}-module{self.module_idx}, cleaned up{cleared_count}LUT, release memory: {cleared_memory_mb:.2f}MB"
            )
        return {"cleared_count": cleared_count, "cleared_memory_mb": cleared_memory_mb}

    def get_memory_usage(self) -> Dict[str, Any]:
        cpu_memory = 0
        gpu_memory = 0
        for bit, lut in self.cpu_luts.items():
            cpu_memory += lut.numel() * lut.element_size()
        if self.current_gpu_precision is not None:
            lut_name = f"lut{self.current_gpu_precision}"
            if hasattr(self.original_layer, lut_name):
                gpu_lut = getattr(self.original_layer, lut_name)
                gpu_memory += gpu_lut.numel() * gpu_lut.element_size()
        return {
            "cpu_memory_mb": cpu_memory / 1024**2,
            "gpu_memory_mb": gpu_memory / 1024**2,
            "current_precision": self.current_gpu_precision,
        }

class OnDemandLayerManager:

    def __init__(self, layer_to_aplinears: List[List]):
        self.layer_to_aplinears = layer_to_aplinears
        self.on_demand_layers = []
        self.is_on_demand_mode = False
        self._prefetch_stats = defaultdict(float)
        self._prefetch_counts = defaultdict(int)
        self.collect_wait_stats_enabled = False
        self._wait_stats = defaultdict(float)
        self.qweight_offload_enabled = False
        self.qweight_offload_budget_bytes = 0
        self.qweight_pin_memory = False
        self.qweight_offload_policy = "lru"
        self.layer_prefetch = None
        self.requires_prefetch = True
        self._qweight_cache = OrderedDict()
        self._qweight_cache_bytes = 0
        self._qweight_cache_peak_bytes = 0
        logger.info(
            f"OnDemandLayerManager initialization: {len(layer_to_aplinears)} layer"
        )

    def enable_on_demand_mode(self):
        if self.is_on_demand_mode:
            return
        logger.info(" Enable on-demand mode, move LUT to CPU...")
        for layer_idx, layer_aplinears in enumerate(self.layer_to_aplinears):
            layer_on_demand = []
            for module_idx, apl in enumerate(layer_aplinears):
                on_demand_layer = OnDemandLinearLayer(apl, layer_idx, module_idx)
                try:
                    setattr(on_demand_layer.original_layer, "_on_demand_manager", self)
                except Exception:
                    pass
                try:
                    setattr(
                        on_demand_layer.original_layer,
                        "_on_demand_layer",
                        on_demand_layer,
                    )
                except Exception:
                    pass
                layer_on_demand.append(on_demand_layer)
            self.on_demand_layers.append(layer_on_demand)
        self.is_on_demand_mode = True
        logger.info(" On-demand mode enabled")

    def disable_on_demand_mode(self):
        if not self.is_on_demand_mode:
            return
        if getattr(self, "original_hbf", None) is not None:
            raise RuntimeError("Reload a resident model to disable an attached HBF runtime")
        logger.info(" Disable on-demand mode, restore LUT to GPU...")
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                for bit, cpu_lut in on_demand_layer.cpu_luts.items():
                    lut_name = f"lut{bit}"
                    gpu_lut = cpu_lut.to(on_demand_layer.get_target_device())
                    on_demand_layer.original_layer.register_buffer(lut_name, gpu_lut)
                if hasattr(on_demand_layer.original_layer, "_on_demand_manager"):
                    try:
                        delattr(on_demand_layer.original_layer, "_on_demand_manager")
                    except Exception:
                        pass
                if hasattr(on_demand_layer.original_layer, "_on_demand_layer"):
                    try:
                        delattr(on_demand_layer.original_layer, "_on_demand_layer")
                    except Exception:
                        pass
        self.on_demand_layers = []
        self.is_on_demand_mode = False
        self._qweight_cache.clear()
        self._qweight_cache_bytes = 0
        self._qweight_cache_peak_bytes = 0
        logger.info(" On-Demand Mode is disabled")

    def _touch_qweight_cache(self, on_demand_layer):
        # The controller owns cache ordering and accounts live/retired storage.
        return

    def apply_precision_strategy_pipeline(self, tasks, pipeline_config):
        controller = getattr(self, "original_hbf", None)
        if controller is None:
            raise RuntimeError("Configure the bounded HBF runtime before on-demand inference")
        return controller.accept_map(tasks)

    def clear_all_gpu_luts(self):
        if not self.is_on_demand_mode:
            return
        logger.debug("Clean LUTs on all GPUs...")
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                on_demand_layer.clear_gpu_luts(safe_mode=False)
        if torch.cuda.is_available() and bool(
            int(os.environ.get("QAQ_CLEAR_ALL_SYNC", "0"))
        ):
            torch.cuda.synchronize()
        if torch.cuda.is_available() and bool(
            int(os.environ.get("QAQ_CLEAR_ALL_EMPTY_CACHE", "0"))
        ):
            torch.cuda.empty_cache()
        if bool(int(os.environ.get("QAQ_CLEAR_ALL_GC", "0"))):
            gc.collect()

    def clear_all_gpu_weights(self):
        if not self.is_on_demand_mode:
            return
        if self.layer_prefetch is not None:
            self.layer_prefetch.reset()
            return
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                try:
                    if on_demand_layer.original_layer.qweight.is_cuda:
                        on_demand_layer.target_device = (
                            on_demand_layer.original_layer.qweight.device
                        )
                        # Rebind the buffer, releasing any previous view's _base too.
                        on_demand_layer.original_layer.qweight = (
                            on_demand_layer.cpu_qweight
                        )
                except Exception as e:
                    logger.debug(f"Cleaning qweight failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._qweight_cache.clear()
        self._qweight_cache_bytes = 0
        self._qweight_cache_peak_bytes = 0

    def get_memory_summary(self) -> Dict[str, Any]:
        if not self.is_on_demand_mode:
            return {"message": "On-demand mode is not enabled"}
        total_cpu_mb = 0
        total_gpu_mb = 0
        precision_distribution = defaultdict(int)
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                mem_stats = on_demand_layer.get_memory_usage()
                total_cpu_mb += mem_stats["cpu_memory_mb"]
                total_gpu_mb += mem_stats["gpu_memory_mb"]
                if mem_stats["current_precision"]:
                    precision_distribution[mem_stats["current_precision"]] += 1
        return {
            "total_cpu_lut_mb": total_cpu_mb,
            "total_gpu_lut_mb": total_gpu_mb,
            "gpu_memory_saved_mb": total_cpu_mb - total_gpu_mb,
            "precision_distribution": dict(precision_distribution),
        }

    def get_total_qweight_bytes(self) -> int:
        total = 0
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                cpu_qweight = getattr(on_demand_layer, "cpu_qweight", None)
                if cpu_qweight is not None:
                    total += cpu_qweight.numel() * cpu_qweight.element_size()
        return total

    def configure_qweight_offload(
        self,
        enable: bool,
        budget_gb: Optional[float] = None,
        pin_memory: bool = False,
        policy: str = "lru",
    ) -> None:
        self.qweight_offload_enabled = bool(enable)
        self.qweight_pin_memory = bool(pin_memory)
        self.qweight_offload_policy = str(policy)
        if budget_gb is not None:
            self.qweight_offload_budget_bytes = int(float(budget_gb) * 1024**3)
        if self.qweight_offload_enabled:
            self.clear_all_gpu_weights()
        else:
            self._qweight_cache.clear()
            self._qweight_cache_bytes = 0
            self._qweight_cache_peak_bytes = 0

    def clear_pending_events(self) -> None:
        for layer_on_demand in self.on_demand_layers:
            for on_demand_layer in layer_on_demand:
                original_layer = on_demand_layer.original_layer
                if hasattr(original_layer, "clear_ready_events"):
                    original_layer.clear_ready_events()
                elif hasattr(original_layer, "_lut_ready_events"):
                    try:
                        original_layer._lut_ready_events.clear()
                    except Exception:
                        pass
                if hasattr(original_layer, "_qweight_ready_event"):
                    original_layer._qweight_ready_event = None

    def collect_prefetch_stats(
        self, sync: bool = False, clear: bool = False
    ) -> Dict[str, Any]:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        stats = dict(self._prefetch_stats)
        counts = dict(self._prefetch_counts)
        stats.update({f"{k}_count": v for k, v in counts.items()})
        if clear:
            self._prefetch_stats.clear()
            self._prefetch_counts.clear()
        return stats

    def reset_wait_stats(self) -> None:
        self._wait_stats.clear()

    def collect_wait_stats(self, clear: bool = False) -> Dict[str, Any]:
        stats = dict(self._wait_stats)
        if clear:
            self._wait_stats.clear()
        return stats

    def validate_weight_loading(self) -> Dict[str, Any]:
        validation_results = {
            "total_layers": 0,
            "total_modules": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "failed_details": [],
        }
        if not self.is_on_demand_mode:
            return {"error": "On-demand mode is not enabled"}
        for layer_idx, layer_on_demand in enumerate(self.on_demand_layers):
            validation_results["total_layers"] += 1
            for module_idx, on_demand_layer in enumerate(layer_on_demand):
                validation_results["total_modules"] += 1
                try:
                    if not on_demand_layer.cpu_luts:
                        validation_results["validation_failed"] += 1
                        validation_results["failed_details"].append(
                            {
                                "layer_idx": layer_idx,
                                "module_idx": module_idx,
                                "error": "CPU LUT is empty",
                            }
                        )
                        continue
                    if on_demand_layer.current_gpu_precision is not None:
                        precision = on_demand_layer.current_gpu_precision
                        lut_name = f"lut{precision}"
                        if not hasattr(on_demand_layer.original_layer, lut_name):
                            validation_results["validation_failed"] += 1
                            validation_results["failed_details"].append(
                                {
                                    "layer_idx": layer_idx,
                                    "module_idx": module_idx,
                                    "error": f"LUT missing on GPU{precision}",
                                }
                            )
                            continue
                        gpu_lut = getattr(on_demand_layer.original_layer, lut_name)
                        cpu_lut = on_demand_layer.cpu_luts[precision]
                        if not torch.equal(gpu_lut.cpu(), cpu_lut):
                            validation_results["validation_failed"] += 1
                            validation_results["failed_details"].append(
                                {
                                    "layer_idx": layer_idx,
                                    "module_idx": module_idx,
                                    "error": f"GPU LUT is inconsistent with CPU LUT (precision={precision})",
                                }
                            )
                            continue
                    validation_results["validation_passed"] += 1
                except Exception as e:
                    validation_results["validation_failed"] += 1
                    validation_results["failed_details"].append(
                        {
                            "layer_idx": layer_idx,
                            "module_idx": module_idx,
                            "error": f"Verification exception: {str(e)}",
                        }
                    )
        total_modules = validation_results["total_modules"]
        if total_modules > 0:
            validation_results["success_rate"] = (
                validation_results["validation_passed"] / total_modules
            )
        else:
            validation_results["success_rate"] = 0.0
        return validation_results

class OnDemandMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_demand_enabled = False
        self.on_demand_manager = None
        self.on_demand_memory_limit_gb = 4.0
        self.use_pipeline_scheduling = False
        self.pipeline_config = {}
        self.on_demand_stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "memory_savings": [],
            "precision_usage": defaultdict(int),
        }

    def enable_on_demand_inference(
        self,
        memory_limit_gb: float = 4.0,
        use_batch_optimization: bool = False,
        batch_size: int = 64,
    ):
        if self.on_demand_enabled:
            logger.warning("On-demand inference is enabled")
            return
        logger.info(
            f" Enable on-demand inference mode, memory limit: {memory_limit_gb}GB"
        )
        if use_batch_optimization:
            raise ValueError("Legacy batch loading is not supported")
        if not hasattr(self, "layer_to_aplinears") or not self.layer_to_aplinears:
            raise RuntimeError(
                "The model has not been initialized yet, please load the model first"
            )
        self.on_demand_manager = OnDemandLayerManager(self.layer_to_aplinears)
        self.on_demand_manager.enable_on_demand_mode()
        self.on_demand_enabled = True
        self.on_demand_memory_limit_gb = memory_limit_gb
        self.use_batch_optimization = use_batch_optimization
        self.batch_size = batch_size
        logger.info(" On-demand inference mode is enabled")

    def configure_qweight_offload(
        self,
        enable: bool = False,
        budget_gb: Optional[float] = None,
        pin_memory: bool = False,
        policy: str = "lru",
    ) -> None:
        if not self.on_demand_manager:
            logger.warning(
                "On-demand manager not initialized, unable to configure qweight offload"
            )
            return
        self.on_demand_manager.configure_qweight_offload(
            enable=enable, budget_gb=budget_gb, pin_memory=pin_memory, policy=policy
        )

    def disable_on_demand_inference(self):
        if not self.on_demand_enabled:
            logger.warning("On-demand inference is not enabled")
            return
        logger.info(" Disable on-demand inference mode")
        if self.on_demand_manager:
            self.on_demand_manager.disable_on_demand_mode()
            self.on_demand_manager = None
        self.on_demand_enabled = False
        logger.info(" On-demand inference mode is disabled")

    def get_on_demand_stats(self) -> Dict[str, Any]:
        if not self.on_demand_enabled:
            return {"message": "On-demand inference is not enabled"}
        avg_memory_saved = 0
        if self.on_demand_stats["memory_savings"]:
            avg_memory_saved = sum(
                (
                    s["gpu_memory_saved_mb"]
                    for s in self.on_demand_stats["memory_savings"]
                )
            ) / len(self.on_demand_stats["memory_savings"])
        return {
            "on_demand_enabled": self.on_demand_enabled,
            "total_inferences": self.on_demand_stats["total_inferences"],
            "total_time": self.on_demand_stats["total_time"],
            "average_memory_saved_mb": avg_memory_saved,
            "precision_usage": dict(self.on_demand_stats["precision_usage"]),
            "memory_summary": (
                self.on_demand_manager.get_memory_summary()
                if self.on_demand_manager
                else None
            ),
        }

    def _apply_on_demand_precision_strategy(self, *args, **kwargs):
        if not self.on_demand_enabled or self.on_demand_manager is None:
            return
        precision_map = kwargs.get("precision_map", kwargs.get("precision"))
        if precision_map is None:
            raise ValueError("Resolve the Router map once before resource scheduling")
        if isinstance(precision_map, int):
            precision_map = {i: precision_map for i in range(len(self.layer_to_aplinears))}
        tasks = self._build_pipeline_tasks(precision_map)
        result = self.on_demand_manager.apply_precision_strategy_pipeline(tasks, self.pipeline_config)
        self.on_demand_stats.setdefault("pipeline_results", []).append(result)

    def _build_pipeline_tasks(
        self, precision_map: Any, tokenwise_frequency: bool = False
    ) -> List[PipelineTask]:
        if not self.on_demand_manager:
            return []
        layer_modules = self.on_demand_manager.on_demand_layers
        if not layer_modules:
            return []

        def _module_names() -> List[str]:
            try:
                return list(self.config.anyprec["arch_config"]["module_names"])
            except Exception:
                return []

        module_names = _module_names()
        mha_modules = set()
        mlp_modules = set()
        for name in module_names:
            lname = name.lower()
            if "self_attn" in lname or "attn" in lname:
                mha_modules.add(name)
            elif "mlp" in lname or "fc" in lname:
                mlp_modules.add(name)

        def _flatten_single_map(single_map: Any) -> Dict[Tuple[int, int], int]:
            flat: Dict[Tuple[int, int], int] = {}
            if not isinstance(single_map, dict):
                return flat
            if any((isinstance(k, tuple) and len(k) == 2 for k in single_map.keys())):
                for (layer_idx, module_name), bit in single_map.items():
                    if not isinstance(layer_idx, int) or not isinstance(bit, int):
                        continue
                    if module_name in module_names:
                        module_idx = module_names.index(module_name)
                        flat[layer_idx, module_idx] = bit
                return flat
            for layer_idx, layer_strategy in single_map.items():
                if not isinstance(layer_idx, int) or layer_idx >= len(layer_modules):
                    continue
                module_count = len(layer_modules[layer_idx])
                if isinstance(layer_strategy, int):
                    for module_idx in range(module_count):
                        flat[layer_idx, module_idx] = layer_strategy
                elif isinstance(layer_strategy, dict):
                    if "mha" in layer_strategy or "mlp" in layer_strategy:
                        for module_idx in range(module_count):
                            module_name = (
                                module_names[module_idx]
                                if module_idx < len(module_names)
                                else ""
                            )
                            if module_name in mha_modules and "mha" in layer_strategy:
                                flat[layer_idx, module_idx] = int(layer_strategy["mha"])
                            elif module_name in mlp_modules and "mlp" in layer_strategy:
                                flat[layer_idx, module_idx] = int(layer_strategy["mlp"])
                    else:
                        for module_name, bit in layer_strategy.items():
                            if module_name in module_names and isinstance(bit, int):
                                module_idx = module_names.index(module_name)
                                flat[layer_idx, module_idx] = bit
            return flat

        frequency_map: Dict[Tuple[int, int, int], int] = defaultdict(int)
        if tokenwise_frequency and isinstance(precision_map, (list, tuple)):
            for entry in precision_map:
                flat = _flatten_single_map(entry)
                for (layer_idx, module_idx), bit in flat.items():
                    frequency_map[layer_idx, module_idx, bit] += 1
        else:
            flat = _flatten_single_map(precision_map)
            for (layer_idx, module_idx), bit in flat.items():
                frequency_map[layer_idx, module_idx, bit] += 1
        tasks: List[PipelineTask] = []
        for (layer_idx, module_idx, bit), freq in frequency_map.items():
            if layer_idx >= len(layer_modules) or module_idx >= len(
                layer_modules[layer_idx]
            ):
                continue
            on_demand_layer = layer_modules[layer_idx][module_idx]
            if bit not in on_demand_layer.cpu_luts:
                if on_demand_layer.supported_bits:
                    bit = max(on_demand_layer.supported_bits)
                else:
                    continue
            lut = on_demand_layer.cpu_luts.get(bit)
            lut_bytes = int(lut.numel() * lut.element_size()) if lut is not None else 0
            tasks.append(
                PipelineTask(
                    layer_idx=layer_idx,
                    module_idx=module_idx,
                    precision=int(bit),
                    frequency=int(freq),
                    lut_bytes=lut_bytes,
                )
            )
        return tasks

    def _cleanup_on_demand_weights(self):
        if self.on_demand_enabled and self.on_demand_manager:
            cleanup = bool(int(os.environ.get("QAQ_CLEANUP_LUTS_AFTER_GENERATE", "0")))
            cfg = getattr(self, "pipeline_config", None)
            if isinstance(cfg, dict) and bool(
                cfg.get("cleanup_luts_after_generate", False)
            ):
                cleanup = True
            if cleanup:
                self.on_demand_manager.clear_all_gpu_luts()
            elif bool(int(os.environ.get("QAQ_CLEAR_EVENTS_AFTER_GENERATE", "0"))):
                try:
                    self.on_demand_manager.clear_pending_events()
                except Exception:
                    pass
