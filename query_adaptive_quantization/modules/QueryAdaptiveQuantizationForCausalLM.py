import gc
from collections import defaultdict
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import init_empty_weights, load_checkpoint_and_dispatch
from .QueryAdaptiveQuantizationLinear import QueryAdaptiveQuantizationLinear
from .Router import OnlineLearningPrecisionRouter
from .OnDemandMixin import OnDemandMixin
from query_adaptive_quantization.analyzer.analyzer import get_analyzer
from query_adaptive_quantization.batch_routing import shared_precision_map


def setup_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def replace_module_by_name(layer, module_name, new_module):
    logger = logging.getLogger(__name__)
    logger.debug(f"Start replacing modules: {module_name}")
    logger.debug(
        f"Original module type: {type(getattr(layer, module_name.split('.')[0], None))}"
    )
    logger.debug(f"New module types: {type(new_module)}")
    levels = module_name.split(".")
    module = layer
    for i, level in enumerate(levels[:-1]):
        logger.debug(f"access level {i}: {level}")
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
        logger.debug(f"Current module type: {type(module)}")
    logger.debug(f"Set final properties: {levels[-1]}")
    setattr(module, levels[-1], new_module)
    logger.debug(f"Module replacement completed: {module_name}")


class QueryAdaptiveQuantizationForCausalLM(OnDemandMixin, nn.Module):
    """Packed transformer with attention/feed-forward precision control and an optional Router."""
    router_class = OnlineLearningPrecisionRouter


    def __init__(
        self,
        model_path,
        config,
        precisions=None,
        torch_dtype=torch.float16,
        fuse_layers=False,
        trust_remote_code=True,
        enable_router=False,
        router_config=None,
        enable_on_demand=False,
    ):
        super().__init__()
        self._enable_on_demand_hint = enable_on_demand
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("=" * 80)
        self.logger.info("Initialize QueryAdaptiveQuantizationForCausalLM")
        self.logger.info("=" * 80)
        self.logger.debug(f"Model path: {model_path}")
        self.logger.debug(f"Configuration type: {type(config)}")
        self.logger.debug(f"torch data type: {torch_dtype}")
        self.logger.debug(f"Whether to fuse layers: {fuse_layers}")
        self.logger.debug(f"Trust remote code: {trust_remote_code}")
        self.config = config
        self.logger.debug(f"anyprec parameter in configuration: {config.anyprec}")
        self.supported_bits = list(
            range(
                self.config.anyprec["seed_precision"],
                self.config.anyprec["parent_precision"] + 1,
            )
        )
        self.logger.info(f"Supported quantization bit range: {self.supported_bits}")
        if precisions is None:
            self.precisions = self.supported_bits
            self.logger.debug("Use default precision settings: all supported digits")
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(
                (bit in self.supported_bits for bit in precisions)
            ), f"Supported bits {precisions} must be a subset of model supported bits {self.supported_bits}"
            self.precisions = precisions
            self.logger.debug(f"Use custom precision settings: {precisions}")
        if enable_router and router_config:
            router_precision_levels = router_config.get(
                "precision_levels", [3, 4, 5, 6, 7, 8]
            )
            original_precisions = set(self.precisions)
            router_precisions = set(router_precision_levels)
            valid_router_precisions = router_precisions & set(self.supported_bits)
            self.precisions = sorted(
                list(original_precisions | valid_router_precisions)
            )
            self.logger.info(
                f" Router mode: Extended precision settings from {list(original_precisions)} arrive {self.precisions}"
            )
            self.logger.info(
                f"   The precision required by Router: {sorted(list(router_precisions))}"
            )
            self.logger.info(
                f"   Effective Router precision: {sorted(list(valid_router_precisions))}"
            )
            if router_precisions - valid_router_precisions:
                missing_precisions = router_precisions - valid_router_precisions
                self.logger.warning(
                    f" Some precisions in Router configuration are beyond the model support range and have been ignored: {sorted(list(missing_precisions))}"
                )
        self.precision = max(self.precisions)
        self.logger.info(f"Current maximum precision: {self.precision}")
        self.logger.debug("Start initializing the empty weight model...")
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.logger.info(
            f"Model initialization is completed, model type: {type(self.model)}"
        )
        self.logger.debug("Get model analyzer...")
        self.analyzer = get_analyzer(self.model)
        self.logger.info(f"Analyzer type: {type(self.analyzer)}")
        self.layer_to_aplinears = []
        self.ap_linears = []
        self.logger.debug("Initialize the quantization layer management list")
        self.logger.info("Start loading and replacing the quantization module...")
        self._load_quantized_modules()
        self.logger.debug("Bind model weights...")
        self.tie_weights()
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = "cpu"
        self.logger.info(f"Use device mapping strategy: {device_map}")
        self.logger.info("Start loading checkpoints and distributing to devices...")
        try:
            load_checkpoint_and_dispatch(
                self.model,
                checkpoint=model_path,
                device_map=device_map,
                no_split_module_classes=[self.layer_type],
                dtype=torch_dtype,
                offload_folder=None,
                offload_state_dict=False,
            )
        except Exception as e:
            self.logger.error(f"Loading using accelerate failed: {e}")
            self.logger.info("Falling back to traditional loading methods...")
            from safetensors import safe_open
            import os

            safetensors_files = []
            pytorch_files = []
            for file in os.listdir(model_path):
                if file.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(model_path, file))
                elif file.endswith(".bin") or file == "pytorch_model.bin":
                    pytorch_files.append(os.path.join(model_path, file))
            if safetensors_files:
                self.logger.info(f"Find the safetensors file: {safetensors_files}")
                self.logger.info(
                    "Use the to_empty method to move the model to avoid meta tensor errors..."
                )
                self.model = self.model.to_empty(device=target_device)
                for safetensors_file in safetensors_files:
                    with safe_open(
                        safetensors_file, framework="pt", device=target_device
                    ) as f:
                        state_dict = {}
                        for key in f.keys():
                            if key in self.model.state_dict():
                                tensor = f.get_tensor(key)
                                tensor = tensor.to(
                                    device=target_device, dtype=torch_dtype
                                )
                                state_dict[key] = tensor
                        self.logger.info(f"load {len(state_dict)} a weight...")
                        self.model.load_state_dict(state_dict, strict=False)
                self.logger.info("safetensors weight loading completed")
            elif pytorch_files:
                self.logger.info(f"Find the pytorch model file: {pytorch_files}")
                self.logger.info(
                    "Use the to_empty method to move the model to avoid meta tensor errors..."
                )
                self.model = self.model.to_empty(device=target_device)
                for pytorch_file in pytorch_files:
                    self.logger.info(f"from {pytorch_file} Loading weights...")
                    state_dict = torch.load(
                        pytorch_file, map_location=target_device, weights_only=False
                    )
                    filtered_state_dict = {}
                    for key, tensor in state_dict.items():
                        if key in self.model.state_dict():
                            tensor = tensor.to(device=target_device, dtype=torch_dtype)
                            filtered_state_dict[key] = tensor
                    self.logger.info(f"load {len(filtered_state_dict)} a weight...")
                    self.model.load_state_dict(filtered_state_dict, strict=False)
                self.logger.info("pytorch weight loading completed")
            else:
                raise Exception(
                    f"Weights file not found (safetensors or pytorch format): {model_path}"
                )
        self.logger.info("Checkpoint loading and device distribution completed")
        if fuse_layers:
            self.logger.info("Start blending layers...")
            self.fuse_layers()
        else:
            self.logger.debug("Skip layer fusion")
        if enable_on_demand:
            self.logger.info(
                "On-demand mode enabled, skips precision pruning to preserve the LUT at all precisions"
            )
        else:
            self.logger.info("Start precision pruning...")
            self.prune_precisions()
        self.enable_router = enable_router
        self.router = None
        if enable_router:
            self._init_router(router_config)
            if router_config:
                router_precision_levels = router_config.get(
                    "precision_levels", [3, 4, 5, 6, 7, 8]
                )
                valid_router_precisions = [
                    p for p in router_precision_levels if p in self.supported_bits
                ]
                will_enable_on_demand = (
                    hasattr(self, "_enable_on_demand_hint")
                    and self._enable_on_demand_hint
                )
                if will_enable_on_demand:
                    protected_precisions = [3, 4, 5, 6, 7, 8]
                    self.logger.info(
                        f" On-demand mode: protect required precisions {protected_precisions}, other precisions are loaded on demand"
                    )
                else:
                    protected_precisions = valid_router_precisions
                    self.logger.info(
                        f" Traditional mode: protect all Router precisions {protected_precisions}"
                    )
                for ap_linear in self.ap_linears:
                    ap_linear.set_router_protected_precisions(protected_precisions)
        self.logger.info(
            "QueryAdaptiveQuantizationForCausalLM initialization completed"
        )
        self.logger.info("=" * 80)

    def _init_router(self, router_config):
        self.logger.info("Initializing the threshold-map Router...")
        online_config = {
            "precision_levels": self.precisions,
            "router_type": "online_ttt",
            "router_hidden_size": 256,
            "temperature": 1.0,
            "use_gumbel_softmax": True,
            "dropout_rate": 0.1,
            "pooling_method": "mean",
            "inference_sampling_prob": 0.0,
            "inference_low_bit_bias": 0.0,
            "router_stateful": False,
            "router_state_detach": True,
            "router_update_state": True,
            "concept_dim": 352,
            "logit_dim": 64,
            "ttt_lr_linear": 0.1,
            "ttt_lr_ln": 0.1,
            "router_ttt_layers": 2,
            "router_ttt_heads": 1,
            "router_ttt_residual": "none",
            "router_input_source": "prelude_hidden",
            "router_prelude_layers": 2,
            "router_prelude_precision": 5,
            "router_segment_count": 8,
            "router_readout": "segment_attention",
            "router_layer_decoder": "layer_gru",
            "router_utility_num_groups": 0,
            "router_enable_utility_scores": False,
            "router_output_granularity": "block",
        }
        online_config.update(router_config or {})
        router_type = str(
            online_config.pop("router_type", "online_ttt")
        ).lower()
        if router_type != "online_ttt":
            raise ValueError(
                f"Unsupported router_type={router_type!r}; the active VAMP "
                "workflow only supports 'online_ttt'. Historical Router "
                "architectures are not supported by this package."
            )
        self.router_input_source = str(
            online_config.get("router_input_source", "embedding")
        ).lower()
        self.router_prelude_layers = max(
            0, int(online_config.get("router_prelude_layers", 0))
        )
        self.router_prelude_precision = int(
            online_config.get("router_prelude_precision", 5)
        )
        hidden_size = self.config.hidden_size
        num_layers = len(self.layer_to_aplinears)
        self.logger.info(
            f"Router configuration: hidden layer size ={hidden_size}, number of layers ={num_layers}"
        )
        self.logger.info(f"Precision levels: {online_config['precision_levels']}")
        self.logger.info("Router type: online_ttt")
        self.router = self.router_class(
            hidden_size=hidden_size,
            num_layers=num_layers,
            **online_config,
        )
        device = next(self.model.parameters()).device
        self.router = self.router.to(device)
        self.logger.info(f"Router moved to device: {device}")
        self.logger.info("Threshold-map Router initialization completed")

    def _get_model_input_device(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return next(self.model.model.embed_tokens.parameters()).device
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "decoder")
            and hasattr(self.model.model.decoder, "embed_tokens")
        ):
            return next(self.model.model.decoder.embed_tokens.parameters()).device
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return next(self.model.transformer.wte.parameters()).device
        if hasattr(self.model, "embeddings"):
            return next(self.model.embeddings.parameters()).device
        return next(self.model.parameters()).device

    def _embed_router_input_ids(self, input_ids):
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens(input_ids)
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "decoder")
            and hasattr(self.model.model.decoder, "embed_tokens")
        ):
            return self.model.model.decoder.embed_tokens(input_ids)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte(input_ids)
        if hasattr(self.model, "embeddings"):
            return self.model.embeddings(input_ids)
        raise ValueError("Unable to identify the embedding layer of the model")

    def get_router_input_hidden_states(
        self,
        input_ids,
        attention_mask=None,
        detach: bool = True,
        feature_extraction=None,
    ):
        model_device = self._get_model_input_device()
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        use_prelude = (
            self.router_input_source == "prelude_hidden"
            and self.router_prelude_layers > 0
        )
        if not use_prelude:
            hidden_states = self._embed_router_input_ids(input_ids)
            return hidden_states.detach() if detach else hidden_states

        previous_precisions = self._snapshot_precisions()
        was_training = self.model.training
        feature_extraction = feature_extraction or getattr(
            self, "router_feature_extraction", "full_forward"
        )
        if feature_extraction not in {"full_forward", "early_exit"}:
            raise ValueError(f"Unknown Router feature extraction: {feature_extraction}")
        capture_handle = None
        captured = []

        class PreludeComplete(Exception):
            pass

        def capture_prelude(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)
            raise PreludeComplete

        try:
            self._set_precision_silent(self.router_prelude_precision)
            self.model.eval()
            with torch.no_grad():
                model_inputs = {
                    "input_ids": input_ids,
                    "use_cache": False,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                if feature_extraction == "early_exit":
                    layers = self.analyzer.get_layers()
                    if not 0 < self.router_prelude_layers < len(layers):
                        raise ValueError("Early-exit Prelude must end before the final layer")
                    capture_handle = layers[self.router_prelude_layers - 1].register_forward_hook(
                        capture_prelude
                    )
                    model_inputs["output_hidden_states"] = False
                    try:
                        self.model(**model_inputs)
                    except PreludeComplete:
                        pass
                    if not captured:
                        raise RuntimeError("Prelude layer hook was not reached")
                    return captured[0].detach() if detach else captured[0]
                outputs = self.model(**model_inputs)
                hidden_tuple = getattr(outputs, "hidden_states", None)
                if hidden_tuple is None:
                    self.logger.warning(
                        "Prelude hidden extraction did not return hidden_states; "
                        "falling back to input embeddings."
                    )
                    hidden_states = self._embed_router_input_ids(input_ids)
                else:
                    hidden_idx = min(self.router_prelude_layers, len(hidden_tuple) - 1)
                    hidden_states = hidden_tuple[hidden_idx]
        finally:
            if capture_handle is not None:
                capture_handle.remove()
            self._restore_precisions(previous_precisions)
            if was_training:
                self.model.train()
        return hidden_states.detach() if detach else hidden_states

    def _get_router_precision(self, *args, **kwargs):
        if not self.enable_router or self.router is None:
            return None
        if args:
            input_ids = args[0]
        else:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("Router inference requires input_ids")
        attention_mask = kwargs.get("attention_mask")
        model_device = self._get_model_input_device()
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        hidden_states = self.get_router_input_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            detach=not self.training,
        )
        if self.training:
            router_output = self.router(hidden_states, attention_mask)
        else:
            with torch.no_grad():
                router_output = self.router(hidden_states, attention_mask)
        values = router_output["precision_values"]
        # Training retains its existing semantics; shared maps are inference-only.
        selected = values[0] if self.training else shared_precision_map(
            values, kwargs.get("batch_map_policy")
        )
        precision_schedule = self.router.precision_values_to_schedule(selected)
        self.logger.debug(
            "Router output precision schedule: %s",
            precision_schedule,
        )
        return precision_schedule

    def _snapshot_precisions(self):
        return [
            [ap_linear.precision for ap_linear in layer]
            for layer in self.layer_to_aplinears
        ]

    def _current_module_precision_map(self):
        names = self.config.anyprec["arch_config"]["module_names"]
        return {
            layer_idx: {name: int(module.precision) for name, module in zip(names, modules)}
            for layer_idx, modules in enumerate(self.layer_to_aplinears)
        }

    def _restore_precisions(self, precision_list):
        if precision_list and isinstance(precision_list[0], (list, tuple)):
            if len(precision_list) != len(self.layer_to_aplinears):
                raise ValueError(
                    "Precision snapshot layer count does not match the model"
                )
            max_precision = min(self.precisions)
            for layer, layer_precisions in zip(
                self.layer_to_aplinears, precision_list
            ):
                if len(layer_precisions) != len(layer):
                    raise ValueError(
                        "Precision snapshot module count does not match the model"
                    )
                for ap_linear, precision in zip(layer, layer_precisions):
                    ap_linear.set_precision(precision)
                    max_precision = max(max_precision, int(precision))
            self.precision = max_precision
            return
        self._set_precision_silent(precision_list)

    def _debug_count_gpu_luts(self):
        lut_count = {}
        if hasattr(self, "ap_linears") and self.ap_linears:
            for i, ap_linear in enumerate(self.ap_linears):
                for bit in [3, 4, 5, 6, 7, 8]:
                    lut_name = f"lut{bit}"
                    if hasattr(ap_linear, lut_name):
                        lut = getattr(ap_linear, lut_name)
                        if lut is not None and lut.device.type == "cuda":
                            lut_count[bit] = lut_count.get(bit, 0) + 1
        return lut_count

    def _set_precision_silent(self, precision):
        if isinstance(precision, int):
            for apl in self.ap_linears:
                apl.set_precision(precision)
            self.precision = precision
        elif isinstance(precision, (list, tuple)):
            assert len(precision) == len(
                self.layer_to_aplinears
            ), f"precision list length {len(precision)} Must be equal to the number of layers {len(self.layer_to_aplinears)}"
            for layer_idx, bit in enumerate(precision):
                for apl in self.layer_to_aplinears[layer_idx]:
                    apl.set_precision(bit)
            self.precision = max(precision)
        elif isinstance(precision, dict):
            if any((isinstance(v, dict) for v in precision.values())):
                self._set_blockwise_precision_silent(precision)
            elif any((isinstance(k, tuple) and len(k) == 2 for k in precision.keys())):
                self._set_module_precision_silent(precision)
            else:
                for layer_idx, bit in precision.items():
                    for apl in self.layer_to_aplinears[layer_idx]:
                        apl.set_precision(bit)
                self.precision = max(self.precision, max(precision.values()))
        else:
            raise TypeError(
                "precision must be int | list/tuple[int] | dict[int,int] | dict[int,dict] | dict[tuple,int]"
            )

    def _set_blockwise_precision_silent(self, precision_map):
        module_names = self.config.anyprec["arch_config"]["module_names"]
        mha_modules = []
        mlp_modules = []
        for module_name in module_names:
            if "self_attn" in module_name or "attn" in module_name:
                mha_modules.append(module_name)
            elif "mlp" in module_name or "fc" in module_name:
                mlp_modules.append(module_name)
        max_precision = self.precision
        for layer_idx, block_precision in precision_map.items():
            layer_aplinears = self.layer_to_aplinears[layer_idx]
            for apl_idx, apl in enumerate(layer_aplinears):
                module_name = module_names[apl_idx]
                if module_name in mha_modules:
                    if "mha" in block_precision:
                        target_precision = block_precision["mha"]
                        apl.set_precision(target_precision)
                        max_precision = max(max_precision, target_precision)
                elif module_name in mlp_modules:
                    if "mlp" in block_precision:
                        target_precision = block_precision["mlp"]
                        apl.set_precision(target_precision)
                        max_precision = max(max_precision, target_precision)
        self.precision = max_precision

    def _set_module_precision_silent(self, precision_map):
        module_names = self.config.anyprec["arch_config"]["module_names"]
        max_precision = self.precision
        for (layer_idx, module_name), bit in precision_map.items():
            if module_name in module_names:
                module_idx = module_names.index(module_name)
                if layer_idx < len(self.layer_to_aplinears) and module_idx < len(
                    self.layer_to_aplinears[layer_idx]
                ):
                    apl = self.layer_to_aplinears[layer_idx][module_idx]
                    apl.set_precision(bit)
                    max_precision = max(max_precision, bit)
        self.precision = max_precision

    def forward(self, *args, **kwargs):
        self.logger.debug("=" * 50)
        self.logger.debug("Start forward propagation")
        self.logger.debug(
            f"Number of input parameters: args={len(args)}, kwargs={len(kwargs)}"
        )
        self.logger.debug(f"kwargs keys: {list(kwargs.keys())}")
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            self.logger.debug(
                f" [DEBUG] GPU memory when Forward starts: {initial_memory:.3f}GB"
            )
        prev_state = self._snapshot_precisions()
        precision_changed = False
        use_router = kwargs.pop("use_router", False) and self.enable_router
        batch_map_policy = kwargs.pop("batch_map_policy", None)
        if self.on_demand_enabled and (
            "precision" in kwargs or "precision_map" in kwargs
        ):
            try:
                self._apply_on_demand_precision_strategy(*args, **kwargs)
                self.on_demand_stats["total_inferences"] += 1
            except Exception as e:
                self.logger.warning(
                    f"On-demand precision policy application failed: {e}"
                )
        if self.enable_router and self.router is not None and self.training:
            use_router = True
            self.logger.debug("Router training mode: force Router call to be enabled")
        try:
            if "precision" in kwargs:
                precision = kwargs.pop("precision")
                self.logger.info(f"Precision parameter detected: {precision}")
                self.set_precision(precision)
                precision_changed = True
            if "precision_map" in kwargs:
                precision_map = kwargs.pop("precision_map")
                self.logger.info(f"Precision_map parameter detected: {precision_map}")
                self.set_precision(precision_map)
                precision_changed = True
            router_precision_map = None
            if use_router and (not precision_changed):
                self.logger.info("Using dynamic routers for precision selection...")
                router_precision_map = self._get_router_precision(
                    *args, batch_map_policy=batch_map_policy, **kwargs
                )
                if router_precision_map:
                    if self.on_demand_enabled:
                        self.logger.debug(
                            "[DEBUG] On-demand mode detected, apply on-demand weight loading first..."
                        )
                        if self.use_pipeline_scheduling and self.pipeline_config:
                            tasks = self._build_pipeline_tasks(
                                router_precision_map,
                                tokenwise_frequency=self.pipeline_config.get(
                                    "tokenwise_frequency", False
                                ),
                            )
                            pipeline_result = self.on_demand_manager.apply_precision_strategy_pipeline(
                                tasks, self.pipeline_config
                            )
                            self.on_demand_stats.setdefault(
                                "pipeline_results", []
                            ).append(pipeline_result)
                            self.on_demand_stats["total_inferences"] += 1
                            self.set_precision(router_precision_map)
                        else:
                            raise RuntimeError("Configure bounded HBF before on-demand inference")
                    else:
                        self.set_precision(router_precision_map)
                    precision_changed = True
                    self.logger.info(
                        f"Router precision map: {router_precision_map}"
                    )
                else:
                    self.logger.warning(
                        "Router precision selection failed, using current precision settings"
                    )
            elif self.on_demand_enabled and (not precision_changed):
                self.logger.debug(
                    "On-demand mode (no Router) detected, default precision policy applied..."
                )
                current_precisions = self._snapshot_precisions()
                self._apply_on_demand_precision_strategy(
                    precision_map=self._current_module_precision_map()
                )
                self._restore_precisions(current_precisions)
                self.on_demand_stats["total_inferences"] += 1
            if not precision_changed:
                self.logger.debug(
                    "Precision parameter not detected, use current precision setting"
                )
            self.logger.debug("Call model forward propagation...")
            results = self.model.forward(*args, **kwargs)
            self.logger.debug(
                f"Forward propagation is completed, output type: {type(results)}"
            )
            return results
        finally:
            self.logger.debug("Restore precision settings...")
            self._restore_precisions(prev_state)
            self.logger.debug("End of forward propagation")
            self.logger.debug("=" * 50)

    def generate(self, *args, **kwargs):
        self.logger.debug("=" * 50)
        self.logger.debug("Start generating sequence")
        self.logger.debug(
            f"Number of input parameters: args={len(args)}, kwargs={len(kwargs)}"
        )
        self.logger.debug(f"kwargs keys: {list(kwargs.keys())}")
        prev_state = self._snapshot_precisions()
        precision_changed = False
        use_router = kwargs.pop("use_router", False) and self.enable_router
        batch_map_policy = kwargs.pop("batch_map_policy", None)
        try:
            if "precision" in kwargs:
                precision = kwargs.pop("precision")
                self.logger.info(f"Precision parameter detected: {precision}")
                self.set_precision(precision)
                precision_changed = True
            if "precision_map" in kwargs:
                precision_map = kwargs.pop("precision_map")
                self.logger.info(f"Precision_map parameter detected: {precision_map}")
                self.set_precision(precision_map)
                precision_changed = True
            if use_router and (not precision_changed):
                self.logger.info("Using dynamic routers for precision selection...")
                router_precision_map = self._get_router_precision(
                    *args, batch_map_policy=batch_map_policy, **kwargs
                )
                if router_precision_map:
                    self.set_precision(router_precision_map)
                    precision_changed = True
                    self.logger.info(
                        f"Router precision map: {router_precision_map}"
                    )
            if not precision_changed:
                self.logger.debug(
                    "Precision parameter not detected, use current precision setting"
                )
            if self.on_demand_enabled:
                # Loading must respect the resolved per-module map, never route again.
                current_precisions = self._snapshot_precisions()
                resolved_map = self._current_module_precision_map()
                self._apply_on_demand_precision_strategy(precision_map=resolved_map)
                self._restore_precisions(current_precisions)
                self.on_demand_stats["total_inferences"] += 1
            self.logger.debug("Enter inference mode and start generating...")
            with torch.inference_mode():
                results = self.model.generate(*args, **kwargs)
            self.logger.debug(
                f"The generation is completed and the output shape is: {(results.shape if hasattr(results, 'shape') else 'N/A')}"
            )
            return results
        finally:
            if self.on_demand_enabled:
                collect_mem = bool(
                    getattr(self, "collect_on_demand_memory_summary", False)
                )
                try:
                    if isinstance(getattr(self, "pipeline_config", None), dict):
                        collect_mem = collect_mem or bool(
                            self.pipeline_config.get(
                                "collect_on_demand_memory_summary", False
                            )
                        )
                except Exception:
                    pass
                if collect_mem:
                    memory_summary = self.on_demand_manager.get_memory_summary()
                    if memory_summary and "gpu_memory_saved_mb" in memory_summary:
                        self.on_demand_stats["memory_savings"].append(memory_summary)
                self.logger.debug("Clean on-demand weights...")
                self._cleanup_on_demand_weights()
            self.logger.debug("Restore precision settings...")
            self._restore_precisions(prev_state)
            self.logger.debug("End of sequence generation")
            self.logger.debug("=" * 50)

    @staticmethod
    def _load_config(model_path, trust_remote_code=True):
        logger = logging.getLogger(__name__)
        logger.debug(f"Load configuration from path: {model_path}")
        logger.debug(f"Trust remote code: {trust_remote_code}")
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        logger.debug(f"Configuration loading completed, type: {type(config)}")
        return config

    @classmethod
    def from_quantized(
        cls,
        quant_model_path,
        trust_remote_code=True,
        fuse_layers=False,
        precisions=None,
        enable_router=False,
        router_config=None,
        enable_on_demand=False,
        on_demand_memory_limit_gb=4.0,
    ):
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("Create an instance from a quantized model")
        logger.info("=" * 80)
        logger.debug(f"Packed model path: {quant_model_path}")
        logger.debug(f"Trust remote code: {trust_remote_code}")
        logger.debug(f"Fusion layer: {fuse_layers}")
        logger.debug(f"Specify precision: {precisions}")
        logger.debug(f"Enable the router: {enable_router}")
        logger.debug(f"Router configuration: {router_config}")
        logger.debug(f"Enable on-demand inference: {enable_on_demand}")
        logger.debug(
            f"On-demand inference memory limits: {on_demand_memory_limit_gb}GB"
        )
        config = cls._load_config(quant_model_path, trust_remote_code)
        ap_model = cls(
            model_path=quant_model_path,
            precisions=precisions,
            config=config,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
            enable_router=enable_router,
            router_config=router_config,
            enable_on_demand=enable_on_demand,
        )
        if enable_on_demand:
            logger.info("Initialize on-demand inference functionality...")
            ap_model.enable_on_demand_inference(on_demand_memory_limit_gb)
        logger.info("Creating an instance from a quantized model is complete")
        return ap_model

    def _load_quantized_modules(self):
        self.logger.info("Start loading the quantization module...")
        layers = self.analyzer.get_layers()
        self.logger.info(f"Get {len(layers)} model layer")
        for layer_idx, layer in enumerate(tqdm(layers, desc="Loading AP Layers")):
            self.logger.debug(
                f"processing section {layer_idx} layer, type: {type(layer)}"
            )
            self.layer_to_aplinears.append([])
            self.logger.debug(
                f"for the first {layer_idx} Layer initialization AP linear layer list"
            )
            named_linears = self.analyzer.get_modules(layer)
            self.logger.debug(
                f"No. {layer_idx} found in layer {len(named_linears)} linear modules: {list(named_linears.keys())}"
            )
            for name, module in named_linears.items():
                self.logger.debug(
                    f"Replace module {name}, input features: {module.in_features}, output features: {module.out_features}"
                )
                self.logger.debug(
                    f"The original module has an offset: {module.bias is not None}"
                )
                self.logger.debug(f"Original module equipment: {module.weight.device}")
                wqlinear = QueryAdaptiveQuantizationLinear(
                    module.in_features,
                    module.out_features,
                    self.supported_bits,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    device=module.weight.device,
                    dtype=torch.float16,
                )
                self.logger.debug(
                    f"Create a new quantized linear layer: {type(wqlinear)}"
                )
                self.ap_linears.append(wqlinear)
                self.layer_to_aplinears[layer_idx].append(wqlinear)
                self.logger.debug(
                    f"Current total number of AP linear layers: {len(self.ap_linears)}"
                )
                replace_module_by_name(layer, name, wqlinear)
                self.logger.debug(f"module {name} Replacement completed")
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.debug(
                f"No. {layer_idx} Layer processing completed, memory cleared"
            )
        self.logger.info(
            f"The quantization module is loaded and a total of {len(self.layer_to_aplinears)} layer,{len(self.ap_linears)} linear layers"
        )

    def prune_precisions(self):
        self.logger.info("Start precision pruning...")
        self.logger.debug(
            f"to {len(self.ap_linears)} AP linear layer for precision pruning"
        )
        for i, ap_linear in enumerate(self.ap_linears):
            self.logger.debug(f"pruning {i} AP linear layer")
            ap_linear.prune_precisions()
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info(
            "Precision pruning is completed and the memory has been cleared"
        )

    def set_precision(self, precision):
        self.logger.debug(f"Set precision: {precision}, type: {type(precision)}")
        if isinstance(precision, int):
            self.logger.info(f"Set global unified precision: {precision}")
            for i, apl in enumerate(self.ap_linears):
                self.logger.debug(
                    f"Setup {i} The precision of the AP linear layer is {precision}"
                )
                apl.set_precision(precision)
            self.precision = precision
            self.logger.debug(
                f"Global precision setting is completed, current precision: {self.precision}"
            )
        elif isinstance(precision, (list, tuple)):
            if (
                len(precision) != len(self.layer_to_aplinears)
                and len(precision) > 0
                and all((isinstance(x, dict) for x in precision))
            ):
                layer_bits = defaultdict(list)
                for entry in precision:
                    flat = self._convert_router_precision_strategy(entry)
                    for layer_idx, bit in flat.items():
                        if isinstance(layer_idx, int) and isinstance(bit, int):
                            layer_bits[int(layer_idx)].append(int(bit))
                collapsed = {}
                for layer_idx, bits in layer_bits.items():
                    if not bits:
                        continue
                    counts = defaultdict(int)
                    for b in bits:
                        counts[int(b)] += 1
                    max_count = max(counts.values())
                    candidates = [b for b, c in counts.items() if c == max_count]
                    collapsed[layer_idx] = max(candidates)
                self.logger.info(
                    f"Detected tokenwise precision list(len={len(precision)}), collapsed into a sparse precision map (len={len(collapsed)})"
                )
                return self.set_precision(collapsed)
            self.logger.info(f"Set layer-by-layer precision: {precision}")
            assert len(precision) == len(
                self.layer_to_aplinears
            ), f"precision list length {len(precision)} Must be equal to the number of layers {len(self.layer_to_aplinears)}"
            for layer_idx, bit in enumerate(precision):
                self.logger.debug(f"Setup {layer_idx} The layer precision is {bit}")
                for apl_idx, apl in enumerate(self.layer_to_aplinears[layer_idx]):
                    self.logger.debug(
                        f"  - Setup {layer_idx} level {apl_idx} The precision of the AP linear layer is {bit}"
                    )
                    apl.set_precision(bit)
            self.precision = max(precision)
            self.logger.info(
                f"Layer-by-layer precision setting is completed, maximum precision: {self.precision}"
            )
        elif isinstance(precision, dict):
            if any((isinstance(v, dict) for v in precision.values())):
                self.logger.info(f"Set block-level precision mapping: {precision}")
                self._set_blockwise_precision(precision)
            elif any((isinstance(k, tuple) and len(k) == 2 for k in precision.keys())):
                self.logger.info(f"Set module level precision mapping: {precision}")
                self._set_module_precision(precision)
            else:
                self.logger.info(f"Set up sparse precision mapping: {precision}")
                for layer_idx, bit in precision.items():
                    self.logger.debug(f"Setup {layer_idx} The layer precision is {bit}")
                    for apl_idx, apl in enumerate(self.layer_to_aplinears[layer_idx]):
                        self.logger.debug(
                            f"  - Setup {layer_idx} level {apl_idx} The precision of the AP linear layer is {bit}"
                        )
                        apl.set_precision(bit)
                self.precision = max(self.precision, max(precision.values()))
                self.logger.info(
                    f"The sparse precision mapping setting is completed, the current maximum precision is: {self.precision}"
                )
        else:
            self.logger.error(f"Unsupported precision types: {type(precision)}")
            raise TypeError(
                "precision must be int | list/tuple[int] | dict[int,int] | dict[int,dict] | dict[tuple,int]"
            )

    def _set_blockwise_precision(self, precision_map):
        module_names = self.config.anyprec["arch_config"]["module_names"]
        mha_modules = []
        mlp_modules = []
        for module_name in module_names:
            if "self_attn" in module_name or "attn" in module_name:
                mha_modules.append(module_name)
            elif "mlp" in module_name or "fc" in module_name:
                mlp_modules.append(module_name)
        self.logger.debug(f"MHA module: {mha_modules}")
        self.logger.debug(f"MLP module: {mlp_modules}")
        max_precision = self.precision
        for layer_idx, block_precision in precision_map.items():
            self.logger.debug(
                f"Setup {layer_idx} Block level precision: {block_precision}"
            )
            layer_aplinears = self.layer_to_aplinears[layer_idx]
            for apl_idx, apl in enumerate(layer_aplinears):
                module_name = module_names[apl_idx]
                if module_name in mha_modules:
                    if "mha" in block_precision:
                        target_precision = block_precision["mha"]
                        self.logger.debug(
                            f"  - Set up MHA module {module_name} The precision is {target_precision}"
                        )
                        apl.set_precision(target_precision)
                        max_precision = max(max_precision, target_precision)
                elif module_name in mlp_modules:
                    if "mlp" in block_precision:
                        target_precision = block_precision["mlp"]
                        self.logger.debug(
                            f"  - Set up MLP module {module_name} The precision is {target_precision}"
                        )
                        apl.set_precision(target_precision)
                        max_precision = max(max_precision, target_precision)
                else:
                    self.logger.warning(f"Unknown module type: {module_name}")
        self.precision = max_precision
        self.logger.info(
            f"Block level precision setting is completed, current maximum precision: {self.precision}"
        )

    def _set_module_precision(self, precision_map):
        module_names = self.config.anyprec["arch_config"]["module_names"]
        max_precision = self.precision
        for (layer_idx, module_name), bit in precision_map.items():
            self.logger.debug(
                f"Setup {layer_idx} layer {module_name} The module precision is {bit}"
            )
            if module_name in module_names:
                module_idx = module_names.index(module_name)
                if layer_idx < len(self.layer_to_aplinears) and module_idx < len(
                    self.layer_to_aplinears[layer_idx]
                ):
                    apl = self.layer_to_aplinears[layer_idx][module_idx]
                    apl.set_precision(bit)
                    max_precision = max(max_precision, bit)
                    self.logger.debug(
                        f"  -Successfully set up {layer_idx} layer {module_name} The precision is {bit}"
                    )
                else:
                    self.logger.warning(
                        f"layer index {layer_idx} or module index {module_idx} out of range"
                    )
            else:
                self.logger.warning(f"Module not found: {module_name}")
        self.precision = max_precision
        self.logger.info(
            f"Module level precision setting is completed, current maximum precision: {self.precision}"
        )

    def get_precision_info(self):
        module_names = self.config.anyprec["arch_config"]["module_names"]
        precision_info = {}
        for layer_idx, layer_aplinears in enumerate(self.layer_to_aplinears):
            layer_info = {}
            for module_idx, apl in enumerate(layer_aplinears):
                module_name = module_names[module_idx]
                layer_info[module_name] = apl.precision
            precision_info[layer_idx] = layer_info
        return precision_info

    def print_precision_summary(self):
        precision_info = self.get_precision_info()
        self.logger.info("=" * 60)
        self.logger.info("Summary of current precision settings")
        self.logger.info("=" * 60)
        for layer_idx, layer_info in precision_info.items():
            self.logger.info(f"No. {layer_idx} layer:")
            mha_modules = {k: v for k, v in layer_info.items() if "attn" in k}
            mlp_modules = {
                k: v for k, v in layer_info.items() if "mlp" in k or "fc" in k
            }
            if mha_modules:
                self.logger.info(f"  MHA: {mha_modules}")
            if mlp_modules:
                self.logger.info(f"  MLP: {mlp_modules}")
        self.logger.info("=" * 60)

    def tie_weights(self):
        self.logger.debug("Start binding model weights...")
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
            self.logger.debug("Weight binding completed")
        else:
            self.logger.debug("The model does not support weight binding")

    def get_model_layers(self):
        self.logger.debug("Get model layer...")
        module = self.model
        model_path = self.config.anyprec["arch_config"]["model_name"].split(".")
        self.logger.debug(f"Model path: {model_path}")
        for attrib_name in model_path:
            self.logger.debug(f"Access properties: {attrib_name}")
            module = getattr(module, attrib_name)
            self.logger.debug(f"Current module type: {type(module)}")
        layers_name = self.config.anyprec["arch_config"]["layers_name"]
        self.logger.debug(f"Layer name: {layers_name}")
        layers = getattr(module, layers_name)
        self.logger.debug(f"Get the number of layers: {len(layers)}")
        return layers

    def fuse_layers(self):
        self.logger.debug("Try blending layers...")
        if "fuse_target_layers" not in self.model_config:
            self.logger.warning("Model does not support layer fusion")
            raise NotImplementedError("This model does not support layer fusion")
        self.logger.debug("Layer fusion function to be implemented")
        pass

    @property
    def layer_type(self):
        self.logger.debug("Determine layer type...")
        for i, layer in enumerate(self.get_model_layers()):
            layer_class_name = layer.__class__.__name__
            self.logger.debug(f"No. {i} Layer class name: {layer_class_name}")
            if layer_class_name.endswith("DecoderLayer"):
                self.logger.debug(f"Determine the layer type as: {layer_class_name}")
                return layer_class_name
        self.logger.warning("DecoderLayer type not found")
        return None

    @property
    def device(self):
        device = self.model.device
        self.logger.debug(f"Model equipment: {device}")
        return device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.logger.debug("Enable gradient checkpointing...")
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            self.logger.debug("Gradient checkpointing is enabled")
        else:
            self.logger.warning(
                "The underlying model does not support gradient checkpoints"
            )

    def gradient_checkpointing_disable(self):
        self.logger.debug("Disable gradient checkpointing...")
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
            self.logger.debug("Gradient checkpoints disabled")
        else:
            self.logger.warning(
                "The underlying model does not support gradient checkpoints"
            )

    @property
    def supports_gradient_checkpointing(self):
        return (
            hasattr(self.model, "supports_gradient_checkpointing")
            and self.model.supports_gradient_checkpointing
        )
