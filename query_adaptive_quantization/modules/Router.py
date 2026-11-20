"""Prompt-level precision-map Router used by the quality-threshold workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineLearningPrecisionRouter(nn.Module):
    """Canonical prompt-level Online TTT precision Router.

    The layer count follows the base model. Both token-local fast-state
    learning rates share one configurable step size. Parameter names retain
    compatibility with existing Router checkpoints.
    """

    STATE_KEYS = (
        "weight_linear_tmp",
        "weight_ln_tmp",
        "bias_linear_tmp",
        "bias_ln_tmp",
    )
    BLOCK_NAMES = ("mha", "mlp")
    INFERENCE_TOKEN_CHUNK_SIZE = 128

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 32,
        precision_levels: Optional[List[int]] = None,
        router_hidden_size: int = 256,
        temperature: float = 1.0,
        use_gumbel_softmax: bool = True,
        dropout_rate: float = 0.1,
        pooling_method: str = "mean",
        inference_sampling_prob: float = 0.0,
        inference_low_bit_bias: float = 0.0,
        concept_dim: int = 352,
        logit_dim: int = 64,
        ttt_lr_linear: float = 0.1,
        ttt_lr_ln: float = 0.1,
        router_ttt_layers: int = 2,
        router_ttt_heads: int = 1,
        router_ttt_residual: str = "none",
        initializer_range: float = 0.02,
        router_input_source: str = "prelude_hidden",
        router_prelude_layers: int = 2,
        router_prelude_precision: int = 5,
        router_segment_count: int = 8,
        router_readout: str = "segment_attention",
        router_layer_decoder: str = "layer_gru",
        router_output_granularity: str = "block",
        # Accepted only so existing threshold artifacts remain config-compatible.
        router_stateful: bool = False,
        router_state_detach: bool = True,
        router_update_state: bool = True,
        router_utility_num_groups: int = 0,
        router_enable_utility_scores: bool = False,
        **legacy_options: Any,
    ):
        super().__init__()
        if legacy_options:
            names = ", ".join(sorted(legacy_options))
            raise TypeError(f"Unsupported Router options: {names}")

        precision_levels = precision_levels or [3, 4, 5, 6, 7, 8]
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.precision_levels = [int(bit) for bit in precision_levels]
        self.num_precisions = len(self.precision_levels)
        self.num_precision_levels = self.num_precisions
        self.num_precision_blocks = len(self.BLOCK_NAMES)
        self.precision_block_names = self.BLOCK_NAMES
        self.router_hidden_size = int(router_hidden_size)
        self.temperature = float(temperature)
        self.use_gumbel_softmax = bool(use_gumbel_softmax)
        self.pooling_method = str(pooling_method)
        self.inference_sampling_prob = float(inference_sampling_prob)
        self.inference_low_bit_bias = float(inference_low_bit_bias)
        self.concept_dim = int(concept_dim)
        self.logit_dim = int(logit_dim)
        self.ttt_lr_linear = float(ttt_lr_linear)
        self.ttt_lr_ln = float(ttt_lr_ln)
        self.router_ttt_layers = int(router_ttt_layers)
        self.router_ttt_heads = int(router_ttt_heads)
        self.router_ttt_residual = str(router_ttt_residual).lower()
        self.initializer_range = float(initializer_range)
        self.router_input_source = str(router_input_source).lower()
        self.router_prelude_layers = int(router_prelude_layers)
        self.router_prelude_precision = int(router_prelude_precision)
        self.router_segment_count = int(router_segment_count)
        self.router_readout = str(router_readout).lower()
        self.router_layer_decoder = str(router_layer_decoder).lower()
        self.router_output_granularity = str(router_output_granularity).lower()

        canonical_requirements = {
            "precision_levels": (self.precision_levels, [3, 4, 5, 6, 7, 8]),
            "router_hidden_size": (self.router_hidden_size, 256),
            "temperature": (self.temperature, 1.0),
            "use_gumbel_softmax": (self.use_gumbel_softmax, True),
            "dropout_rate": (float(dropout_rate), 0.1),
            "pooling_method": (self.pooling_method, "mean"),
            "inference_sampling_prob": (self.inference_sampling_prob, 0.0),
            "inference_low_bit_bias": (self.inference_low_bit_bias, 0.0),
            "concept_dim": (self.concept_dim, 352),
            "logit_dim": (self.logit_dim, 64),
            "router_ttt_layers": (self.router_ttt_layers, 2),
            "router_stateful": (bool(router_stateful), False),
            "router_state_detach": (bool(router_state_detach), True),
            "router_update_state": (bool(router_update_state), True),
            "router_ttt_heads": (self.router_ttt_heads, 1),
            "router_ttt_residual": (self.router_ttt_residual, "none"),
            "initializer_range": (self.initializer_range, 0.02),
            "router_input_source": (
                self.router_input_source,
                "prelude_hidden",
            ),
            "router_prelude_layers": (self.router_prelude_layers, 2),
            "router_prelude_precision": (self.router_prelude_precision, 5),
            "router_segment_count": (self.router_segment_count, 8),
            "router_readout": (self.router_readout, "segment_attention"),
            "router_layer_decoder": (self.router_layer_decoder, "layer_gru"),
            "router_output_granularity": (
                self.router_output_granularity,
                "block",
            ),
            "router_utility_num_groups": (int(router_utility_num_groups), 0),
            "router_enable_utility_scores": (
                bool(router_enable_utility_scores),
                False,
            ),
        }
        invalid = {
            name: value
            for name, (value, expected) in canonical_requirements.items()
            if value != expected
        }
        if self.ttt_lr_linear < 0.0 or self.ttt_lr_ln < 0.0:
            invalid["ttt_lr"] = (
                self.ttt_lr_linear,
                self.ttt_lr_ln,
            )
        if self.ttt_lr_linear != self.ttt_lr_ln:
            invalid["tied_ttt_lr"] = (
                self.ttt_lr_linear,
                self.ttt_lr_ln,
            )
        if invalid:
            details = ", ".join(f"{key}={value!r}" for key, value in invalid.items())
            raise ValueError(
                "The canonical VAMP Router only supports the configured "
                f"architecture; incompatible options: {details}"
            )
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        self.native_precision_blocks = self.num_precision_blocks
        self.output_size = (
            self.num_layers * self.native_precision_blocks * self.num_precisions
        )

        self.register_buffer(
            "precision_usage_stats",
            torch.zeros(
                self.num_layers,
                self.num_precision_blocks,
                self.num_precisions,
            ),
        )
        self.register_buffer("total_forwards", torch.tensor(0.0))

        self.task_qkv_proj = nn.Linear(
            self.hidden_size, self.concept_dim * 3, bias=False
        )
        self.extra_ttt_qkv_projs = nn.ModuleList(
            [
                nn.Linear(self.concept_dim, self.concept_dim * 3, bias=False)
                for _ in range(self.router_ttt_layers - 1)
            ]
        )
        self.weight_linear = nn.Parameter(
            torch.empty(self.concept_dim, self.concept_dim, self.logit_dim)
        )
        self.weight_ln = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
        self.bias_linear = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
        self.bias_ln = nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
        self.extra_weight_linear = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.concept_dim, self.concept_dim, self.logit_dim)
                )
                for _ in range(self.router_ttt_layers - 1)
            ]
        )
        self.extra_weight_ln = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
                for _ in range(self.router_ttt_layers - 1)
            ]
        )
        self.extra_bias_linear = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
                for _ in range(self.router_ttt_layers - 1)
            ]
        )
        self.extra_bias_ln = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.concept_dim, self.logit_dim))
                for _ in range(self.router_ttt_layers - 1)
            ]
        )

        self.segment_readout_norm = nn.LayerNorm(self.concept_dim)
        self.segment_readout_score = nn.Linear(self.concept_dim, 1)

        # Kept for strict loading of existing threshold-map artifacts. The
        # layer-GRU path uses ``layer_precision_head`` for prediction.
        self.precision_head = nn.Sequential(
            nn.LayerNorm(self.concept_dim),
            nn.Linear(self.concept_dim, self.router_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.router_hidden_size, self.output_size),
        )
        self.layer_embeddings = nn.Parameter(
            torch.empty(self.num_layers, self.concept_dim)
        )
        self.layer_gru = nn.GRU(
            input_size=self.concept_dim,
            hidden_size=self.concept_dim,
            batch_first=True,
        )
        self.layer_precision_head = nn.Sequential(
            nn.LayerNorm(self.concept_dim),
            nn.Linear(
                self.concept_dim,
                self.native_precision_blocks * self.num_precisions,
            ),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.task_qkv_proj.weight)
        for projection in self.extra_ttt_qkv_projs:
            nn.init.xavier_uniform_(projection.weight)
        nn.init.normal_(self.weight_linear, mean=0.0, std=self.initializer_range)
        nn.init._no_grad_fill_(self.weight_ln, 1.0 / self.logit_dim)
        nn.init.normal_(
            self.bias_linear,
            mean=0.0,
            std=self.initializer_range / self.logit_dim,
        )
        nn.init.normal_(
            self.bias_ln,
            mean=0.0,
            std=self.initializer_range / self.logit_dim,
        )
        for weight in self.extra_weight_linear:
            nn.init.normal_(weight, mean=0.0, std=self.initializer_range)
        for weight in self.extra_weight_ln:
            nn.init._no_grad_fill_(weight, 1.0 / self.logit_dim)
        for bias in self.extra_bias_linear:
            nn.init.normal_(
                bias, mean=0.0, std=self.initializer_range / self.logit_dim
            )
        for bias in self.extra_bias_ln:
            nn.init.normal_(
                bias, mean=0.0, std=self.initializer_range / self.logit_dim
            )
        if self.segment_readout_score is not None:
            nn.init.xavier_uniform_(self.segment_readout_score.weight)
            nn.init.constant_(self.segment_readout_score.bias, 0.0)
        for module in self.precision_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        if self.layer_embeddings is not None:
            nn.init.normal_(
                self.layer_embeddings, mean=0.0, std=self.initializer_range
            )
        if self.layer_gru is not None:
            for name, parameter in self.layer_gru.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.constant_(parameter, 0.0)
        if self.layer_precision_head is not None:
            for module in self.layer_precision_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)

    def _init_fast_weights(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        shape = (batch_size, sequence_length, self.logit_dim)
        return {
            name: torch.ones(shape, device=device, dtype=dtype)
            for name in self.STATE_KEYS
        }

    def _fast_weight_bank(
        self, layer_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_index == 0:
            return self.weight_linear, self.weight_ln, self.bias_linear, self.bias_ln
        index = layer_index - 1
        return (
            self.extra_weight_linear[index],
            self.extra_weight_ln[index],
            self.extra_bias_linear[index],
            self.extra_bias_ln[index],
        )

    def _expanded_fast_weights(
        self, state: Dict[str, torch.Tensor], layer_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_linear, weight_ln, bias_linear, bias_ln = self._fast_weight_bank(
            layer_index
        )
        return (
            torch.einsum(
                "iol,bsl->bsio", weight_linear, state["weight_linear_tmp"]
            ),
            torch.einsum("ol,bsl->bso", weight_ln, state["weight_ln_tmp"]),
            torch.einsum(
                "ol,bsl->bso", bias_linear, state["bias_linear_tmp"]
            ),
            torch.einsum("ol,bsl->bso", bias_ln, state["bias_ln_tmp"]),
        )

    def _learn(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: Dict[str, torch.Tensor],
        layer_index: int,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        (
            weight_linear_tmp,
            weight_ln_tmp,
            bias_linear_tmp,
            bias_ln_tmp,
        ) = self._expanded_fast_weights(state, layer_index)
        z = torch.einsum("bsi,bsio->bso", key, weight_linear_tmp) + bias_linear_tmp
        mean = z.mean(dim=-1, keepdim=True)
        variance = z.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(variance + eps)
        normalized = (z - mean) / std
        error = weight_ln_tmp * normalized + bias_ln_tmp + key - value

        grad_weight_ln = error * normalized
        grad_linear = weight_ln_tmp * error / std
        grad_weight_linear = torch.einsum("bsi,bso->bsio", key, grad_linear)
        weight_linear, weight_ln, bias_linear, bias_ln = self._fast_weight_bank(
            layer_index
        )
        return {
            "weight_linear_tmp": state["weight_linear_tmp"]
            - self.ttt_lr_linear
            * torch.einsum("iol,bsio->bsl", weight_linear, grad_weight_linear),
            "weight_ln_tmp": state["weight_ln_tmp"]
            - self.ttt_lr_ln
            * torch.einsum("ol,bso->bsl", weight_ln, grad_weight_ln),
            "bias_linear_tmp": state["bias_linear_tmp"]
            - self.ttt_lr_linear
            * torch.einsum("ol,bso->bsl", bias_linear, grad_linear),
            "bias_ln_tmp": state["bias_ln_tmp"]
            - self.ttt_lr_ln
            * torch.einsum("ol,bso->bsl", bias_ln, error),
        }

    def _predict(
        self,
        query: torch.Tensor,
        state: Dict[str, torch.Tensor],
        layer_index: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        (
            weight_linear_tmp,
            weight_ln_tmp,
            bias_linear_tmp,
            bias_ln_tmp,
        ) = self._expanded_fast_weights(state, layer_index)
        z = torch.einsum("bsi,bsio->bso", query, weight_linear_tmp) + bias_linear_tmp
        mean = z.mean(dim=-1, keepdim=True)
        variance = z.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (z - mean) / torch.sqrt(variance + eps)
        return weight_ln_tmp * normalized + bias_ln_tmp + query

    def _apply_ttt_tokens(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        state = self._init_fast_weights(
            batch_size=query.shape[0],
            sequence_length=query.shape[1],
            device=query.device,
            dtype=query.dtype,
        )
        state = self._learn(key, value, state, layer_index)
        return self._predict(query, state, layer_index)

    def _run_ttt_block(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            return self._apply_ttt_tokens(query, key, value, layer_index)
        chunk_size = self.INFERENCE_TOKEN_CHUNK_SIZE
        if query.shape[1] <= chunk_size:
            return self._apply_ttt_tokens(query, key, value, layer_index)
        # Fast states are token-local. Bound the [B, S, D, D] temporaries;
        # readout still sees the complete sequence after all TTT blocks.
        return torch.cat(
            [
                self._apply_ttt_tokens(
                    query[:, start:start + chunk_size],
                    key[:, start:start + chunk_size],
                    value[:, start:start + chunk_size],
                    layer_index,
                )
                for start in range(0, query.shape[1], chunk_size)
            ],
            dim=1,
        )

    @staticmethod
    def _masked_mean(
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return states.mean(dim=1)
        mask = attention_mask.to(device=states.device, dtype=states.dtype)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if tuple(mask.shape) != tuple(states.shape[:2]):
            return states.mean(dim=1)
        mask = mask.unsqueeze(-1)
        return (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

    def _segment_readout(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        segment_count = min(self.router_segment_count, max(sequence_length, 1))
        boundaries = torch.linspace(
            0,
            sequence_length,
            steps=segment_count + 1,
            device=states.device,
        ).round().long()
        representations = []
        valid_segments = []
        for segment_index in range(segment_count):
            start = int(boundaries[segment_index].item())
            end = max(int(boundaries[segment_index + 1].item()), start + 1)
            segment = states[:, start:end, :]
            if attention_mask is None:
                representations.append(segment.mean(dim=1))
                valid_segments.append(
                    torch.ones(
                        batch_size, device=states.device, dtype=torch.bool
                    )
                )
            else:
                segment_mask = attention_mask[:, start:end]
                representations.append(self._masked_mean(segment, segment_mask))
                valid_segments.append(segment_mask.sum(dim=1) > 0)
        segment_states = torch.stack(representations, dim=1)
        valid_mask = torch.stack(valid_segments, dim=1)
        scores = self.segment_readout_score(
            self.segment_readout_norm(segment_states)
        ).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1e4)
        weights = F.softmax(scores, dim=-1).to(dtype=segment_states.dtype)
        return (segment_states * weights.unsqueeze(-1)).sum(dim=1)

    def _decode(self, states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        readout = self._segment_readout(states, attention_mask)
        decoder_input = (
            readout.unsqueeze(1)
            + self.layer_embeddings.unsqueeze(0).to(
                device=readout.device, dtype=readout.dtype
            )
        )
        decoded, _ = self.layer_gru(decoder_input)
        return self.layer_precision_head(decoded).view(
            readout.shape[0],
            self.num_layers,
            self.native_precision_blocks,
            self.num_precisions,
        )

    def _selection_outputs(
        self,
        precision_logits: torch.Tensor,
        hard_selection: Optional[bool],
    ) -> Dict[str, torch.Tensor]:
        probabilities = F.softmax(precision_logits / self.temperature, dim=-1)
        if hard_selection is None:
            hard_selection = not self.training
        if hard_selection:
            selected = torch.argmax(probabilities, dim=-1)
            one_hot = F.one_hot(selected, num_classes=self.num_precisions).float()
        elif self.use_gumbel_softmax:
            one_hot = F.gumbel_softmax(
                precision_logits,
                tau=self.temperature,
                hard=False,
                dim=-1,
            )
            selected = torch.argmax(one_hot, dim=-1)
        else:
            selected = torch.multinomial(
                probabilities.reshape(-1, self.num_precisions), 1
            ).view(*probabilities.shape[:-1])
            one_hot = F.one_hot(selected, num_classes=self.num_precisions).float()

        bit_values = torch.tensor(
            self.precision_levels,
            device=precision_logits.device,
            dtype=one_hot.dtype,
        )
        precision_values = torch.sum(one_hot * bit_values, dim=-1)
        with torch.no_grad():
            self.total_forwards += precision_logits.shape[0]
            self.precision_usage_stats += F.one_hot(
                selected.detach(), num_classes=self.num_precisions
            ).float().sum(dim=0)
        return {
            "precision_logits": precision_logits,
            "precision_probs": probabilities,
            "selected_precisions": selected,
            "precision_values": precision_values,
            "one_hot": one_hot,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hard_selection: Optional[bool] = None,
    ) -> Dict[str, Any]:
        router_input = hidden_states.to(self.task_qkv_proj.weight.dtype)
        query, key, value = self.task_qkv_proj(router_input).chunk(3, dim=-1)
        if attention_mask is not None:
            token_mask = attention_mask.to(
                device=query.device, dtype=query.dtype
            ).unsqueeze(-1)
            query = query * token_mask
            key = key * token_mask
            value = value * token_mask

        states = self._run_ttt_block(query, key, value, layer_index=0)
        for layer_index, projection in enumerate(
            self.extra_ttt_qkv_projs, start=1
        ):
            query, key, value = projection(states).chunk(3, dim=-1)
            if attention_mask is not None:
                token_mask = attention_mask.to(
                    device=query.device, dtype=query.dtype
                ).unsqueeze(-1)
                query = query * token_mask
                key = key * token_mask
                value = value * token_mask
            states = self._run_ttt_block(query, key, value, layer_index)

        return self._selection_outputs(
            self._decode(states, attention_mask),
            hard_selection,
        )

    def precision_values_to_schedule(
        self, precision_values: torch.Tensor
    ) -> Dict[int, Dict[str, int]]:
        values = torch.as_tensor(precision_values).detach()
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError("Expected one prompt-level precision map")
            values = values[0]
        expected_shape = (self.num_layers, self.num_precision_blocks)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Precision map must have shape {expected_shape}, got "
                f"{tuple(values.shape)}"
            )
        schedule: Dict[int, Dict[str, int]] = {}
        for layer_index in range(self.num_layers):
            mha_bit = int(values[layer_index, 0].item())
            mlp_bit = int(values[layer_index, 1].item())
            if mha_bit not in self.precision_levels or mlp_bit not in self.precision_levels:
                raise ValueError(
                    f"Unsupported precision at layer {layer_index}: "
                    f"mha={mha_bit}, mlp={mlp_bit}"
                )
            schedule[layer_index] = {"mha": mha_bit, "mlp": mlp_bit}
        return schedule

    def get_statistics(self) -> Dict[str, Any]:
        if not bool(self.total_forwards.item()):
            return {"message": "No Router forward has been performed"}
        usage = self.precision_usage_stats / self.total_forwards
        bit_values = torch.tensor(
            self.precision_levels,
            device=usage.device,
            dtype=usage.dtype,
        )
        branch_statistics: Dict[str, Any] = {}
        for branch_index, branch_name in enumerate(self.BLOCK_NAMES):
            branch_usage = usage[:, branch_index, :]
            branch_average = (branch_usage * bit_values).sum(dim=-1)
            overall_distribution = branch_usage.mean(dim=0)
            branch_statistics[branch_name] = {
                "overall_precision_distribution": {
                    f"{bit}-bit": float(overall_distribution[index])
                    for index, bit in enumerate(self.precision_levels)
                },
                "average_precision_per_layer": branch_average.tolist(),
                "overall_average_precision": float(branch_average.mean()),
            }
        overall_distribution = usage.reshape(-1, self.num_precisions).mean(dim=0)
        layer_usage = usage.mean(dim=1)
        layer_average = (layer_usage * bit_values).sum(dim=-1)
        return {
            "usage_stat_source": "hard_selected_precisions",
            "router_output_granularity": self.router_output_granularity,
            "most_used_precision_per_layer": [
                self.precision_levels[index]
                for index in torch.argmax(layer_usage, dim=-1).tolist()
            ],
            "branch_statistics": branch_statistics,
            "overall_precision_distribution": {
                f"{bit}-bit": float(overall_distribution[index])
                for index, bit in enumerate(self.precision_levels)
            },
            "average_precision_per_layer": layer_average.tolist(),
            "overall_average_precision": float(layer_average.mean()),
        }

    def reset_statistics(self) -> None:
        self.precision_usage_stats.zero_()
        self.total_forwards.zero_()

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, concept_dim={self.concept_dim}, "
            f"logit_dim={self.logit_dim}, ttt_layers={self.router_ttt_layers}, "
            f"num_layers={self.num_layers}, readout={self.router_readout}, "
            f"decoder={self.router_layer_decoder}, "
            f"output_granularity={self.router_output_granularity}, "
            f"precision_levels={self.precision_levels}"
        )


Router = OnlineLearningPrecisionRouter
