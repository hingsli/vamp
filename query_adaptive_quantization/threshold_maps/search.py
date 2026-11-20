"""Threshold-constrained precision-map search against a uniform reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F


BRANCH_NAMES = ("mha", "mlp")


@dataclass
class PrecisionReference:
    bit: int
    probabilities: torch.Tensor
    entropy_term: torch.Tensor
    valid_mask: torch.Tensor
    nll_sum: float
    token_count: int


@dataclass
class MapEvaluation:
    precision_map: torch.Tensor
    kl_to_reference: float
    nll_sum: float
    token_count: int
    weighted_cost: float
    weighted_effective_bit: float
    average_bit: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision_map": self.precision_map.detach().cpu().tolist(),
            "kl_to_reference": self.kl_to_reference,
            "nll_sum": self.nll_sum,
            "token_count": self.token_count,
            "weighted_cost": self.weighted_cost,
            "weighted_effective_bit": self.weighted_effective_bit,
            "average_bit": self.average_bit,
        }


class BlockWeightCostModel:
    """Weight-byte proxy for each layer's MHA and MLP branch."""

    def __init__(self, weights: torch.Tensor):
        if weights.ndim != 2 or weights.shape[1] != 2:
            raise ValueError(f"Expected [layers, 2] branch weights, got {weights.shape}")
        if not torch.all(weights > 0):
            raise ValueError("Every layer/branch must contain quantized weights")
        self.weights = weights.detach().double().cpu()
        self.total_weights = float(self.weights.sum().item())

    @classmethod
    def from_model(cls, model) -> "BlockWeightCostModel":
        module_names = list(model.config.anyprec["arch_config"]["module_names"])
        rows = []
        for layer_idx, layer_modules in enumerate(model.layer_to_aplinears):
            if len(layer_modules) != len(module_names):
                raise ValueError(
                    f"Layer {layer_idx} has {len(layer_modules)} APLinear modules, "
                    f"expected {len(module_names)}"
                )
            branch_weights = [0, 0]
            for name, module in zip(module_names, layer_modules):
                if "attn" in name:
                    branch_idx = 0
                elif "mlp" in name or "fc" in name:
                    branch_idx = 1
                else:
                    raise ValueError(f"Cannot assign quantized module {name!r} to a branch")
                branch_weights[branch_idx] += int(module.in_features) * int(
                    module.out_features
                )
            rows.append(branch_weights)
        return cls(torch.tensor(rows, dtype=torch.float64))

    def cost(self, precision_map: torch.Tensor) -> float:
        values = precision_map.detach().double().cpu()
        if tuple(values.shape) != tuple(self.weights.shape):
            raise ValueError(
                f"Precision map shape {tuple(values.shape)} does not match "
                f"cost shape {tuple(self.weights.shape)}"
            )
        return float((values * self.weights).sum().item())

    def effective_bit(self, precision_map: torch.Tensor) -> float:
        return self.cost(precision_map) / self.total_weights

    def manifest(self) -> dict[str, Any]:
        branch_totals = self.weights.sum(dim=0)
        return {
            "layer_branch_weight_counts": self.weights.long().tolist(),
            "branch_weight_counts": {
                name: int(branch_totals[idx].item())
                for idx, name in enumerate(BRANCH_NAMES)
            },
            "branch_weight_fractions": {
                name: float(branch_totals[idx].item() / self.total_weights)
                for idx, name in enumerate(BRANCH_NAMES)
            },
            "total_weight_count": int(self.total_weights),
            "cost_definition": "sum(in_features*out_features*bit)",
        }


class ThresholdPrecisionMapSearcher:
    """Generate verified low-cost maps under a per-sample quality threshold."""

    def __init__(
        self,
        model,
        *,
        precision_levels: Iterable[int] = (3, 4, 5, 6, 7, 8),
        reference_bit: int = 8,
        num_groups: int = 6,
        max_dp_candidates: int = 8,
        refinement_passes: int = 2,
        max_swap_candidates: int = 16,
        feasibility_tolerance: float = 0.0,
        max_frontier_states: int = 4096,
    ):
        self.model = model
        self.precision_levels = tuple(sorted({int(bit) for bit in precision_levels}))
        if len(self.precision_levels) < 2:
            raise ValueError("At least two candidate precision levels are required")
        self.reference_bit = int(reference_bit)
        if self.reference_bit not in self.precision_levels:
            raise ValueError(
                f"reference_bit={self.reference_bit} must be present in "
                f"candidate precision levels {self.precision_levels}"
            )
        if not any(bit < self.reference_bit for bit in self.precision_levels):
            raise ValueError(
                "At least one candidate precision must be lower than reference_bit"
            )
        self._lower_bit = {
            bit: self.precision_levels[index - 1]
            for index, bit in enumerate(self.precision_levels)
            if index > 0
        }
        self._higher_bit = {
            bit: self.precision_levels[index + 1]
            for index, bit in enumerate(self.precision_levels[:-1])
        }
        self.num_layers = int(len(model.layer_to_aplinears))
        self.num_groups = max(1, min(int(num_groups), self.num_layers))
        self.max_dp_candidates = max(1, int(max_dp_candidates))
        self.refinement_passes = max(0, int(refinement_passes))
        self.max_swap_candidates = max(0, int(max_swap_candidates))
        self.feasibility_tolerance = max(0.0, float(feasibility_tolerance))
        self.max_frontier_states = max(32, int(max_frontier_states))
        self.cost_model = BlockWeightCostModel.from_model(model)
        self.group_layers = self._contiguous_groups(self.num_layers, self.num_groups)
        self.forward_count = 0

    @staticmethod
    def _contiguous_groups(num_layers: int, num_groups: int) -> list[list[int]]:
        groups: list[list[int]] = []
        for group_idx in range(num_groups):
            start = (group_idx * num_layers) // num_groups
            end = ((group_idx + 1) * num_layers) // num_groups
            groups.append(list(range(start, end)))
        return groups

    def full_map(self, bit: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (self.num_layers, 2), int(bit), dtype=torch.long, device=device
        )

    @staticmethod
    def map_to_schedule(precision_map: torch.Tensor) -> dict[int, dict[str, int]]:
        values = precision_map.detach().cpu()
        return {
            layer_idx: {
                "mha": int(values[layer_idx, 0].item()),
                "mlp": int(values[layer_idx, 1].item()),
            }
            for layer_idx in range(int(values.shape[0]))
        }

    def _set_map(self, precision_map: torch.Tensor) -> None:
        schedule = self.map_to_schedule(precision_map)
        if hasattr(self.model, "_set_precision_silent"):
            self.model._set_precision_silent(schedule)
        else:
            self.model.set_precision(schedule)

    @staticmethod
    def _extract_logits(outputs) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Model output does not contain logits")

    @staticmethod
    def _valid_next_token_mask(
        input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if input_ids.shape[1] < 2:
            raise ValueError("Threshold-map search requires at least two input tokens")
        if attention_mask is None:
            return torch.ones_like(input_ids[:, 1:], dtype=torch.bool)
        return attention_mask[:, 1:].to(device=input_ids.device).bool()

    @staticmethod
    def _nll_stats(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[float, int]:
        shift_logits = logits[:, :-1, :].float()
        targets = input_ids[:, 1:].to(device=shift_logits.device)
        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        mask = valid_mask.to(device=token_nll.device, dtype=token_nll.dtype)
        return float((token_nll * mask).sum().item()), int(mask.sum().item())

    @torch.no_grad()
    def build_reference(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> PrecisionReference:
        self._set_map(self.full_map(self.reference_bit, input_ids.device))
        model_inputs = {"input_ids": input_ids, "use_router": False, "use_cache": False}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        logits = self._extract_logits(self.model(**model_inputs)).detach()
        self.forward_count += 1
        valid_mask = self._valid_next_token_mask(input_ids, attention_mask)
        nll_sum, token_count = self._nll_stats(logits, input_ids, valid_mask)
        reference_logits = logits[:, :-1, :].float()
        log_probs = F.log_softmax(reference_logits, dim=-1)
        probabilities = log_probs.exp()
        entropy_term = (probabilities * log_probs).sum(dim=-1)
        del logits, reference_logits, log_probs
        return PrecisionReference(
            bit=self.reference_bit,
            probabilities=probabilities,
            entropy_term=entropy_term,
            valid_mask=valid_mask,
            nll_sum=nll_sum,
            token_count=token_count,
        )

    @torch.no_grad()
    def evaluate_map(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        precision_map: torch.Tensor,
        reference: PrecisionReference,
    ) -> MapEvaluation:
        self._set_map(precision_map)
        model_inputs = {"input_ids": input_ids, "use_router": False, "use_cache": False}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        logits = self._extract_logits(self.model(**model_inputs)).detach()
        self.forward_count += 1
        candidate_log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        kl_token = reference.entropy_term - (
            reference.probabilities * candidate_log_probs
        ).sum(dim=-1)
        mask = reference.valid_mask.to(device=kl_token.device, dtype=kl_token.dtype)
        kl_value = float(((kl_token * mask).sum() / mask.sum().clamp_min(1)).item())
        nll_sum, token_count = self._nll_stats(logits, input_ids, reference.valid_mask)
        del logits, candidate_log_probs, kl_token
        return MapEvaluation(
            precision_map=precision_map.detach().clone(),
            kl_to_reference=max(0.0, kl_value),
            nll_sum=nll_sum,
            token_count=token_count,
            weighted_cost=self.cost_model.cost(precision_map),
            weighted_effective_bit=self.cost_model.effective_bit(precision_map),
            average_bit=float(precision_map.detach().float().mean().item()),
        )

    def _group_map_from_bits(
        self, group_bits: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        if len(group_bits) != self.num_groups * 2:
            raise ValueError("Group-bit path length does not match search decisions")
        precision_map = self.full_map(self.reference_bit, device)
        for group_idx, layers in enumerate(self.group_layers):
            for branch_idx in range(2):
                bit = int(group_bits[group_idx * 2 + branch_idx])
                precision_map[layers, branch_idx] = bit
        return precision_map

    def _group_slot_cost(self, group_idx: int, branch_idx: int, bit: int) -> float:
        layers = self.group_layers[group_idx]
        weight = float(self.cost_model.weights[layers, branch_idx].sum().item())
        return weight * int(bit)

    def _prune_frontier(
        self, states: list[tuple[float, float, tuple[int, ...]]]
    ) -> list[tuple[float, float, tuple[int, ...]]]:
        states.sort(key=lambda item: (item[0], item[1], item[2]))
        frontier = []
        best_loss = math.inf
        for state in states:
            if state[1] < best_loss - 1e-12:
                frontier.append(state)
                best_loss = state[1]
        if len(frontier) <= self.max_frontier_states:
            return frontier
        stride = math.ceil(len(frontier) / self.max_frontier_states)
        reduced = frontier[::stride]
        if reduced[-1] != frontier[-1]:
            reduced.append(frontier[-1])
        return reduced

    def _pareto_group_paths(
        self, local_kl: torch.Tensor
    ) -> list[tuple[float, float, tuple[int, ...]]]:
        states: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
        for slot_idx in range(self.num_groups * 2):
            group_idx, branch_idx = divmod(slot_idx, 2)
            expanded = []
            for cost, predicted_kl, path in states:
                for level_idx, bit in enumerate(self.precision_levels):
                    expanded.append(
                        (
                            cost + self._group_slot_cost(group_idx, branch_idx, bit),
                            predicted_kl
                            + max(0.0, float(local_kl[group_idx, branch_idx, level_idx])),
                            path + (int(bit),),
                        )
                    )
            states = self._prune_frontier(expanded)
        return states

    @staticmethod
    def _evaluation_key(evaluation: MapEvaluation) -> tuple[Any, ...]:
        flat_map = tuple(
            int(value) for value in evaluation.precision_map.detach().cpu().reshape(-1)
        )
        return (evaluation.weighted_cost, evaluation.kl_to_reference, flat_map)

    def _is_feasible(self, evaluation: MapEvaluation, threshold: float) -> bool:
        return (
            evaluation.kl_to_reference
            <= float(threshold) + self.feasibility_tolerance
        )

    @staticmethod
    def _evaluation_from_record(
        record: dict[str, Any], device: torch.device
    ) -> MapEvaluation:
        kl_to_reference = record.get("kl_to_reference", record.get("kl_to_int8"))
        if kl_to_reference is None:
            raise ValueError("Map record is missing kl_to_reference")
        return MapEvaluation(
            precision_map=torch.tensor(
                record["precision_map"], dtype=torch.long, device=device
            ),
            kl_to_reference=float(kl_to_reference),
            nll_sum=float(record["nll_sum"]),
            token_count=int(record["token_count"]),
            weighted_cost=float(record["weighted_cost"]),
            weighted_effective_bit=float(record["weighted_effective_bit"]),
            average_bit=float(record["average_bit"]),
        )

    @torch.no_grad()
    def group_sweep(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        thresholds: Iterable[float],
        reference: Optional[PrecisionReference] = None,
    ) -> dict[str, Any]:
        previous = (
            self.model._snapshot_precisions()
            if hasattr(self.model, "_snapshot_precisions")
            else None
        )
        start_count = self.forward_count
        try:
            if reference is None:
                reference = self.build_reference(input_ids, attention_mask)
            reference_map = self.full_map(self.reference_bit, input_ids.device)
            reference_evaluation = MapEvaluation(
                precision_map=reference_map,
                kl_to_reference=0.0,
                nll_sum=reference.nll_sum,
                token_count=reference.token_count,
                weighted_cost=self.cost_model.cost(reference_map),
                weighted_effective_bit=float(self.reference_bit),
                average_bit=float(self.reference_bit),
            )
            uniform = {str(self.reference_bit): reference_evaluation.as_dict()}
            for bit in self.precision_levels:
                if bit == self.reference_bit:
                    continue
                uniform[str(bit)] = self.evaluate_map(
                    input_ids,
                    attention_mask,
                    self.full_map(bit, input_ids.device),
                    reference,
                ).as_dict()

            local_kl = torch.zeros(
                self.num_groups, 2, len(self.precision_levels), dtype=torch.float64
            )
            for group_idx, layers in enumerate(self.group_layers):
                for branch_idx in range(2):
                    for level_idx, bit in enumerate(self.precision_levels):
                        if bit == self.reference_bit:
                            continue
                        candidate = reference_map.clone()
                        candidate[layers, branch_idx] = int(bit)
                        evaluation = self.evaluate_map(
                            input_ids, attention_mask, candidate, reference
                        )
                        local_kl[group_idx, branch_idx, level_idx] = (
                            evaluation.kl_to_reference
                        )

            frontier = self._pareto_group_paths(local_kl)
            threshold_results = {}
            for threshold in sorted({float(value) for value in thresholds}):
                predicted_feasible = [
                    state
                    for state in frontier
                    if state[1] <= threshold + self.feasibility_tolerance
                ]
                predicted_feasible.sort(key=lambda item: (item[0], item[1], item[2]))
                proposals = predicted_feasible[: self.max_dp_candidates]
                exact_evaluations = []
                for _, predicted_kl, path in proposals:
                    evaluation = self.evaluate_map(
                        input_ids,
                        attention_mask,
                        self._group_map_from_bits(path, input_ids.device),
                        reference,
                    )
                    record = evaluation.as_dict()
                    record["predicted_additive_kl"] = float(predicted_kl)
                    exact_evaluations.append((evaluation, record))
                feasible = [
                    pair for pair in exact_evaluations if self._is_feasible(pair[0], threshold)
                ]
                if feasible:
                    selected_eval, selected_record = min(
                        feasible, key=lambda pair: self._evaluation_key(pair[0])
                    )
                    status = "group_dp_feasible"
                else:
                    selected_eval = reference_evaluation
                    selected_record = reference_evaluation.as_dict()
                    selected_record["predicted_additive_kl"] = 0.0
                    status = "fallback_reference"
                threshold_results[f"{threshold:.10g}"] = {
                    **selected_record,
                    "threshold": threshold,
                    "threshold_margin": (
                        threshold - selected_eval.kl_to_reference
                    ),
                    "search_status": status,
                    "num_exact_dp_candidates": len(exact_evaluations),
                }
            return {
                "reference_bit": self.reference_bit,
                "candidate_bits": list(self.precision_levels),
                "reference": reference_evaluation.as_dict(),
                "uniform": uniform,
                "threshold_results": threshold_results,
                "local_group_kl": local_kl.tolist(),
                "group_layers": self.group_layers,
                "search_forwards": self.forward_count - start_count,
            }
        finally:
            if previous is not None and hasattr(self.model, "_restore_precisions"):
                self.model._restore_precisions(previous)

    def _refine_downgrades(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        reference: PrecisionReference,
        current: MapEvaluation,
        threshold: float,
    ) -> MapEvaluation:
        for _ in range(self.refinement_passes):
            accepted = 0
            slots = []
            for layer_idx in range(self.num_layers):
                for branch_idx in range(2):
                    bit = int(current.precision_map[layer_idx, branch_idx].item())
                    lower_bit = self._lower_bit.get(bit)
                    if lower_bit is None:
                        continue
                    saving = float(
                        self.cost_model.weights[layer_idx, branch_idx].item()
                    ) * (bit - lower_bit)
                    slots.append((-saving, layer_idx, branch_idx))
            slots.sort()
            for _, layer_idx, branch_idx in slots:
                candidate = current.precision_map.clone()
                current_bit = int(candidate[layer_idx, branch_idx].item())
                candidate[layer_idx, branch_idx] = self._lower_bit[current_bit]
                evaluation = self.evaluate_map(
                    input_ids, attention_mask, candidate, reference
                )
                if self._is_feasible(evaluation, threshold) and (
                    evaluation.weighted_cost < current.weighted_cost
                ):
                    current = evaluation
                    accepted += 1
            if accepted == 0:
                break
        return current

    def _refine_swaps(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        reference: PrecisionReference,
        current: MapEvaluation,
        threshold: float,
    ) -> MapEvaluation:
        if self.max_swap_candidates <= 0:
            return current
        proposals = []
        values = current.precision_map
        for up_layer in range(self.num_layers):
            for up_branch in range(2):
                current_up_bit = int(values[up_layer, up_branch].item())
                higher_bit = self._higher_bit.get(current_up_bit)
                if higher_bit is None:
                    continue
                up_cost = float(
                    self.cost_model.weights[up_layer, up_branch].item()
                ) * (higher_bit - current_up_bit)
                for down_layer in range(self.num_layers):
                    for down_branch in range(2):
                        if up_layer == down_layer and up_branch == down_branch:
                            continue
                        current_down_bit = int(
                            values[down_layer, down_branch].item()
                        )
                        lower_bit = self._lower_bit.get(current_down_bit)
                        if lower_bit is None:
                            continue
                        down_saving = float(
                            self.cost_model.weights[down_layer, down_branch].item()
                        ) * (current_down_bit - lower_bit)
                        net_cost = up_cost - down_saving
                        if net_cost >= 0:
                            continue
                        proposals.append(
                            (
                                net_cost,
                                up_layer,
                                up_branch,
                                down_layer,
                                down_branch,
                                higher_bit,
                                lower_bit,
                            )
                        )
        proposals.sort()
        best = current
        seen = set()
        for proposal in proposals:
            (
                _,
                up_layer,
                up_branch,
                down_layer,
                down_branch,
                higher_bit,
                lower_bit,
            ) = proposal
            candidate = current.precision_map.clone()
            candidate[up_layer, up_branch] = higher_bit
            candidate[down_layer, down_branch] = lower_bit
            key = tuple(int(value) for value in candidate.detach().cpu().reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            evaluation = self.evaluate_map(
                input_ids, attention_mask, candidate, reference
            )
            if self._is_feasible(evaluation, threshold) and (
                self._evaluation_key(evaluation) < self._evaluation_key(best)
            ):
                best = evaluation
            if len(seen) >= self.max_swap_candidates:
                break
        return best

    @torch.no_grad()
    def search_selected_threshold(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        threshold: float,
    ) -> dict[str, Any]:
        previous = (
            self.model._snapshot_precisions()
            if hasattr(self.model, "_snapshot_precisions")
            else None
        )
        start_count = self.forward_count
        try:
            reference = self.build_reference(input_ids, attention_mask)
            sweep = self.group_sweep(
                input_ids, attention_mask, [threshold], reference=reference
            )
            group_record = sweep["threshold_results"][f"{float(threshold):.10g}"]
            group_eval = self._evaluation_from_record(
                group_record, input_ids.device
            )
            if not self._is_feasible(group_eval, threshold):
                reference_map = self.full_map(
                    self.reference_bit, input_ids.device
                )
                group_eval = MapEvaluation(
                    precision_map=reference_map,
                    kl_to_reference=0.0,
                    nll_sum=reference.nll_sum,
                    token_count=reference.token_count,
                    weighted_cost=self.cost_model.cost(reference_map),
                    weighted_effective_bit=float(self.reference_bit),
                    average_bit=float(self.reference_bit),
                )
            refined = self._refine_downgrades(
                input_ids, attention_mask, reference, group_eval, threshold
            )
            refined = self._refine_swaps(
                input_ids, attention_mask, reference, refined, threshold
            )
            if not self._is_feasible(refined, threshold):
                raise RuntimeError("Refinement produced an infeasible precision map")
            return {
                **refined.as_dict(),
                "reference_bit": self.reference_bit,
                "reference_nll_sum": float(reference.nll_sum),
                "reference_token_count": int(reference.token_count),
                "threshold": float(threshold),
                "threshold_margin": (
                    float(threshold) - refined.kl_to_reference
                ),
                "group_map": group_record["precision_map"],
                "group_map_kl": group_record.get(
                    "kl_to_reference", group_record.get("kl_to_int8")
                ),
                "group_map_weighted_effective_bit": group_record[
                    "weighted_effective_bit"
                ],
                "uniform_baseline": sweep["uniform"],
                "search_status": "refined_feasible",
                "search_forwards": self.forward_count - start_count,
            }
        finally:
            if previous is not None and hasattr(self.model, "_restore_precisions"):
                self.model._restore_precisions(previous)

    @torch.no_grad()
    def refine_selected_group_record(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        threshold: float,
        group_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Refine a verified group result already produced during a pilot sweep."""
        previous = (
            self.model._snapshot_precisions()
            if hasattr(self.model, "_snapshot_precisions")
            else None
        )
        start_count = self.forward_count
        try:
            reference = self.build_reference(input_ids, attention_mask)
            group_eval = self._evaluation_from_record(group_record, input_ids.device)
            if not self._is_feasible(group_eval, threshold):
                reference_map = self.full_map(
                    self.reference_bit, input_ids.device
                )
                group_eval = MapEvaluation(
                    precision_map=reference_map,
                    kl_to_reference=0.0,
                    nll_sum=reference.nll_sum,
                    token_count=reference.token_count,
                    weighted_cost=self.cost_model.cost(reference_map),
                    weighted_effective_bit=float(self.reference_bit),
                    average_bit=float(self.reference_bit),
                )
            refined = self._refine_downgrades(
                input_ids, attention_mask, reference, group_eval, threshold
            )
            refined = self._refine_swaps(
                input_ids, attention_mask, reference, refined, threshold
            )
            if not self._is_feasible(refined, threshold):
                raise RuntimeError("Refinement produced an infeasible precision map")
            return {
                **refined.as_dict(),
                "reference_bit": self.reference_bit,
                "reference_nll_sum": float(reference.nll_sum),
                "reference_token_count": int(reference.token_count),
                "threshold": float(threshold),
                "threshold_margin": (
                    float(threshold) - refined.kl_to_reference
                ),
                "group_map": group_record["precision_map"],
                "group_map_kl": float(
                    group_record.get(
                        "kl_to_reference", group_record.get("kl_to_int8")
                    )
                ),
                "group_map_weighted_effective_bit": float(
                    group_record["weighted_effective_bit"]
                ),
                "search_status": "refined_cached_group_feasible",
                "search_forwards": self.forward_count - start_count,
            }
        finally:
            if previous is not None and hasattr(self.model, "_restore_precisions"):
                self.model._restore_precisions(previous)
