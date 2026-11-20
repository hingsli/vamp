"""Capacity-bounded HBF prefetch, with optional reuse-aware cache admission.

The original HBF controller stays frozen as a separately measured control.
This controller preserves Router decisions, kernels, and physical byte limits.
"""
from .hbf_runtime import ResourceController


VERSION = "qaq-bounded-hbf-v1"


def choose_residents(tasks, sizes, io_costs, budget, window):
    """Reserve a complete lookahead window and one replacement allocation.

The conservative reserve counts a resident again if it occurs in the window.
No assumption about future requests enters the selection.
"""
    if not tasks:
        return set(), 0
    reserve = max(sum(sizes[k] for k in tasks[i:i + window]) for i in range(len(tasks))) + max(sizes.values())
    capacity = max(0, budget - reserve)
    ranked = sorted(tasks, key=lambda k: (-io_costs[k] / sizes[k], k))
    selected, used = set(), 0
    for key in ranked:
        if used + sizes[key] <= capacity:
            selected.add(key)
            used += sizes[key]
    return selected, reserve


class BoundedHBFController(ResourceController):
    def __init__(self, *args, lookahead_modules=14, **kwargs):
        if lookahead_modules < 2:
            raise ValueError("Lookahead must cover at least two modules")
        self.lookahead = int(lookahead_modules)
        self.admission = "protected"
        self.protected = set()
        self.residents = {"qweight": set(), "lut": set()}
        self.plan_key = None
        self.execution = []
        self.position = self.submitted = 0
        self.task_by_key = {}
        super().__init__(*args, **kwargs)
        # QAQ's direct-logit path calls CausalLM.forward(), bypassing hooks on
        # that outer module. Its decoder backbone is invoked through __call__
        # in both direct-logit checks and generation, so attach there once.
        self.handle.remove()
        backbone = getattr(self.model.model, "model", self.model.model)
        self.handle = backbone.register_forward_pre_hook(self._forward, with_kwargs=True)

    def _room(self, kind, size, exclude=()):
        budget = self.qbudget if kind == "qweight" else self.lut_budget
        if size > budget:
            raise RuntimeError(f"One {kind} allocation exceeds its budget")
        self._reap()
        cache = self.qcache if kind == "qweight" else self.lcache
        blocked = self.protected | set(exclude) | self.residents[kind]
        while self.live[kind] + size > budget:
            if self.retired:
                self._count("capacity_waits")
                self._reap(wait=True)
                if self.live[kind] + size <= budget:
                    break
            candidates = [k for k in cache if k not in blocked]
            if not candidates:
                raise RuntimeError(f"Protected {kind} working set exceeds physical budget")
            key = min(candidates, key=lambda k: -cache[k]["bit"]) if kind == "lut" else candidates[0]
            self._drop(kind, key)

    def _drop(self, kind, key):
        if key in self.protected:
            raise RuntimeError("Attempt to evict prefetched, unconsumed resource")
        super()._drop(kind, key)

    def _prepare(self, length, label):
        from .OnDemandMixin import PipelineTask
        tasks = []
        for (i, j), layer in self.layers.items():
            if self.feature_mode and i >= self.model.router_prelude_layers:
                continue
            bit = int(layer.original_layer.precision)
            tasks.append(PipelineTask(i, j, bit, self.history.frequency((i, j, bit)),
                                     layer.cpu_luts[bit].numel() * layer.cpu_luts[bit].element_size()))
        tasks.sort(key=lambda t: (t.layer_idx, t.module_idx))
        phase = self.profile["profiles"].get(str(length))
        if phase is None:
            raise ValueError(f"No measured costs for sequence length {length}")
        costs = {int(b): {"comp": v["comp_ms"], "io": v["lut_io_ms"] +
                         (v["qweight_io_ms"] if self.offload else 0)} for b, v in phase.items()}
        self.planner = self.Planner(use_hebf=self.schedule == "hebf", bitwidth_profiles=costs)
        self.execution = [(t.layer_idx, t.module_idx) for t in tasks]
        self.task_by_key = {(t.layer_idx, t.module_idx): t for t in tasks}
        self.residents = {"qweight": set(), "lut": set()}
        reserves = {}
        for kind, budget in (("qweight", self.qbudget), ("lut", self.lut_budget)):
            if kind == "qweight" and not self.offload:
                continue
            sizes, benefits = {}, {}
            for task in tasks:
                key = task.layer_idx, task.module_idx
                layer = self.layers[key]
                source = layer.cpu_qweight[:task.precision] if kind == "qweight" else layer.cpu_luts[task.precision]
                sizes[key] = source.numel() * source.element_size()
                benefits[key] = task.frequency * phase[str(task.precision)][kind + "_io_ms"]
            if max(sum(sizes[k] for k in self.execution[i:i + self.lookahead]) for i in range(len(tasks))) > budget:
                raise RuntimeError(f"Lookahead does not fit {kind} budget")
            if self.admission == "protected":
                self.residents[kind], reserves[kind] = choose_residents(self.execution, sizes, benefits, budget, self.lookahead)
        self.reserves = reserves
        # A changed map can leave wider qweights behind. Release incompatible
        # admitted entries before relying on the selected physical byte bound.
        for kind, cache in (("qweight", self.qcache), ("lut", self.lcache)):
            for key in self.residents[kind] & cache.keys():
                if cache[key]["bit"] != self.task_by_key[key].precision:
                    self._drop(kind, key)

    def _forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            ids = kwargs.get("inputs_embeds")
        if ids is None:
            raise RuntimeError("Cannot determine HBF execution phase")
        if self.position != len(self.execution):
            raise RuntimeError("Previous bounded forward did not consume its complete plan")
        length = int(ids.shape[1])
        label = "features" if self.feature_mode else ("decode" if length == 1 else "prefill")
        self.protected.clear()
        # Frequencies are fixed within a request; compute each phase plan once.
        signature = (label, length, tuple(int(l.original_layer.precision) for l in self.layers.values()))
        if signature != self.plan_key:
            self._prepare(length, label)
            self.plan_key = signature
        self.position = self.submitted = 0
        detail = self.audit and label not in self.seen_phases
        self.stage = {"phase": label, "length": length, "tasks": [], "consumers": [], "batches": [],
                      "residents": {k: [list(x) for x in sorted(v)] for k, v in self.residents.items()},
                      "reserved_bytes": self.reserves} if detail else None
        if detail:
            self.stage_audits.append(self.stage)
        self.seen_phases.add(label)
        self.calls += 1
        self._fill()

    def _fill(self):
        end = min(len(self.execution), self.position + self.lookahead)
        self.protected = set(self.execution[self.position:end])
        tasks = [self.task_by_key[k] for k in self.execution[self.submitted:end]]
        queue = self.planner.build_schedule(tasks)
        if self.stage is not None:
            self.stage["batches"].append({"consumer_position": self.position, "start": self.submitted,
                                          "end": end, "keys": [[t.layer_idx, t.module_idx] for t in queue]})
        for task in queue:
            key = task.layer_idx, task.module_idx
            loaded = self._ensure(key, task.precision)
            if self.stage is not None:
                self.stage["tasks"].append({"layer": key[0], "module": key[1], "bit": task.precision,
                    "frequency": task.frequency, "score": self.planner._task_priority(task), "loads": loaded})
        self.submitted = end

    def consume(self, module, bit):
        key = self.module_keys[id(module)]
        if self.position >= len(self.execution) or key != self.execution[self.position] or key not in self.protected:
            raise RuntimeError(f"Consumer {key} differs from bounded execution order at {self.position}/{len(self.execution)}")
        if int(bit) != self.task_by_key[key].precision:
            raise RuntimeError("Consumer changed scheduled precision")
        for kind, cache in (("qweight", self.qcache), ("lut", self.lcache)):
            if kind == "qweight" and not self.offload:
                continue
            if key not in cache or (cache[key]["bit"] < bit if kind == "qweight" else cache[key]["bit"] != bit):
                raise RuntimeError("Prefetched resource was lost before consumption")
        super().consume(module, bit)

    def consumed(self, module):
        super().consumed(module)
        self.position += 1
        self.protected.discard(self.module_keys[id(module)])
        # Refill half a window at a time, after kernels have been submitted.
        # This provides several eligible tasks for HBF to reorder, while keeping
        # GPU computation ahead of the host's next prefetch submission.
        if self.position < len(self.execution) and self.submitted - self.position <= self.lookahead // 2:
            self._fill()

    def begin_request(self):
        self.plan_key = None
        super().begin_request()

    def reset_cache(self):
        self.protected.clear()
        self.plan_key = None
        super().reset_cache()

    def report(self):
        result = super().report()
        result.update(runtime_version=VERSION, lookahead_modules=self.lookahead, cache_admission=self.admission)
        return result
