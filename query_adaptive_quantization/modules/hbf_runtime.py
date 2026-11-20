"""Shared HBF history, cost validation, CUDA transfers and physical storage accounting.

Scheduling and admission are supplied by the bounded controller.
Scheduling uses measured computation and transfer costs.
"""
from collections import Counter, OrderedDict, deque
import hashlib
import json
from ..cost_profile import validate_profiles


VERSION = "qaq-hbf-v1"


class RequestHistory:
    def __init__(self, window=32):
        if window <= 0:
            raise ValueError("History window must be positive")
        self.window, self.requests, self.counts = window, deque(), Counter()

    def frequency(self, key):
        return max(1, self.counts[key])

    def complete(self, keys):
        keys = frozenset(keys)
        self.requests.append(keys)
        self.counts.update(keys)
        if len(self.requests) > self.window:
            self.counts.subtract(self.requests.popleft())
            self.counts += Counter()

    def digest(self):
        return hashlib.sha256(json.dumps(sorted(self.counts.items())).encode()).hexdigest()




class ResourceController:
    def __init__(self, model, profile, *, schedule="hebf", asynchronous=True,
                 lut_budget_bytes, frequency_window=32, audit=False):
        import torch
        from .OnDemandMixin import IOComputePipelinePlanner
        self.torch, self.Planner = torch, IOComputePipelinePlanner
        self.model, self.manager = model, model.on_demand_manager
        if self.manager.layer_prefetch is not None:
            raise ValueError("Original HBF cannot use a fixed-prefix controller")
        if schedule not in ("hebf", "layer_order"):
            raise ValueError("Unknown scheduling policy")
        self.profile = validate_profiles(profile)
        self.schedule, self.asynchronous = schedule, asynchronous
        self.history, self.audit = RequestHistory(frequency_window), audit
        self.lut_budget = int(lut_budget_bytes)
        self.offload = self.manager.qweight_offload_enabled
        self.qbudget = int(self.manager.qweight_offload_budget_bytes)
        self.device = model.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.compute = torch.cuda.default_stream(self.device)
        self.layers = {(i, j): l for i, row in enumerate(self.manager.on_demand_layers)
                       for j, l in enumerate(row)}
        self.module_keys = {id(l.original_layer): k for k, l in self.layers.items()}
        self.qcache, self.lcache = OrderedDict(), OrderedDict()
        self.retired = []
        self.live = {"qweight": 0, "lut": 0}
        self.peak = dict(self.live)
        self.counts = Counter()
        self.feature_mode = False
        self.stage = None
        self.stage_audits, self.seen_phases = [], set()
        self.request_keys = None
        self.request_active = False
        self.calls = self.consumers = 0
        self.manager.original_hbf = self
        self.handle = model.model.register_forward_pre_hook(self._forward, with_kwargs=True)
        # Attach transfer hooks only when this controller is explicitly enabled.
        self.consumer_handles = []
        for layer in self.layers.values():
            self.consumer_handles.append(layer.original_layer.register_forward_pre_hook(self._before_linear, with_kwargs=True))
            self.consumer_handles.append(layer.original_layer.register_forward_hook(self._after_linear))
        original = model.get_router_input_hidden_states
        def features(*args, **kwargs):
            self.feature_mode = True
            try:
                return original(*args, **kwargs)
            finally:
                self.feature_mode = False
        model.get_router_input_hidden_states = features

    def _before_linear(self, module, args, kwargs):
        self.consume(module, int(kwargs.get("precision", module.precision)))

    def _after_linear(self, module, args, output):
        self.consumed(module)

    def _count(self, key, value=1):
        if self.audit:
            self.counts[key] += value

    def _account(self, kind, amount):
        self.live[kind] += amount
        self.peak[kind] = max(self.peak[kind], self.live[kind])
        limit = self.qbudget if kind == "qweight" else self.lut_budget
        if self.live[kind] < 0 or self.live[kind] > limit:
            raise RuntimeError(f"Physical {kind} budget violation: {self.live[kind]}/{limit}")
        if kind == "qweight":
            self.manager._qweight_cache_bytes = self.live[kind]
            self.manager._qweight_cache_peak_bytes = self.peak[kind]

    def _reap(self, wait=False):
        pending = []
        for kind, tensor, event in self.retired:
            if wait:
                event.synchronize()
            if event.query():
                self._account(kind, -tensor.untyped_storage().nbytes())
            else:
                pending.append((kind, tensor, event))
        self.retired = pending

    def _retire(self, kind, entry):
        # The retirement event follows both its upload and last consuming kernel.
        with self.torch.cuda.stream(self.stream):
            self.stream.wait_event(entry["ready"])
            if entry.get("used") is not None:
                self.stream.wait_event(entry["used"])
            event = self.torch.cuda.Event()
            event.record(self.stream)
        self.retired.append((kind, entry["tensor"], event))

    def _drop(self, kind, key):
        cache = self.qcache if kind == "qweight" else self.lcache
        entry = cache.pop(key)
        l = self.layers[key]
        if kind == "qweight":
            l.original_layer.qweight = l.cpu_qweight
        else:
            l.original_layer._buffers.pop(f"lut{entry['bit']}", None)
            l.current_gpu_precision = None
        self._count(kind + "_eviction_events")
        self._count(kind + "_evicted_bytes", entry["tensor"].numel() * entry["tensor"].element_size())
        self._retire(kind, entry)

    def _room(self, kind, size, exclude=()):
        raise NotImplementedError("Choose an explicit scheduling controller")

    def _upload(self, kind, key, bit):
        torch = self.torch
        l = self.layers[key]
        if kind == "qweight":
            if l._cpu_qweight_pinned is None:
                l._cpu_qweight_pinned = l.cpu_qweight.pin_memory()
            source = l._cpu_qweight_pinned[:bit]
        else:
            if bit not in l._cpu_luts_pinned:
                l._cpu_luts_pinned[bit] = l.cpu_luts[bit].pin_memory()
            source = l._cpu_luts_pinned[bit]
        size = source.numel() * source.element_size()
        self._room(kind, size, exclude=(key,))
        stream = self.stream if self.asynchronous else self.compute
        with torch.cuda.stream(stream):
            tensor = torch.empty(source.shape, dtype=source.dtype, device=self.device)
            self._account(kind, tensor.untyped_storage().nbytes())
            tensor.copy_(source, non_blocking=self.asynchronous)
            ready = torch.cuda.Event()
            ready.record(stream)
        cache = self.qcache if kind == "qweight" else self.lcache
        old = cache.pop(key, None)
        if old is not None:
            self._retire(kind, old)
        cache[key] = {"tensor": tensor, "bit": bit, "ready": ready, "used": None}
        if kind == "qweight":
            l.original_layer.qweight = tensor
        else:
            if old is not None:
                l.original_layer._buffers.pop(f"lut{old['bit']}", None)
            l.original_layer.register_buffer(f"lut{bit}", tensor)
            l.current_gpu_precision = bit
        self._count(kind + "_upload_bytes", size)
        self._count(kind + "_upload_calls")
        self._reap()

    def _ensure(self, key, bit):
        if bit not in self.layers[key].supported_bits:
            raise ValueError("HBF cannot silently substitute precision")
        loaded = []
        for kind, cache in (("qweight", self.qcache), ("lut", self.lcache)):
            if kind == "qweight" and not self.offload:
                continue
            hit = key in cache and (cache[key]["bit"] >= bit if kind == "qweight" else cache[key]["bit"] == bit)
            if hit:
                self._count(kind + "_cache_hits")
                cache.move_to_end(key)
            else:
                self._upload(kind, key, bit)
                loaded.append(kind)
        return loaded

    def accept_map(self, tasks):
        if self.request_active:
            self.request_keys = {(t.layer_idx, t.module_idx, t.precision) for t in tasks}
        return {"loaded_count": 0, "schedule_len": len(tasks), "hbf_consumer_controller": VERSION}

    def _forward(self, module, args, kwargs):
        raise NotImplementedError("Choose an explicit scheduling controller")

    def consume(self, module, bit):
        if self.torch.cuda.current_stream(self.device) != self.compute:
            raise RuntimeError("Packed HBF kernels require the default compute stream")
        key = self.module_keys[id(module)]
        loaded = self._ensure(key, int(bit))
        for kind, cache in (("qweight", self.qcache), ("lut", self.lcache)):
            if kind == "qweight" and not self.offload:
                continue
            self.compute.wait_event(cache[key]["ready"])
            cache[key]["tensor"].record_stream(self.compute)
        self.consumers += 1
        if self.stage is not None:
            self.stage["consumers"].append({"layer": key[0], "module": key[1], "bit": bit, "reloads": loaded})

    def consumed(self, module):
        key = self.module_keys[id(module)]
        used = self.torch.cuda.Event()
        used.record(self.compute)
        for cache in (self.qcache, self.lcache):
            if key in cache:
                cache[key]["used"] = used

    def begin_request(self):
        self.stage_audits, self.seen_phases = [], set()
        self.calls = self.consumers = 0
        self.request_keys, self.request_active = None, True
        self.history_before = self.history.digest() if self.audit else None
        self.frequencies = {str(k): self.history.frequency(k) for k in sorted(self.history.counts)} if self.audit else None

    def finish_request(self):
        if not self.request_active or self.request_keys is None or not self.calls or not self.consumers:
            raise RuntimeError("HBF request bypassed routing or execution")
        self.history.complete(self.request_keys)
        self.request_active = False
        self._reap()

    def report(self):
        self._reap(wait=True)
        return {"runtime_version": VERSION, "schedule_policy": self.schedule,
                "asynchronous": self.asynchronous, "frequency_window": self.history.window,
                "hbf_forward_calls": self.calls, "hbf_consumer_calls": self.consumers,
                "qweight_physical_peak_bytes": self.peak["qweight"], "lut_physical_peak_bytes": self.peak["lut"],
                "lut_budget_bytes": self.lut_budget,
                **({"history_before_sha256": self.history_before, "frequencies": self.frequencies,
                     "history_after_sha256": self.history.digest(), "stages": self.stage_audits} if self.audit else {})}

    def reset_cache(self):
        for kind, cache in (("qweight", self.qcache), ("lut", self.lcache)):
            for key in list(cache):
                self._drop(kind, key)
        self._reap(wait=True)
        self.stage = None
        self.peak = dict(self.live)
