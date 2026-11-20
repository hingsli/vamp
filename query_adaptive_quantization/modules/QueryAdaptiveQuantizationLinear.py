import os
import torch
import torch.nn as nn

if os.name == "posix":
    try:
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            _ld = os.environ.get("LD_LIBRARY_PATH", "")
            _paths = _ld.split(":") if _ld else []
            if _torch_lib not in _paths:
                os.environ["LD_LIBRARY_PATH"] = _torch_lib + (":" + _ld if _ld else "")
    except Exception:
        pass
try:
    from query_adaptive_quantization_ext import matmul_kbit, dequant_kbit
except Exception:
    matmul_kbit, dequant_kbit = (None, None)


class QueryAdaptiveQuantizationLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        supported_bits,
        bias=True,
        precisions=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dequant_kbit is None or matmul_kbit is None:
            raise ModuleNotFoundError(
                "Please install query adaptive quantization CUDA kernel extension from modules/kernels."
            )
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError("supported_bits must be a list of integers.")
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError("Only float16 is supported for now.")
        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits
        self._lut_ready_events = {}
        self._qweight_ready_event = None
        self.register_buffer(
            "qweight",
            torch.empty(
                (max(supported_bits), out_features, in_features // 32),
                dtype=torch.int32,
                device=device,
            ),
        )
        for bit in supported_bits:
            self.register_buffer(
                f"lut{bit}",
                torch.empty((out_features, 2**bit), dtype=dtype, device=device),
            )
        if bias:
            self.register_buffer(
                "bias", torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

    def prune_precisions(self):
        bits = max(self.precisions)
        # A view keeps its original GPU storage alive even after .data is offloaded.
        self.qweight = (
            self.qweight[:bits].clone()
            if bits < self.qweight.shape[0] else self.qweight.detach()
        )
        for bit in self.supported_bits:
            if bit not in self.precisions:
                if hasattr(self, "_router_protected_precisions"):
                    if bit in self._router_protected_precisions:
                        continue
                delattr(self, f"lut{bit}")

    def set_router_protected_precisions(self, protected_precisions):
        self._router_protected_precisions = set(protected_precisions)

    def forward(self, x, **kwargs):
        if "precision" in kwargs:
            w_bits = kwargs["precision"]
        else:
            w_bits = self.precision
        return self._forward_with_precision(x, w_bits)

    def _forward_with_precision(self, x, w_bits):
        if torch.cuda.is_available() and x.is_cuda:
            on_demand_layer = getattr(self, "_on_demand_layer", None)
            manager = getattr(self, "_on_demand_manager", None)
            need_qweight = not self.qweight.is_cuda or (
                self.qweight.ndim >= 1 and int(self.qweight.shape[0]) < int(w_bits)
            )
            if getattr(manager, "requires_prefetch", False):
                lut = self._buffers.get(f"lut{w_bits}")
                if need_qweight or lut is None or not lut.is_cuda:
                    raise RuntimeError("Bounded HBF did not prepare the requested resource")
            if need_qweight and getattr(manager, "layer_prefetch", None) is not None:
                raise RuntimeError("Layer streaming did not prepare the requested module precision")
            if need_qweight:
                if on_demand_layer is not None:
                    if manager is not None and getattr(
                        manager, "qweight_offload_enabled", False
                    ):
                        try:
                            manager._ensure_qweight_budget(
                                on_demand_layer, required_bits=int(w_bits)
                            )
                        except Exception:
                            pass
                    try:
                        on_demand_layer.ensure_qweight_on_gpu(
                            required_bits=int(w_bits),
                            pin_memory=(
                                bool(getattr(manager, "qweight_pin_memory", False))
                                if manager is not None
                                else False
                            ),
                            stream=None,
                            record_event=False,
                        )
                    except Exception:
                        self.qweight = self.qweight.to(device=x.device)
                    if manager is not None and getattr(
                        manager, "qweight_offload_enabled", False
                    ):
                        try:
                            manager._touch_qweight_cache(on_demand_layer)
                            manager._evict_qweight_cache(
                                exclude_keys=[manager._qweight_key(on_demand_layer)],
                                required_bytes=0,
                            )
                        except Exception:
                            pass
                else:
                    self.qweight = self.qweight.to(device=x.device)
            elif (
                on_demand_layer is not None
                and manager is not None
                and getattr(manager, "qweight_offload_enabled", False)
            ):
                try:
                    manager._touch_qweight_cache(on_demand_layer)
                except Exception:
                    pass
            lut_name = f"lut{w_bits}"
            lut_buf = self._buffers.get(lut_name)
            if lut_buf is None or not lut_buf.is_cuda:
                if on_demand_layer is not None:
                    try:
                        on_demand_layer.load_precision_to_gpu(
                            w_bits, load_qweight=False
                        )
                    except Exception:
                        pass
                lut_buf = self._buffers.get(lut_name)
                if lut_buf is not None and (not lut_buf.is_cuda):
                    self._buffers[lut_name].data = lut_buf.to(device=x.device)
            self._wait_for_qweight()
            self._wait_for_lut(w_bits)
            if on_demand_layer is not None:
                # Prefetched buffers can be evicted while the consuming kernel is queued.
                stream = torch.cuda.current_stream(x.device)
                self.qweight.record_stream(stream)
                self._buffers[f"lut{w_bits}"].record_stream(stream)
        if x.numel() // x.shape[-1] > 8:
            weight = dequant_kbit(self.qweight, self._buffers[f"lut{w_bits}"], w_bits)
            x = torch.matmul(x, weight.T)
        else:
            x = matmul_kbit(x, self.qweight, self._buffers[f"lut{w_bits}"], w_bits)
        if self.bias is not None:
            x += self.bias
        return x

    def set_lut_ready_event(self, precision, event):
        self._lut_ready_events[int(precision)] = event

    def clear_ready_events(self):
        self._lut_ready_events.clear()
        self._qweight_ready_event = None

    def _record_wait(self, key: str) -> None:
        manager = getattr(self, "_on_demand_manager", None)
        if manager is None:
            return
        if not getattr(manager, "collect_wait_stats_enabled", False):
            return
        try:
            manager._wait_stats[key] += 1
        except Exception:
            pass

    def _wait_for_lut(self, precision):
        if not self._lut_ready_events:
            return False
        event = self._lut_ready_events.pop(int(precision), None)
        if event is None:
            return False
        self._record_wait("w_lut")
        torch.cuda.current_stream().wait_event(event)
        return True

    def set_qweight_ready_event(self, event):
        self._qweight_ready_event = event

    def _wait_for_qweight(self):
        event = self._qweight_ready_event
        if event is None:
            return False
        self._qweight_ready_event = None
        self._record_wait("w_qw")
        torch.cuda.current_stream().wait_event(event)
        return True

    def set_precision(self, precision):
        if precision not in self.precisions:
            raise RuntimeError(
                f"{self.precisions}-bit precisions are supported but {precision}-bit was specified."
            )
        self.precision = precision

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
