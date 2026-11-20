import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor
import flash1dkmeans

from .validation import validate_parent_reconstruction


_FALLBACK_NONE = 0
_FALLBACK_NONFINITE = 1
_FALLBACK_NONPOSITIVE = 2
_FALLBACK_INSUFFICIENT_SUPPORT = 3
_FALLBACK_NAMES = {
    _FALLBACK_NONFINITE: "nonfinite",
    _FALLBACK_NONPOSITIVE: "nonpositive",
    _FALLBACK_INSUFFICIENT_SUPPORT: "insufficient_support",
}
_DEFAULT_MAX_PARENT_RELATIVE_RMSE = 0.05


@numba.njit(cache=True)
def _prepare_weighted_statistics(
    sorted_X,
    sorted_weights,
    parent_cluster_count,
):
    sorted_X_fp64 = sorted_X.astype(np.float64)
    sorted_weights_fp64 = sorted_weights.astype(np.float64)
    finite_nonnegative = True
    positive_support = 0
    weight_sum = 0.0
    for index in range(len(sorted_weights_fp64)):
        value = sorted_weights_fp64[index]
        if not np.isfinite(value) or value < 0:
            finite_nonnegative = False
        elif value > 0:
            positive_support += 1
            weight_sum += value

    fallback_reason = _FALLBACK_NONE
    if not finite_nonnegative:
        fallback_reason = _FALLBACK_NONFINITE
    elif weight_sum <= 0:
        fallback_reason = _FALLBACK_NONPOSITIVE
    elif positive_support < parent_cluster_count:
        fallback_reason = _FALLBACK_INSUFFICIENT_SUPPORT

    if fallback_reason != _FALLBACK_NONE:
        weights_prefix_sum = np.arange(
            1, len(sorted_weights_fp64) + 1, dtype=np.float64
        )
        weighted_X_prefix_sum = np.cumsum(sorted_X_fp64)
        weighted_X_squared_prefix_sum = np.cumsum(sorted_X_fp64**2)
    else:
        sorted_weighted_X = sorted_X_fp64 * sorted_weights_fp64
        weights_prefix_sum = np.cumsum(sorted_weights_fp64)
        weighted_X_prefix_sum = np.cumsum(sorted_weighted_X)
        weighted_X_squared_prefix_sum = np.cumsum(
            sorted_weighted_X * sorted_X_fp64
        )
    return (
        weights_prefix_sum,
        weighted_X_prefix_sum,
        weighted_X_squared_prefix_sum,
        fallback_reason,
    )


@numba.njit(cache=True)
def _upscale_group(
    orig_centroids,
    orig_cluster_borders,
    weights,
    weighted_X_prefix_sum,
    sample_weight_prefix_sum,
    seed_bit,
    parent_bit,
):
    luts_by_bit = [orig_centroids]
    cluster_borders = orig_cluster_borders
    for i in range(seed_bit, parent_bit):
        centroids, cluster_borders = _increment_group(
            luts_by_bit[-1],
            cluster_borders,
            weights,
            weighted_X_prefix_sum,
            sample_weight_prefix_sum,
            i,
        )
        luts_by_bit.append(centroids)
    return (luts_by_bit, cluster_borders)


@numba.njit(cache=True)
def _increment_group(
    orig_centroids,
    cluster_borders,
    weights,
    weighted_X_prefix_sum,
    sample_weight_prefix_sum,
    seed_bit,
):
    new_centroids = np.empty(2 ** (seed_bit + 1), dtype=np.float32)
    new_cluster_borders = np.empty(2 ** (seed_bit + 1) + 1, dtype=np.int32)
    assert (
        len(orig_centroids) == 2**seed_bit
    ), "The number of centroids should be 2^seed_bit"
    assert (
        len(cluster_borders) == 2**seed_bit + 1
    ), "The number of cluster start indices should be 2^seed_bit + 1"
    for c in range(2**seed_bit):
        start_idx = cluster_borders[c]
        stop_idx = cluster_borders[c + 1]
        if start_idx == stop_idx:
            new_centroids[c * 2] = orig_centroids[c]
            new_centroids[c * 2 + 1] = orig_centroids[c]
            new_cluster_borders[c * 2] = start_idx
            new_cluster_borders[c * 2 + 1] = start_idx
            continue
        cluster_centers, local_cluster_borders = (
            flash1dkmeans.numba_kmeans_1d_two_cluster(
                sorted_X=weights,
                weights_prefix_sum=sample_weight_prefix_sum,
                weighted_X_prefix_sum=weighted_X_prefix_sum,
                start_idx=start_idx,
                stop_idx=stop_idx,
            )
        )
        new_centroids[c * 2] = cluster_centers[0]
        new_centroids[c * 2 + 1] = cluster_centers[1]
        new_cluster_borders[c * 2] = start_idx
        new_cluster_borders[c * 2 + 1] = local_cluster_borders[1]
    new_cluster_borders[-1] = cluster_borders[-1]
    return (new_centroids, new_cluster_borders)


@numba.njit(parallel=True, cache=True)
def _seed_and_upscale_layer(
    layer_gradients, layer_modules, seed_bit, parent_bit, group_count, random_state=None
):
    lut_by_bit_by_module = []
    parent_weights_by_modules = []
    fallback_reasons_by_module = []
    n_cluster = 2**seed_bit
    parent_cluster_count = 2**parent_bit
    for m_idx in range(len(layer_modules)):
        module_gradient = layer_gradients[m_idx]
        module_weight = layer_modules[m_idx]
        row_count = module_weight.shape[0]
        group_size = module_weight.shape[1] // group_count
        assert (
            group_size * group_count == module_weight.shape[1]
        ), f"Group count {group_count} does not divide the number of columns {module_weight.shape[1]}"
        parent_weights = np.empty(
            (row_count, group_count, group_size), dtype=np.uint8
        )
        fallback_reasons = np.zeros((row_count, group_count), dtype=np.uint8)
        lut_by_bit = []
        for bit in range(seed_bit, parent_bit + 1):
            lut_by_bit.append(
                np.empty((row_count, group_count, 2**bit), dtype=np.float32)
            )
        for r_idx in numba.prange(module_weight.shape[0]):
            for g_idx in range(group_count):
                start_col_idx = g_idx * group_size
                end_col_idx = (g_idx + 1) * group_size
                weights_np = module_weight[r_idx, start_col_idx:end_col_idx]
                weight_mask = weights_np != 0
                sample_weight = module_gradient[r_idx, start_col_idx:end_col_idx]
                sample_weight = sample_weight * weight_mask
                X = weights_np
                sorted_indices = np.argsort(X)
                sorted_X = X[sorted_indices]
                sorted_weights = sample_weight[sorted_indices]
                (
                    sorted_weights_prefix_sum,
                    sorted_weighted_X_prefix_sum,
                    sorted_weighted_X_squared_prefix_sum,
                    fallback_reason,
                ) = _prepare_weighted_statistics(
                    sorted_X,
                    sorted_weights,
                    parent_cluster_count,
                )
                fallback_reasons[r_idx, g_idx] = fallback_reason
                centroids, cluster_borders = flash1dkmeans.numba_kmeans_1d_k_cluster(
                    sorted_X=sorted_X,
                    n_clusters=n_cluster,
                    max_iter=50,
                    weights_prefix_sum=sorted_weights_prefix_sum,
                    weighted_X_prefix_sum=sorted_weighted_X_prefix_sum,
                    weighted_X_squared_prefix_sum=sorted_weighted_X_squared_prefix_sum,
                    start_idx=0,
                    stop_idx=len(sorted_X),
                    random_state=random_state,
                )
                centroids = centroids.astype(np.float32)
                lut_per_bit, parent_cluster_borders = _upscale_group(
                    centroids,
                    cluster_borders,
                    sorted_X,
                    sorted_weighted_X_prefix_sum,
                    sorted_weights_prefix_sum,
                    seed_bit,
                    parent_bit,
                )
                for k, bit in enumerate(range(seed_bit, parent_bit + 1)):
                    lut_by_bit[k][r_idx][g_idx] = lut_per_bit[k]
                labels = np.empty(group_size, dtype=np.uint8)
                for i in range(2**parent_bit):
                    labels[
                        parent_cluster_borders[i] : parent_cluster_borders[i + 1]
                    ] = i
                labels = labels[np.argsort(sorted_indices)]
                parent_weights[r_idx][g_idx] = labels
        parent_weights_by_modules.append(parent_weights)
        lut_by_bit_by_module.append(lut_by_bit)
        fallback_reasons_by_module.append(fallback_reasons)
    return (
        lut_by_bit_by_module,
        parent_weights_by_modules,
        fallback_reasons_by_module,
    )


def _get_layer_loader(analyzer, gradients):

    def layer_loader(l):
        gradient_payload = gradients[l]
        weight_payload = analyzer.get_layer_weights(l)
        gradient_layer = [
            gradient_payload[name].float().numpy()
            for name in analyzer.module_names
        ]
        model_layer = [
            weight_payload[name].float().numpy() for name in analyzer.module_names
        ]
        return (gradient_layer, model_layer)

    return layer_loader


def _summarize_fallbacks(module_names, fallback_reasons_by_module):
    modules = {}
    totals = {name: 0 for name in _FALLBACK_NAMES.values()}
    for module_name, reasons in zip(module_names, fallback_reasons_by_module):
        module_counts = {}
        for code, name in _FALLBACK_NAMES.items():
            count = int(np.count_nonzero(reasons == code))
            module_counts[name] = count
            totals[name] += count
        module_counts["total"] = int(sum(module_counts.values()))
        modules[module_name] = module_counts
    return {
        "modules": modules,
        "totals": {
            **totals,
            "total": int(sum(totals.values())),
        },
    }


def _validate_layer_outputs(
    layer_idx,
    module_names,
    layer_modules,
    luts_by_bit_by_module,
    parent_weights,
    fallback_reasons,
    max_parent_relative_rmse,
):
    module_metrics = {}
    for module_idx, module_name in enumerate(module_names):
        parent_lut = np.asarray(
            luts_by_bit_by_module[module_idx][-1], dtype=np.float16
        )
        module_metrics[module_name] = validate_parent_reconstruction(
            layer_modules[module_idx],
            parent_weights[module_idx],
            parent_lut,
            layer_index=layer_idx,
            module_name=module_name,
            max_relative_rmse=max_parent_relative_rmse,
        )
    return {
        "layer": int(layer_idx),
        "max_parent_relative_rmse": float(max_parent_relative_rmse),
        "modules": module_metrics,
        "fallbacks": _summarize_fallbacks(
            module_names,
            fallback_reasons,
        ),
    }


def _save_results(
    parent_parameters_path,
    seed_precision,
    parent_precision,
    module_names,
    luts_by_bit_by_module,
    parent_weights,
    l,
    validation_payload,
):
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {}
        for j in range(len(module_names)):
            lut_dict[module_names[j]] = luts_by_bit_by_module[j][i].astype(np.float16)
        torch.save(lut_dict, output_lut_file_name)
    parent_weight_dict = {
        module_names[j]: parent_weights[j].astype(np.uint8)
        for j in range(len(module_names))
    }
    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)
    validation_file = f"{parent_parameters_path}/validation/l{l}.json"
    os.makedirs(os.path.dirname(validation_file), exist_ok=True)
    temporary_file = validation_file + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as handle:
        json.dump(validation_payload, handle, indent=2)
    os.replace(temporary_file, validation_file)


def _get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):

    def save_results(
        luts_by_bit_by_module,
        parent_weights,
        l,
        validation_payload,
    ):
        return _save_results(
            parent_parameters_path,
            seed_precision,
            parent_precision,
            module_names,
            luts_by_bit_by_module,
            parent_weights,
            l,
            validation_payload,
        )

    return save_results


def _load_progress(
    parent_parameters_path, seed_precision, parent_precision, layer_count
):
    todo_ran = []
    processed_ran = []
    for l in range(layer_count):
        if all(
            [
                os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt")
                for bit in range(seed_precision, parent_precision + 1)
            ]
        ) and os.path.exists(
            f"{parent_parameters_path}/weights/l{l}.pt"
        ) and os.path.exists(
            f"{parent_parameters_path}/validation/l{l}.json"
        ):
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return (todo_ran, processed_ran)


def _write_quantization_summary(
    output_folder,
    *,
    layer_count,
    seed_precision,
    parent_precision,
    max_parent_relative_rmse,
):
    validation_dir = os.path.join(output_folder, "validation")
    layers = {}
    for layer_idx in range(layer_count):
        path = os.path.join(validation_dir, f"l{layer_idx}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            layers[str(layer_idx)] = json.load(handle)
    fallback_totals = {name: 0 for name in _FALLBACK_NAMES.values()}
    observed_max_rmse = 0.0
    for layer_payload in layers.values():
        for name in fallback_totals:
            fallback_totals[name] += int(
                layer_payload["fallbacks"]["totals"].get(name, 0)
            )
        for module_payload in layer_payload["modules"].values():
            observed_max_rmse = max(
                observed_max_rmse,
                float(module_payload["relative_rmse"]),
            )
    payload = {
        "schema_version": 1,
        "status": "complete" if len(layers) == layer_count else "partial",
        "seed_precision": int(seed_precision),
        "parent_precision": int(parent_precision),
        "expected_layers": int(layer_count),
        "validated_layers": len(layers),
        "max_parent_relative_rmse_limit": float(max_parent_relative_rmse),
        "observed_max_module_relative_rmse": float(observed_max_rmse),
        "fallback_totals": {
            **fallback_totals,
            "total": int(sum(fallback_totals.values())),
        },
        "layers": layers,
    }
    output_path = os.path.join(output_folder, "quantization_validation.json")
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, output_path)
    return payload


def seed_and_upscale(
    analyzer,
    gradients,
    output_folder,
    seed_precision,
    parent_precision,
    cpu_count=None,
    random_state=None,
    group_count=1,
    max_parent_relative_rmse=_DEFAULT_MAX_PARENT_RELATIVE_RMSE,
):
    assert (
        seed_precision <= parent_precision
    ), "Parent precision should be equal or higher than seed precision"
    if cpu_count is None:
        cpu_count = os.cpu_count()
    if cpu_count >= 8:
        pipelined_io = True
        io_workers = 2 if cpu_count >= 64 else 1
        numba.set_num_threads(cpu_count - io_workers)
    else:
        pipelined_io = False
        io_workers = 0
        numba.set_num_threads(cpu_count)
    logging.info(f"Using {cpu_count} cores for parallelization")
    logging.info(
        f"Seeding & Upscaling from {seed_precision}-bit to {parent_precision}-bit"
    )
    layers_to_process, completed_layers = _load_progress(
        output_folder, seed_precision, parent_precision, analyzer.num_layers
    )
    if completed_layers:
        logging.info(
            f"The following layers will be skipped as they have already been processed:\n{completed_layers}"
        )
        logging.info(
            f"To reprocess these layers, delete the corresponding files in {output_folder}"
        )
    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        _write_quantization_summary(
            output_folder,
            layer_count=analyzer.num_layers,
            seed_precision=seed_precision,
            parent_precision=parent_precision,
            max_parent_relative_rmse=max_parent_relative_rmse,
        )
        return
    logging.info(f"Quantizing layers {layers_to_process}")
    layer_loader = _get_layer_loader(analyzer, gradients)
    layer_saver = _get_saver(
        output_folder, seed_precision, parent_precision, analyzer.module_names
    )
    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            for l in tqdm(layers_to_process, desc="Quantizing layers..."):
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)
                gradient_layer, model_layer = future_load.result()
                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)
                (
                    luts_by_bit_by_module,
                    parent_weights,
                    fallback_reasons,
                ) = _seed_and_upscale_layer(
                    gradient_layer,
                    model_layer,
                    seed_precision,
                    parent_precision,
                    group_count,
                    random_state=random_state,
                )
                validation_payload = _validate_layer_outputs(
                    l,
                    analyzer.module_names,
                    model_layer,
                    luts_by_bit_by_module,
                    parent_weights,
                    fallback_reasons,
                    max_parent_relative_rmse,
                )
                layer_saver(
                    luts_by_bit_by_module,
                    parent_weights,
                    l,
                    validation_payload,
                )
            logging.info("Waiting for IO to finish...")
    else:
        for l in tqdm(layers_to_process, desc="Quantizing layers..."):
            gradient_layer, model_layer = layer_loader(l)
            (
                luts_by_bit_by_module,
                parent_weights,
                fallback_reasons,
            ) = _seed_and_upscale_layer(
                gradient_layer,
                model_layer,
                seed_precision,
                parent_precision,
                group_count,
                random_state=random_state,
            )
            validation_payload = _validate_layer_outputs(
                l,
                analyzer.module_names,
                model_layer,
                luts_by_bit_by_module,
                parent_weights,
                fallback_reasons,
                max_parent_relative_rmse,
            )
            layer_saver(
                luts_by_bit_by_module,
                parent_weights,
                l,
                validation_payload,
            )
    summary = _write_quantization_summary(
        output_folder,
        layer_count=analyzer.num_layers,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        max_parent_relative_rmse=max_parent_relative_rmse,
    )
    if summary["status"] != "complete":
        raise RuntimeError(
            "Parent quantization validation is incomplete: "
            f"{summary['validated_layers']}/{summary['expected_layers']} layers"
        )
