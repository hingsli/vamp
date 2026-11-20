"""The source distribution must not rely on the private experiment tree."""

import ast
import importlib
from pathlib import Path


def test_core_has_no_experiment_imports():
    root = Path(__file__).resolve().parents[1] / "query_adaptive_quantization"
    forbidden = {"research", "archive", "outputs", "wandb", "lm_eval"}
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([x.name for x in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name.split(".")[0] in forbidden for name in names), path


def test_package_exports_and_cli_imports():
    from query_adaptive_quantization import QueryAdaptiveQuantizationForCausalLM, QAQRuntime, load_runtime
    assert isinstance(QueryAdaptiveQuantizationForCausalLM, type)
    assert isinstance(QAQRuntime, type)
    assert callable(load_runtime)
    for module in ("cli.train", "cli.generate", "threshold_maps.labels"):
        entry = importlib.import_module(f"query_adaptive_quantization.{module}")
        assert callable(entry.main)
