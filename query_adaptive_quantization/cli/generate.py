"""Generate a continuation with one prompt-only precision map."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-model", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if not args.prompt.strip() or args.max_new_tokens < 1:
        parser.error("Provide a nonempty prompt and a positive max-new-tokens.")
    from query_adaptive_quantization import load_runtime
    from query_adaptive_quantization.threshold_maps import BlockWeightCostModel

    runtime = load_runtime(packed_model=args.packed_model, router_artifact=args.router)
    inputs = runtime.tokenizer(args.prompt, return_tensors="pt").to(runtime.model.device)
    output = runtime.generate(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
        pad_token_id=runtime.tokenizer.pad_token_id,
    )
    continuation = output[0, inputs.input_ids.shape[1]:]
    cost = BlockWeightCostModel.from_model(runtime.model)
    print(json.dumps({
        "text": runtime.tokenizer.decode(continuation, skip_special_tokens=True),
        "precision_map": runtime.last_precision_map.tolist(),
        "weighted_bit_width": cost.effective_bit(runtime.last_precision_map),
    }, indent=2))


if __name__ == "__main__":
    main()
