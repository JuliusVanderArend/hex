from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Make sure this import matches your actual file structure
from hex_nn.model import HexPolicyValueNet


class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        policy_logits, value = self.model(x)
        policy_probs = torch.softmax(policy_logits, dim=1)
        return policy_probs, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)

    parser.add_argument("--filters", type=int, default=128, help="Number of ResNet filters (match your training run)")
    parser.add_argument("--blocks", type=int, default=5, help="Number of ResNet blocks (match your training run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    print(f"Loading model with: {args.blocks} blocks, {args.filters} filters")

    model = HexPolicyValueNet(channels=args.filters, num_blocks=args.blocks)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    export_model = InferenceWrapper(model)

    dummy = torch.zeros(1, 6, 11, 11, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy,
        args.output,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={"state": {0: "batch_size"}, "policy": {0: "batch_size"}, "value": {0: "batch_size"}},
        opset_version=18,
    )
    print(f"Exported ONNX to {args.output}")


if __name__ == "__main__":
    main()