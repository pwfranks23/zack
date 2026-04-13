"""Export trained policy network to ONNX for web deployment.

Run this once after training to produce docs/model.onnx, which is
served by the GitHub Pages demo.

Usage:
  python3 export_model.py
"""

from pathlib import Path

import torch

import model

DEVICE = torch.device("cpu")
MODEL_PATH = "checkpoints/model_final.pt"
ONNX_PATH = "docs/model.onnx"


def main() -> None:
  """Export model_final.pt to ONNX and write to docs/model.onnx."""
  Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)

  policy = model.PolicyNetwork().to(DEVICE)
  policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
  policy.eval()

  dummy_input = torch.zeros(1, 4, device=DEVICE)

  torch.onnx.export(
      policy,
      dummy_input,
      ONNX_PATH,
      input_names=["state"],
      output_names=["logits"],
      dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
      opset_version=17,
      export_params=True,
  )
  print(f"Exported model to {ONNX_PATH}")


if __name__ == "__main__":
  main()
