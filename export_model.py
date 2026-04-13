"""Export trained policy network to ONNX for web deployment.

Run this once after training to produce docs/model.onnx, which is
served by the GitHub Pages demo.

Usage:
  python3 export_model.py

Note: weights are inlined into the single .onnx file (no sidecar .data
file) so that onnxruntime-web can load it directly from the browser.
"""

import tempfile
from pathlib import Path

import onnx
import torch

import model

DEVICE = torch.device("cpu")
MODEL_PATH = "checkpoints/model_final.pt"
ONNX_PATH = "docs/model.onnx"


def main() -> None:
  """Export model_final.pt to a self-contained ONNX file at docs/model.onnx."""
  Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)

  policy = model.PolicyNetwork().to(DEVICE)
  policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
  policy.eval()

  dummy_input = torch.zeros(1, 4, device=DEVICE)

  # Export to a temp directory so any sidecar .data file stays isolated.
  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = str(Path(tmp) / "model.onnx")
    torch.onnx.export(
        policy,
        dummy_input,
        tmp_path,
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        export_params=True,
    )
    # Reload and re-save with all tensors inlined — required for
    # onnxruntime-web, which cannot fetch external data files.
    proto = onnx.load(tmp_path)

  onnx.save(proto, ONNX_PATH, save_as_external_data=False)
  print(f"Exported model to {ONNX_PATH}")


if __name__ == "__main__":
  main()
