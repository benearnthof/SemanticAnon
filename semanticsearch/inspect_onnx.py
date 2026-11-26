
from pathlib import Path
import onnxruntime as ort

MODEL_DIR = (
    Path(__file__).parent
    / "models"
    / "qwen3-embedding-0.6b-onnx"
)

onnx_files = list(MODEL_DIR.rglob("*.onnx"))

print("ONNX files:")
for path in onnx_files:
    print(f"  {path} ({path.stat().st_size / 1024**2:.1f} MB)")

if not onnx_files:
    raise FileNotFoundError("No ONNX model found.")

model_path = onnx_files[0]

print(f"\nInspecting: {model_path}")

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")
