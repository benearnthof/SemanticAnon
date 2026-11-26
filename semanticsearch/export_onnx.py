
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

OUTPUT_DIR = (
    Path(__file__).parent
    / "models"
    / "qwen3-embedding-0.6b-onnx"
)


def main():
    print("=" * 60)
    print("QWEN3 EMBEDDING - ONNX EXPORT")
    print("=" * 60)

    print(f"Source: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    print("\nLoading model with ONNX backend...")

    model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={
            "provider": "CPUExecutionProvider",
            "file_name": "model.onnx",
        },
    )

    model.max_seq_length = 512

    print("\nSaving exported model...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))

    print("\nExport completed.")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
