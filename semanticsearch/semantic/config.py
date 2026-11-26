from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    # Model and embedding artifacts
    model_path: Path = (
        PROJECT_ROOT
        / "semanticsearch"
        / "models"
        / "qwen3-embedding-0.6b-onnx"
    )
    embeddings_path: Path = (
        PROJECT_ROOT
        / "semanticsearch"
        / "embeddings"
        / "qwen3-embedding-0.6b"
        / "embeddings.npy"
    )
    passage_ids_path: Path = (
        PROJECT_ROOT
        / "semanticsearch"
        / "embeddings"
        / "qwen3-embedding-0.6b"
        / "passage_ids.json"
    )
    metadata_path: Path = (
        PROJECT_ROOT
        / "semanticsearch"
        / "embeddings"
        / "qwen3-embedding-0.6b"
        / "metadata.json"
    )

    # Inference configuration
    precision: str = "fp32"
    max_seq_length: int = 512
    intra_op_threads: int = 8
    inter_op_threads: int = 1

    # Retrieval configuration
    dimensions: int = 1024
    default_limit: int = 20

    def validate(self) -> None:
        if self.precision != "fp32":
            raise ValueError(
                "This ONNX export currently supports FP32 only. "
                "A separate FP16 export is required for FP16 inference."
            )

        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive.")

        if self.intra_op_threads < 1:
            raise ValueError("intra_op_threads must be positive.")

        if self.inter_op_threads < 1:
            raise ValueError("inter_op_threads must be positive.")

        if self.dimensions != 1024:
            raise ValueError(
                f"Expected 1024 dimensions, got {self.dimensions}."
            )

        if self.default_limit < 1:
            raise ValueError("default_limit must be positive.")

        for path in (
            self.model_path,
            self.embeddings_path,
            self.passage_ids_path,
            self.metadata_path,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required semantic artifact not found: {path}")