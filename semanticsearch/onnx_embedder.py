
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages from podcast "
    "transcripts that discuss the same subject, people, events, concepts, or ideas.\n"
    "Query: "
)


class ONNXEmbedder:
    """
    Direct ONNX Runtime implementation of the Qwen3 embedding model.

    Uses:
      - CPUExecutionProvider
      - Last-token pooling
      - L2 normalization
      - Explicit position_ids

    The query instruction is applied only by encode_query().
    """

    def __init__(
        self,
        model_path,
        num_threads=6,
        max_seq_length=512,
    ):
        self.model_path = Path(model_path)
        self.max_seq_length = max_seq_length

        onnx_path = self.model_path / "onnx" / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            use_fast=True,
        )

        self.tokenizer.padding_side = "left"

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = {
            item.name for item in self.session.get_inputs()
        }

        self.output_name = self.session.get_outputs()[0].name

        print("ONNX embedder initialized")
        print(f"  Model:       {self.model_path}")
        print(f"  Threads:     {num_threads}")
        print(f"  Max length:  {max_seq_length}")
        print(f"  Providers:   {self.session.get_providers()}")

    def encode(self, texts, normalize=True):
        """
        Encode a list of complete input strings.

        Each string is processed independently, avoiding padding-related
        position_ids complications. Returns a float32 NumPy matrix.
        """

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for text in texts:
            encoded = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

            sequence_length = input_ids.shape[1]

            # Inputs are unpadded, so positions are simply 0 ... seq_len - 1.
            position_ids = np.arange(
                sequence_length,
                dtype=np.int64,
            )[None, :]

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            # Only pass inputs actually required by the exported graph.
            ort_inputs = {
                name: value
                for name, value in ort_inputs.items()
                if name in self.input_names
            }

            outputs = self.session.run(
                [self.output_name],
                ort_inputs,
            )

            last_hidden_state = outputs[0]

            # Last-token pooling. There is no padding, so the last
            # sequence position is the final actual token.
            embedding = last_hidden_state[:, -1, :]

            if normalize:
                norms = np.linalg.norm(
                    embedding,
                    axis=1,
                    keepdims=True,
                )
                embedding = embedding / np.maximum(norms, 1e-12)

            embeddings.append(embedding[0])

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query):
        """Encode one search query using the established instruction."""
        return self.encode(QUERY_INSTRUCTION + query)[0]

    def encode_queries(self, queries):
        """Encode multiple search queries using the established instruction."""
        instructed = [
            QUERY_INSTRUCTION + query
            for query in queries
        ]
        return self.encode(instructed)
