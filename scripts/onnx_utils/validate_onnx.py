
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from semanticsearch.onnx_embedder import ONNXEmbedder


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

MODEL_DIR = (
    "semanticsearch/models/qwen3-embedding-0.6b-onnx"
)

TEST_QUERIES = [
    "What is Žižek's concept of surplus enjoyment?",
    "How does capitalism transform social relationships?",
    "What are the implications of artificial general intelligence?",
    "Explain the relationship between Marx and commodity fetishism.",
    "How does entropy relate to information theory?",
]


def main():
    print("Loading ONNX model...")

    onnx_model = ONNXEmbedder(
        model_path=MODEL_DIR,
        num_threads=6,
        max_seq_length=512,
    )

    print("\nLoading PyTorch reference model...")

    torch_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )

    torch_model.max_seq_length = 512

    # Construct exactly the same instructed input for both models.
    from semanticsearch.onnx_embedder import QUERY_INSTRUCTION

    instructed_queries = [
        QUERY_INSTRUCTION + query
        for query in TEST_QUERIES
    ]

    print("\nRunning PyTorch inference...")

    torch_embeddings = torch_model.encode(
        instructed_queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    print("Running ONNX inference...")

    onnx_embeddings = onnx_model.encode(
        instructed_queries,
        normalize=True,
    )

    print("\nResults")
    print("-" * 65)

    similarities = []

    for i, query in enumerate(TEST_QUERIES):
        torch_vector = torch_embeddings[i]
        onnx_vector = onnx_embeddings[i]

        cosine_similarity = np.dot(
            torch_vector,
            onnx_vector,
        ) / (
            np.linalg.norm(torch_vector)
            * np.linalg.norm(onnx_vector)
        )

        max_difference = np.max(
            np.abs(torch_vector - onnx_vector)
        )

        similarities.append(cosine_similarity)

        print(f"\nQuery: {query}")
        print(f"  Cosine similarity: {cosine_similarity:.8f}")
        print(f"  Max abs difference: {max_difference:.8e}")

    print("\n" + "-" * 65)
    print(f"Mean cosine similarity: {np.mean(similarities):.8f}")
    print(f"Minimum similarity:     {np.min(similarities):.8f}")

    if np.min(similarities) > 0.999:
        print("\nPASS: ONNX embeddings closely match PyTorch.")
    else:
        print("\nWARNING: Embeddings differ. Investigate before deployment.")


if __name__ == "__main__":
    main()
