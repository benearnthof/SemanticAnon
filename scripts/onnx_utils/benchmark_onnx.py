
import gc
import time
import statistics

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from semanticsearch.onnx_embedder import ONNXEmbedder, QUERY_INSTRUCTION


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIR = "semanticsearch/models/qwen3-embedding-0.6b-onnx"

NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10

QUERIES = [
    "What is Žižek's concept of surplus enjoyment?",
    "How does capitalism transform social relationships?",
    "What are the implications of artificial general intelligence?",
    "Explain the relationship between Marx and commodity fetishism.",
    "How does entropy relate to information theory?",
]


def benchmark(name, function):
    print(f"\nBenchmarking {name}...")

    for _ in range(WARMUP_ITERATIONS):
        function()

    timings = []

    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000)

    timings = np.asarray(timings)

    print(f"  Median: {np.median(timings):.2f} ms")
    print(f"  Mean:   {np.mean(timings):.2f} ms")
    print(f"  P95:    {np.percentile(timings, 95):.2f} ms")
    print(f"  Min:    {np.min(timings):.2f} ms")

    return timings


def main():
    query = QUERY_INSTRUCTION + QUERIES[0]

    print("Loading PyTorch model...")

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)

    torch_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    torch_model.max_seq_length = 512

    torch_function = lambda: torch_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    torch_results = benchmark("PyTorch FP32 / 6 threads", torch_function)

    del torch_model
    gc.collect()

    print("\nLoading ONNX models...")

    onnx_models = {}

    for threads in [4, 6, 8]:
        onnx_models[threads] = ONNXEmbedder(
            model_path=MODEL_DIR,
            num_threads=threads,
            max_seq_length=512,
        )

    onnx_results = {}

    for threads, model in onnx_models.items():
        onnx_results[threads] = benchmark(
            f"ONNX / {threads} threads",
            lambda model=model: model.encode(query),
        )

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    baseline = np.median(torch_results)

    print(f"\nPyTorch FP32 / 6 threads: {baseline:.2f} ms")

    for threads, results in onnx_results.items():
        median = np.median(results)
        speedup = baseline / median

        print(
            f"ONNX / {threads} threads: "
            f"{median:.2f} ms "
            f"({speedup:.2f}x speedup)"
        )


if __name__ == "__main__":
    main()
