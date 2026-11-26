
import os
import time
import statistics

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages "
    "from podcast transcripts that discuss the same subject, "
    "people, events, concepts, or ideas.\n"
    "Query: "
)

QUERIES = [
    "What is the relationship between capitalism and ideology?",
    "How does technological acceleration affect society?",
    "What is Žižek's concept of surplus enjoyment?",
    "How does speaker diarization work?",
    "What are the philosophical implications of artificial intelligence?",
]

MAX_SEQ_LENGTH = 512
NUM_THREADS = 6 # min(8, os.cpu_count() or 4)
NUM_WARMUP = 3
NUM_ITERATIONS = 30


def main():
    print("=" * 60)
    print("QWEN3 EMBEDDING - CPU BENCHMARK")
    print("=" * 60)

    print(f"CPU threads: {NUM_THREADS}")
    print(f"PyTorch version: {torch.__version__}")

    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(1)

    print("\nLoading model on CPU...")

    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"dtype": torch.float32},
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    load_time = time.perf_counter() - start

    print(f"Model loaded in {load_time:.2f}s")
    print(f"Device: {model.device}")
    print(f"Max sequence length: {model.max_seq_length}")

    print("\nRunning warmup...")

    for _ in range(NUM_WARMUP):
        model.encode(
            [QUERY_INSTRUCTION + QUERIES[0]],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    print("Warmup complete.")

    print("\nBenchmarking individual queries...")

    latencies = []

    for i in range(NUM_ITERATIONS):
        query = QUERIES[i % len(QUERIES)]
        formatted_query = QUERY_INSTRUCTION + query

        start = time.perf_counter()

        embedding = model.encode(
            [formatted_query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)

        if i == 0:
            print(f"Embedding dimension: {embedding.shape[0]}")
            print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Model initialization: {load_time:.2f}s")
    print(f"Iterations:           {NUM_ITERATIONS}")
    print(f"CPU threads:          {NUM_THREADS}")
    print(f"Mean latency:         {statistics.mean(latencies):.2f} ms")
    print(f"Median latency:       {statistics.median(latencies):.2f} ms")
    print(f"Min latency:          {min(latencies):.2f} ms")
    print(f"Max latency:          {max(latencies):.2f} ms")
    print(f"P95 latency:          {np.percentile(latencies, 95):.2f} ms")

    print("=" * 60)


if __name__ == "__main__":
    main()
