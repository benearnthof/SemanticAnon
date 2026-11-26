"""
Rough Outline

Build the passage dataset
Parse all 650 JSON files, construct overlapping passages, preserve timestamps, speaker identities, episode metadata, and original text.
Output: passages.parquet or a SQLite database.



Implement lexical search
Build SQLite FTS5, exact phrase matching, normalized text matching, and timestamp recovery.
Output: A working exact/full-text search engine.



Generate embeddings
Benchmark Qwen3-Embedding-0.6B against BGE-large-en-v1.5. Generate and persist passage embeddings.
Output: A vector index.



Implement hybrid retrieval

Combine lexical and semantic rankings using RRF. Implement exact-match prioritization and duplicate passage merging.

Output: A unified search API.



Add reranking

Evaluate a cross-encoder on the retrieved candidates. Compare its contribution using the benchmark.

Output: Improved semantic relevance.



Build the Streamlit interface

Search field, filters for speakers and episodes, timestamped results, transcript previews, and MP3 playback.

Output: A usable research interface.



Evaluate and improve

Build the benchmark, measure retrieval quality, and investigate entity-aware search and query expansion.

Output: A measurable improvement cycle.
"""