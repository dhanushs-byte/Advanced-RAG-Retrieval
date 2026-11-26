from pathlib import Path
import argparse
import time

from src.core import (
    load_documents, chunk_documents, build_embeddings, build_bm25,
    LLMHelper, ROOT, load_json
)
from src.evaluation import evaluate_hybrid, evaluate_rewriting, evaluate_hyde


def main():
    parser = argparse.ArgumentParser(
        description="Week 7 Day 2 - Advanced RAG Retrieval Assessment"
    )
    parser.add_argument(
        "--task", choices=["all", "hybrid", "rewrite", "hyde", "qdrant"],
        default="all", help="Which assessment tasks to run"
    )
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("Loading Week7-Day1 source documents...")
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    print("\nLoading embedding model and creating embeddings...")
    start = time.perf_counter()
    model, embeddings = build_embeddings(chunks)
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Embedding build time: {time.perf_counter()-start:.2f}s")

    bm25 = build_bm25(chunks)
    llm = LLMHelper()

    eval_questions = load_json(ROOT / "data" / "evaluation_questions.json")
    vague_questions = load_json(ROOT / "data" / "vague_questions.json")

    if args.task in {"all", "hybrid"}:
        evaluate_hybrid(
            chunks, model, embeddings, bm25, eval_questions, output_dir
        )

    if args.task in {"all", "rewrite"}:
        evaluate_rewriting(
            chunks, model, embeddings, bm25, vague_questions, llm, output_dir
        )

    if args.task in {"all", "hyde"}:
        evaluate_hyde(
            chunks, model, embeddings, bm25, vague_questions, llm, output_dir
        )

    if args.task == "qdrant":
        try:
            from src.qdrant_hybrid_bonus import run_qdrant_hybrid_bonus
            run_qdrant_hybrid_bonus(chunks, embeddings, eval_questions, model)
        except ImportError as exc:
            raise SystemExit(
                "Qdrant bonus dependencies are missing. Run: "
                "pip install 'qdrant-client[fastembed]'"
            ) from exc

    print("\nCompleted. Check the outputs/ folder for CSV evaluation tables.")
    print("Important: Hit@3 depends on your actual documents and expected_terms.")


if __name__ == "__main__":
    main()
