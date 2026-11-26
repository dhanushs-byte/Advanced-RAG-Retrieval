from __future__ import annotations

import csv
import time
from pathlib import Path

from src.core import (
    bm25_search, vector_search, rrf, is_hit, hit_rate,
    get_chunk_map, load_json
)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_hybrid(chunks, model, embeddings, bm25, questions, output_dir: Path):
    chunk_map = get_chunk_map(chunks)
    rows = []
    weight_summary = {0.3: [], 0.5: [], 0.7: []}
    bm_hits, vec_hits = [], []

    for item in questions:
        q = item["question"]
        b = bm25_search(q, bm25, chunks, 10)
        v = vector_search(q, model, embeddings, chunks, 10)
        bh = is_hit(b[:3], chunk_map, item["expected_terms"])
        vh = is_hit(v[:3], chunk_map, item["expected_terms"])
        bm_hits.append(bh)
        vec_hits.append(vh)

        row = {
            "question_id": item["id"],
            "type": item["type"],
            "question": q,
            "BM25_hit": bh,
            "Vector_hit": vh
        }
        for weight in (0.3, 0.5, 0.7):
            # weight is assigned to BM25; (1-weight) to vector.
            fused = rrf([b, v], weights=[weight, 1.0 - weight])
            ids = [x[0] for x in fused[:3]]
            hit = is_hit(ids, chunk_map, item["expected_terms"])
            row[f"Hybrid_{weight:.1f}_hit"] = hit
            weight_summary[weight].append(hit)
        rows.append(row)

    write_csv(output_dir / "task1_task2_results.csv", rows)
    print("\nTASK 1: BM25 vs VECTOR (Hit@3)")
    print(f"BM25  : {hit_rate(bm_hits):.2f}")
    print(f"Vector: {hit_rate(vec_hits):.2f}")
    print("\nTASK 2: HYBRID RRF WEIGHT EXPERIMENT")
    for weight, hits in weight_summary.items():
        print(f"BM25 weight={weight:.1f}, Vector weight={1-weight:.1f}: Hit@3={hit_rate(hits):.2f}")

    best_weight = max(weight_summary, key=lambda w: hit_rate(weight_summary[w]))
    print(f"Suggested default BM25 weight: {best_weight:.1f} "
          "(selected by measured Hit@3 on this evaluation set).")
    return rows, best_weight


def evaluate_rewriting(chunks, model, embeddings, bm25, vague_questions,
                       llm, output_dir: Path):
    chunk_map = get_chunk_map(chunks)
    rows = []
    raw_hits, rewrite_hits = [], []
    total_latency = 0.0

    for item in vague_questions:
        question = item["question"]
        expected = item["expected_terms"]

        t0 = time.perf_counter()
        raw_ids = vector_search(question, model, embeddings, chunks, 3)
        raw_elapsed = time.perf_counter() - t0
        raw_hit = is_hit(raw_ids, chunk_map, expected)

        cost_before = llm.estimated_cost
        t1 = time.perf_counter()
        alternatives = llm.rewrite(question)
        queries = [question] + alternatives[:3]
        rankings = [vector_search(q, model, embeddings, chunks, 10) for q in queries]
        fused = rrf(rankings, k=60)
        rewrite_ids = [x[0] for x in fused[:3]]
        rewrite_elapsed = time.perf_counter() - t1
        rewrite_cost = llm.estimated_cost - cost_before
        rewrite_hit = is_hit(rewrite_ids, chunk_map, expected)

        raw_hits.append(raw_hit)
        rewrite_hits.append(rewrite_hit)
        total_latency += rewrite_elapsed

        rows.append({
            "question_id": item["id"],
            "question": question,
            "rewrites": " || ".join(alternatives),
            "raw_hit_at_3": raw_hit,
            "rewrite_hit_at_3": rewrite_hit,
            "raw_latency_seconds": round(raw_elapsed, 4),
            "rewrite_latency_seconds": round(rewrite_elapsed, 4),
            "rewrite_estimated_cost_usd": round(rewrite_cost, 10),
            "extra_latency_seconds": round(rewrite_elapsed - raw_elapsed, 4)
        })

    write_csv(output_dir / "task3_query_rewriting_results.csv", rows)
    print("\nTASK 3: QUERY REWRITING")
    print(f"Hit@3 without rewriting: {hit_rate(raw_hits):.2f}")
    print(f"Hit@3 with rewriting   : {hit_rate(rewrite_hits):.2f}")
    print(f"Average extra latency  : {sum(r['extra_latency_seconds'] for r in rows)/len(rows):.4f}s")
    print(f"Average rewrite cost   : ${sum(r['rewrite_estimated_cost_usd'] for r in rows)/len(rows):.8f} per question")
    return rows


def evaluate_hyde(chunks, model, embeddings, bm25, vague_questions,
                  llm, output_dir: Path):
    chunk_map = get_chunk_map(chunks)
    rows = []
    raw_hits, rewrite_hits, hyde_hits, combined_hits = [], [], [], []

    for item in vague_questions:
        question = item["question"]
        expected = item["expected_terms"]

        t0 = time.perf_counter()
        raw_ids = vector_search(question, model, embeddings, chunks, 3)
        raw_time = time.perf_counter() - t0
        raw_hit = is_hit(raw_ids, chunk_map, expected)

        rewrite_cost_before = llm.estimated_cost
        rewrite_t0 = time.perf_counter()
        rewrites = llm.rewrite(question)
        rewrite_rankings = [
            vector_search(q, model, embeddings, chunks, 10)
            for q in [question] + rewrites[:3]
        ]
        rewrite_fused = rrf(rewrite_rankings)
        rewrite_ids = [x[0] for x in rewrite_fused[:3]]
        rewrite_time = time.perf_counter() - rewrite_t0
        rewrite_cost = llm.estimated_cost - rewrite_cost_before
        rewrite_hit = is_hit(rewrite_ids, chunk_map, expected)

        hyde_cost_before = llm.estimated_cost
        t1 = time.perf_counter()
        hypothetical = llm.hyde(question)
        hyde_ids = vector_search(hypothetical, model, embeddings, chunks, 3)
        hyde_time = time.perf_counter() - t1
        hyde_cost = llm.estimated_cost - hyde_cost_before
        hyde_hit = is_hit(hyde_ids, chunk_map, expected)

        # Bonus: combine original, rewritten queries and HyDE retrieval with RRF.
        combined_t0 = time.perf_counter()
        hyde_rankings = [
            vector_search(q, model, embeddings, chunks, 10)
            for q in [question] + rewrites[:3] + [hypothetical]
        ]
        combined = rrf(hyde_rankings, k=60)
        combined_ids = [x[0] for x in combined[:3]]
        combined_time = time.perf_counter() - combined_t0
        combined_hit = is_hit(combined_ids, chunk_map, expected)

        raw_hits.append(raw_hit)
        rewrite_hits.append(rewrite_hit)
        hyde_hits.append(hyde_hit)
        combined_hits.append(combined_hit)

        rows.append({
            "question_id": item["id"],
            "question": question,
            "hypothetical_answer": hypothetical,
            "raw_hit_at_3": raw_hit,
            "rewrite_hit_at_3": rewrite_hit,
            "hyde_hit_at_3": hyde_hit,
            "combined_rrf_hit_at_3": combined_hit,
            "raw_latency_seconds": round(raw_time, 4),
            "rewrite_latency_seconds": round(rewrite_time, 4),
            "hyde_latency_seconds": round(hyde_time, 4),
            "combined_latency_seconds": round(combined_time, 4),
            "rewrite_estimated_cost_usd": round(rewrite_cost, 10),
            "hyde_estimated_cost_usd": round(hyde_cost, 10),
            "combined_estimated_cost_usd": round(rewrite_cost + hyde_cost, 10)
        })

    write_csv(output_dir / "task4_hyde_results.csv", rows)
    print("\nTASK 4: RAW vs QUERY REWRITING vs HyDE")
    print(f"Raw question Hit@3 : {hit_rate(raw_hits):.2f}")
    print(f"Rewriting Hit@3    : {hit_rate(rewrite_hits):.2f}")
    print(f"HyDE Hit@3         : {hit_rate(hyde_hits):.2f}")
    print(f"Combined RRF Hit@3 : {hit_rate(combined_hits):.2f}")
    print("\nSUBMISSION SUMMARY (average per question)")
    print("Strategy                 | Hit@3 | Latency (s) | Estimated cost (USD)")
    print(f"Raw question             | {hit_rate(raw_hits):.2f}  | {sum(r['raw_latency_seconds'] for r in rows)/len(rows):.4f}      | 0.00000000")
    print(f"Query rewriting          | {hit_rate(rewrite_hits):.2f}  | {sum(r['rewrite_latency_seconds'] for r in rows)/len(rows):.4f}      | {sum(r['rewrite_estimated_cost_usd'] for r in rows)/len(rows):.8f}")
    print(f"HyDE                     | {hit_rate(hyde_hits):.2f}  | {sum(r['hyde_latency_seconds'] for r in rows)/len(rows):.4f}      | {sum(r['hyde_estimated_cost_usd'] for r in rows)/len(rows):.8f}")
    print(f"Rewriting + HyDE + RRF   | {hit_rate(combined_hits):.2f}  | {sum(r['combined_latency_seconds'] for r in rows)/len(rows):.4f}      | {sum(r['combined_estimated_cost_usd'] for r in rows)/len(rows):.8f}")
    helped = [r for r in rows if r['hyde_hit_at_3'] and not r['raw_hit_at_3']]
    hurt = [r for r in rows if r['raw_hit_at_3'] and not r['hyde_hit_at_3']]
    print("\nHyDE example where it helps:")
    if helped:
        print(f"{helped[0]['question_id']} - {helped[0]['question']}")
    else:
        print("No help case observed in this run; inspect the CSV and test set.")
    print("HyDE example where it hurts:")
    if hurt:
        print(f"{hurt[0]['question_id']} - {hurt[0]['question']}")
    else:
        print("No hurt case observed in this run; inspect the CSV and test set.")
    return rows
