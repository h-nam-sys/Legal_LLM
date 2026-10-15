import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / "backend" / ".env")
load_dotenv(ROOT / ".env")

from rag.domain_config import load_domain_config
from rag.retrieval_service import (
    _MODEL_LOCK,
    _index,
    _reranker,
    _sigmoid,
)


def inspect_query(query, config, candidate_k, result_k):
    with _MODEL_LOCK:
        index = _index(
            str(config.vector_dir),
            config.embedding_model,
            config.revision,
        )

        candidates = index.as_retriever(
            similarity_top_k=candidate_k
        ).retrieve(query)

        if not candidates:
            print("\nNo candidates found.")
            return

        raw_scores = _reranker().compute_score(
            [[query, node.text] for node in candidates]
        )

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]

        if len(raw_scores) != len(candidates):
            raise RuntimeError("Candidate/score count mismatch")

        ranked = sorted(
            zip(candidates, raw_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

    print(f"\nQuery: {query}")
    print(f"Domain: {config.name}")
    print(f"Candidates retrieved: {len(candidates)}")
    print(f"Minimum sigmoid score: {config.minimum_score}")
    print(f"Maximum records sent: {result_k}")
    print("Reranking uses the existing indexed text, not full SQL records.\n")

    seen_ids = set()
    sent = 0

    for rank, (node, raw_score) in enumerate(ranked, start=1):
        raw_score = float(raw_score)
        score = _sigmoid(raw_score)
        record_id = node.metadata.get("id")
        title = node.metadata.get(config.primary_entity, "(no title)")

        if record_id is None:
            status = "ERROR: missing record ID"
        elif score < config.minimum_score:
            status = "REJECT: below threshold"
        elif str(record_id) in seen_ids:
            status = "SKIP: duplicate record"
        elif sent >= result_k:
            status = "SKIP: result limit reached"
        else:
            status = "SEND"
            seen_ids.add(str(record_id))
            sent += 1

        vector_score = node.score
        vector_text = (
            f"{float(vector_score):.6f}"
            if vector_score is not None
            else "N/A"
        )

        print(f"{rank:02d}. [{status}] {title}")
        print(f"    ID:      {record_id}")
        print(f"    Vector:  {vector_text}")
        print(f"    Raw:     {raw_score:.6f}")
        print(f"    Sigmoid: {score:.9f}")
        print(f"    Text:    {node.text}")
        print()

    print(f"Would send {sent} distinct record(s).")
    print("Vector scores and reranker scores are different scales.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--domain", default=None)
    parser.add_argument(
        "--candidates",
        type=int,
        default=int(os.getenv("RAG_CANDIDATE_K", "20")),
    )
    parser.add_argument(
        "--results",
        type=int,
        default=int(os.getenv("RAG_RESULT_K", "5")),
    )
    args = parser.parse_args()

    if not 1 <= args.results <= args.candidates:
        parser.error("Require 1 <= results <= candidates")

    config = load_domain_config(args.domain)

    from dataclasses import replace

    config = replace(
        config,
        profile={**config.profile, "minimum_score": 0.5},
    )

    if args.query:
        inspect_query(args.query, config, args.candidates, args.results)
        return

    print("Enter questions to inspect. Enter q to quit.")
    print("The first question loads the models; later questions reuse them.")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if query:
            inspect_query(query, config, args.candidates, args.results)


if __name__ == "__main__":
    main()
