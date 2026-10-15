"""Compare original vs prepared searches. Requires the LLM service on port 8002."""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / "backend/.env")
load_dotenv(ROOT / ".env")

from rag.domain_config import load_domain_config
from rag.query_preparation import prepare_search_queries, merge_ranked_records
from rag.retrieval_service import retrieve_and_rerank

SUITE = [
    "Làm giấy khai sinh cho con cần giấy tờ gì?",
    "dang ky ket hon can giay to gi",
    "đăg ký khai sih cho con",
    "Tôi làm mất giấy chứng nhận đăng ký hộ kinh doanh, xin cấp lại thế nào?",
    "Tôi muốn đăng ký kết hôn với người nước ngoài",
    "Cho tôi biết hồ sơ đăng ký khai sinh và đăng ký kết hôn",
    "Tôi muốn đăng ký đất đai và thành lập hộ kinh doanh, mỗi thủ tục cần gì?",
    "Cách nấu phở bò ngon tại nhà?",
    "Xin chào, bạn khỏe không?",
    "Tôi muốn đăng ký, cần giấy tờ gì?",
]


def summarize(records, config, cutoff, limit):
    selected = [r for r in records if r[1] >= cutoff][:limit]
    print(f"Would send {len(selected)} record(s) at cutoff {cutoff:g}")
    for i, (record_id, score, metadata) in enumerate(records[:5], 1):
        status = "SEND" if any(str(r[0]) == str(record_id) for r in selected) else "SKIP"
        print(f"{i}. [{status}] {score:.9f}  {metadata.get(config.primary_entity, record_id)}")
    return [{"id": str(record_id), "score": score,
             "title": metadata.get(config.primary_entity, "")}
            for record_id, score, metadata in selected]


async def compare(query, search_query, args):
    config = load_domain_config(args.domain)
    # Diagnose all candidates, including scores below the production cutoff.
    diagnostic = replace(config, profile={**config.profile, "minimum_score": 0.0})
    async def search(phrase):
        return await asyncio.to_thread(retrieve_and_rerank, phrase, top_k=args.candidates,
                                       candidate_k=args.candidates, config=diagnostic)
    original = await search(search_query)
    started = time.perf_counter()
    prepared = await prepare_search_queries(query, search_query, config, force=True)
    preparation_seconds = time.perf_counter() - started
    groups = []
    for phrase in prepared.queries:
        groups.append(original if phrase == search_query else await search(phrase))
    merged = merge_ranked_records(groups, args.candidates * len(groups))
    print(f"\nQUESTION: {query}")
    print(f"PREPARATION: {prepared.status} ({preparation_seconds:.2f}s)")
    print("SEARCH PHRASES:", json.dumps(prepared.queries, ensure_ascii=False))
    print("\nORIGINAL QUERY")
    before = summarize(original, config, args.threshold, args.results)
    print("\nPREPARED QUERY/QUERIES")
    after = summarize(merged, config, args.threshold, args.results)
    print("Check that phrases preserve your meaning and that BOTH intended procedures survive.")
    return {"question": query, "search_query": search_query, "threshold": args.threshold,
            "preparation_status": prepared.status, "phrases": prepared.queries,
            "preparation_seconds": preparation_seconds, "before": before, "after": after,
            "prepared_top_scores": [{"id": str(r[0]), "score": r[1],
                                     "title": r[2].get(config.primary_entity, "")} for r in merged[:5]]}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--search-query", help="Optional resolved context for a follow-up")
    parser.add_argument("--suite", action="store_true", help="Run the supplied ten diagnostic prompts")
    parser.add_argument("--domain")
    parser.add_argument("--threshold", type=float, default=0.5, help="Diagnostic only, not a production override")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--results", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report file")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 1 <= args.results <= args.candidates:
        parser.error("Require 0 <= threshold <= 1 and 1 <= results <= candidates")
    if args.suite and args.query:
        parser.error("Choose either --suite or --query")
    reports = []
    if args.suite:
        for query in SUITE:
            reports.append(await compare(query, query, args))
    elif args.query:
        reports.append(await compare(args.query, args.search_query or args.query, args))
    else:
        print("LLM service must be running. Enter q to quit; first search loads RAG models.")
        while True:
            query = input("\nQuestion: ").strip()
            if query.casefold() in {"q", "quit", "exit"}:
                break
            if query:
                reports.append(await compare(query, query, args))
    if args.output:
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
