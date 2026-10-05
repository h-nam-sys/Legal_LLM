from backend.services.retrieval_service import retrieve_and_rerank

test_queries = [
    "Làm thủ tục đăng ký khai sinh có mất tiền không?",
    "Hồ sơ hưởng bảo hiểm thất nghiệp gồm những gì?"
]

for query in test_queries:
    proc_id, score, meta = retrieve_and_rerank(query)
    print(f"\nQuery: {query}")
    print(f"-> Match ID: {proc_id} | Confidence: {score:.4f} | Meta: {meta}")
