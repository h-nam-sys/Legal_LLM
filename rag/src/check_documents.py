from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset(
    "vohuutridung/vietnamese-legal-documents",
    "content",
    split="data",
    streaming=True
)

print("Dataset loaded!\n")

MAX_DOCUMENTS = 100

keywords = [
    "người lao động",
    "hợp đồng lao động",
    "đơn phương chấm dứt",
    "chấm dứt hợp đồng",
    "báo trước",
    "nghỉ việc",
]

found = []

for i, item in enumerate(ds):

    if i >= MAX_DOCUMENTS:
        break

    content = item["content"]

    content_lower = content.lower()

    matched = [
        keyword
        for keyword in keywords
        if keyword in content_lower
    ]

    if matched:
        found.append({
            "index": i + 1,
            "id": item["id"],
            "matched": matched,
            "preview": content[:1000]
        })


print("=" * 70)
print(f"SCANNED: {MAX_DOCUMENTS} DOCUMENTS")
print(f"FOUND RELATED DOCUMENTS: {len(found)}")
print("=" * 70)


for doc in found:

    print("\n" + "=" * 70)
    print(f"DOCUMENT #{doc['index']}")
    print("=" * 70)

    print("ID:", doc["id"])
    print("Matched keywords:", doc["matched"])

    print("\nCONTENT:")
    print(doc["preview"])