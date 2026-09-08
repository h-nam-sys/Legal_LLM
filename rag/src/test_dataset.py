from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset(
    "vohuutridung/vietnamese-legal-documents",
    "content",
    split="data",
    streaming=True,
)

print("Dataset loaded!")

# Lấy 100 document đầu tiên để kiểm tra nội dung
for i, item in enumerate(ds):
    print("=" * 70)
    print(f"DOCUMENT {i + 1}")
    print("ID:", item["id"])
    print()
    print(item["content"][:2000])

    if i >= 99:
        break