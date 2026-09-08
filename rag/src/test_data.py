import json

path = "data/mock_legal_data.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        print("=" * 60)
        print("ID:", data["id"])
        print("Law:", data["law_name"])
        print("Article:", data["article"])
        print("Title:", data["title"])
        print("Content:", data["content"])