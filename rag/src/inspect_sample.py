"""
Xem nhanh N thủ tục trong procedures_clean.csv để kiểm tra chất lượng
rag_text sau khi xử lý.

Chạy:
    python inspect_sample.py                 -> in 10 thủ tục đầu tiên
    python inspect_sample.py --random 10      -> in 10 thủ tục ngẫu nhiên
    python inspect_sample.py --proc-id 1.000080  -> in đúng 1 proc_id
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "processed" / "procedures_clean.csv"


def print_row(row: pd.Series, idx: int) -> None:
    print("=" * 100)
    print(f"[{idx}] proc_id = {row['proc_id']}")
    print("-" * 100)
    print(f"Tên          : {row['name']}")
    print(f"Cơ quan      : {row['executing_agency']}")
    print(f"Số hiệu VB   : {row['legal_codes']}")
    print(f"has_mcq      : {row['has_mcq']}")
    print("-" * 100)
    print("RAG_TEXT:")
    print(row["rag_text"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=None, help="Số dòng lấy ngẫu nhiên")
    parser.add_argument("--n", type=int, default=10, help="Số dòng in ra (mặc định 10, lấy từ đầu)")
    parser.add_argument("--proc-id", type=str, default=None, help="In đúng 1 proc_id")
    args = parser.parse_args()

    # keep_default_na=False: tránh pandas biến ô rỗng thành NaN rồi in ra
    # chữ "nan" gây hiểu lầm - trong procedures_clean.csv, ô rỗng nghĩa là
    # dữ liệu gốc không có, không phải lỗi.
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    if args.proc_id:
        subset = df[df["proc_id"] == args.proc_id]
        if subset.empty:
            print(f"Không tìm thấy proc_id = {args.proc_id}")
            return
    elif args.random:
        subset = df.sample(n=min(args.random, len(df)), random_state=None)
    else:
        subset = df.head(args.n)

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        print_row(row, i)


if __name__ == "__main__":
    main()