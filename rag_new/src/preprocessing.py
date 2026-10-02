from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "Tổng hợp các thủ tục hành chính (22.12) (1).xlsx",
    PROJECT_ROOT / "data" / "raw" / "administrative_procedures(5).xlsx",
    PROJECT_ROOT / "data" / "raw" / "CSDL_THU_TUC_HANH_CHINH(2).xlsx",
    PROJECT_ROOT / "data" / "raw" / "CSDL_THU_TUC_HANH_CHINH.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Tong_hop_thu_tuc_hanh_chinh.xlsx",
    Path(__file__).resolve().parent / "Tổng hợp các thủ tục hành chính (22.12) (1).xlsx",
    Path(__file__).resolve().parent / "administrative_procedures(5).xlsx",
    Path(__file__).resolve().parent / "CSDL_THU_TUC_HANH_CHINH(2).xlsx",
    Path(__file__).resolve().parent / "CSDL_THU_TUC_HANH_CHINH.xlsx",
]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PROCEDURES_OUT = OUTPUT_DIR / "procedures_canonical.csv"
DOMAIN_PROFILE_OUT = OUTPUT_DIR / "domain_profile.json"


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(x.strip() for x in text.splitlines()).strip()


def resolve_input() -> Path:
    for path in RAW_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Không tìm thấy dataset. Đã kiểm tra:\n" +
        "\n".join(str(p) for p in RAW_CANDIDATES)
    )


def normalize_col_name(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value


def find_column(df: pd.DataFrame, *names: str) -> str | None:
    mapping = {normalize_col_name(c): c for c in df.columns}
    for name in names:
        key = normalize_col_name(name)
        if key in mapping:
            return mapping[key]
    return None


def get_series(df: pd.DataFrame, *names: str) -> pd.Series:
    column = find_column(df, *names)
    if column is None:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].map(clean_text)


def build_fee_text(value: str) -> str:
    value = clean_text(value)
    if not value:
        return "Chưa có thông tin lệ phí trong dataset."
    return value


def split_items(value: str) -> list[str]:
    value = clean_text(value)
    if not value:
        return []
    parts = re.split(r"\n+|\s*[;；]\s*", value)
    return [p.strip(" -*•\t") for p in parts if p.strip(" -*•\t")]


def build_domain_profile(df: pd.DataFrame) -> dict:
    domain_values = sorted({x for x in df["domain"].map(clean_text) if x})
    submission_values = sorted({x for x in df["submission_method"].map(clean_text) if x})

    intents = [
        "required_documents",
        "fee",
        "location",
        "processing_time",
        "legal_basis",
        "forms",
        "general_information",
    ]

    return {
        "domain": "legal",
        "entity": "administrative_procedure",
        "source_type": "excel",
        "document_count": int(len(df)),
        "domain_values": domain_values,
        "submission_methods": submission_values,
        "intents": intents,
        "canonical_fields": [
            "procedure_id",
            "procedure_name",
            "domain",
            "executing_agency",
            "submission_method",
            "required_documents",
            "processing_time",
            "fee",
            "location",
            "notes",
            "forms",
            "legal_basis",
        ],
        "retrieval_policy": {
            "procedure_required_for_location": True,
            "procedure_required_for_fee": True,
            "prefer_direct_lookup_for_exact_procedure": True,
            "use_semantic_retrieval_for_natural_language": True,
            "rerank_by_intent": True,
            "answer_only_from_verified_evidence": True,
        },
    }


def main() -> None:
    raw_file = resolve_input()
    print("=" * 80)
    print("LEGAL RAG - DOMAIN ONBOARDING / PHASE 1")
    print("=" * 80)
    print(f"Input: {raw_file}")

    xls = pd.ExcelFile(raw_file)
    if not xls.sheet_names:
        raise ValueError("Dataset không có sheet.")

    # Dataset hiện tại là flat Excel: lấy sheet đầu tiên.
    source = xls.parse(xls.sheet_names[0], header=0)
    source.columns = [clean_text(c) for c in source.columns]

    procedure_name = get_series(
        source,
        "Tên thủ tục hành chính",
        "Tên thủ tục",
        "procedure_name",
        "name",
    )
    if procedure_name.eq("").all():
        raise ValueError("Không tìm thấy cột tên thủ tục hành chính.")

    canonical = pd.DataFrame(index=source.index)
    canonical["procedure_id"] = [f"proc_{i:04d}" for i in range(1, len(source) + 1)]
    canonical["procedure_name"] = procedure_name
    canonical["domain"] = get_series(source, "Lĩnh vực", "domain")
    canonical["executing_agency"] = get_series(
        source, "Cơ quan thực hiện", "Cơ quan thực hiện/tiếp nhận", "executing_agency"
    )
    canonical["submission_method"] = get_series(
        source, "Hình thức nộp", "Hình thức nộp hồ sơ", "submission_method"
    )
    canonical["required_documents"] = get_series(
        source, "Thành phần hồ sơ", "Hồ sơ", "required_documents"
    )
    canonical["processing_time"] = get_series(
        source, "Thời gian giải quyết", "Thời hạn giải quyết", "processing_time"
    )
    canonical["fee"] = get_series(source, "Lệ phí", "Phí", "fee")
    canonical["location"] = get_series(
        source,
        "Địa điểm tiếp nhận hồ sơ trực tiếp (nếu có)",
        "Địa điểm tiếp nhận hồ sơ trực tiếp",
        "Địa điểm tiếp nhận hồ sơ",
        "Nơi tiếp nhận hồ sơ",
        "location",
    )
    canonical["notes"] = get_series(source, "Ghi chú", "Mô tả", "notes")
    canonical["forms"] = get_series(source, "Biểu mẫu", "Tệp đính kèm", "forms")
    canonical["legal_basis"] = get_series(source, "Căn cứ pháp lý", "Legal basis", "legal_basis")

    # Optional fields are retained when present in the source.
    canonical["results"] = get_series(source, "Kết quả", "Kết quả thực hiện", "results")
    canonical["conditions"] = get_series(source, "Điều kiện", "Điều kiện thực hiện", "conditions")
    canonical["status"] = get_series(source, "trạng thái", "Trạng thái", "status")

    # Remove completely empty procedure rows.
    canonical = canonical[canonical["procedure_name"].str.strip() != ""].copy()
    canonical = canonical.drop_duplicates(subset=["procedure_id"])

    # Search-ready text. This is deliberately structured, not a free summary.
    def make_search_text(row: pd.Series) -> str:
        sections = [
            f"Tên thủ tục hành chính: {row.procedure_name}",
            f"Lĩnh vực: {row.domain}" if row.domain else "",
            f"Cơ quan thực hiện: {row.executing_agency}" if row.executing_agency else "",
            f"Hình thức nộp: {row.submission_method}" if row.submission_method else "",
            f"Thành phần hồ sơ:\n{row.required_documents}" if row.required_documents else "",
            f"Thời gian giải quyết: {row.processing_time}" if row.processing_time else "",
            f"Lệ phí: {row.fee}" if row.fee else "",
            f"Địa điểm tiếp nhận hồ sơ: {row.location}" if row.location else "",
            f"Điều kiện: {row.conditions}" if row.conditions else "",
            f"Kết quả: {row.results}" if row.results else "",
            f"Biểu mẫu: {row.forms}" if row.forms else "",
            f"Căn cứ pháp lý: {row.legal_basis}" if row.legal_basis else "",
            f"Ghi chú: {row.notes}" if row.notes else "",
        ]
        return "\n\n".join(x for x in sections if x).strip()

    canonical["search_text"] = canonical.apply(make_search_text, axis=1)

    # Basic onboarding statistics.
    profile = build_domain_profile(canonical)
    profile["non_empty_counts"] = {
        field: int(canonical[field].astype(str).str.strip().ne("").sum())
        for field in [
            "domain", "executing_agency", "submission_method",
            "required_documents", "processing_time", "fee", "location",
            "notes", "forms", "legal_basis", "conditions", "results",
        ]
    }
    profile["procedure_name_unique"] = int(canonical["procedure_name"].nunique())
    profile["duplicate_name_groups"] = int(
        (canonical.groupby("procedure_name")["procedure_id"].nunique() > 1).sum()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(PROCEDURES_OUT, index=False, encoding="utf-8-sig")
    DOMAIN_PROFILE_OUT.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nONBOARDING RESULT")
    print(f"  Procedures          : {len(canonical)}")
    print(f"  Unique names        : {canonical['procedure_name'].nunique()}")
    print(f"  Domains             : {canonical['domain'].nunique()}")
    print(f"  Fee populated       : {profile['non_empty_counts']['fee']}")
    print(f"  Location populated  : {profile['non_empty_counts']['location']}")
    print(f"  Legal basis         : {profile['non_empty_counts']['legal_basis']}")
    print(f"\nSaved canonical data : {PROCEDURES_OUT}")
    print(f"Saved domain profile : {DOMAIN_PROFILE_OUT}")

    print("\nSAMPLE")
    for _, row in canonical.head(3).iterrows():
        print("-" * 70)
        print(f"ID       : {row['procedure_id']}")
        print(f"Procedure: {row['procedure_name']}")
        print(f"Domain   : {row['domain']}")
        print(f"Fee      : {row['fee']}")
        print(f"Location : {row['location']}")


if __name__ == "__main__":
    main()
