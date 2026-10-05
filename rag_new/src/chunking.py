from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "procedures_canonical.csv"
DOMAIN_PROFILE_FILE = PROJECT_ROOT / "data" / "processed" / "domain_profile.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

OUTPUT_CSV = OUTPUT_DIR / "chunks.csv"
OUTPUT_JSONL = OUTPUT_DIR / "chunks.jsonl"


# Các intent chính của Legal RAG
INTENT_FIELDS = {
    "required_documents": "required_documents",
    "fee": "fee",
    "location": "location",
    "processing_time": "processing_time",
    "legal_basis": "legal_basis",
    "forms": "forms",
}


# =============================================================================
# UTILS
# =============================================================================

def clean_text(value: Any) -> str:
    """
    Chuẩn hóa text nhưng không tự ý thay đổi nội dung pháp lý.
    """
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    # Chuẩn hóa whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_search(text: str) -> str:
    """
    Chuẩn hóa nhẹ để tạo search_text.
    Không dùng để thay đổi nội dung gốc.
    """
    text = clean_text(text)

    text = text.lower()

    # Chuẩn hóa một số variation phổ biến
    text = text.replace("đăng kí", "đăng ký")
    text = text.replace("dk", "đăng ký")

    return text


def make_chunk_id(procedure_id: str, intent: str) -> str:
    """
    Ví dụ:
        proc_0002 + fee
        -> proc_0002__fee
    """
    return f"{procedure_id}__{intent}"


# =============================================================================
# CHUNK BUILDERS
# =============================================================================

def build_procedure_header(row: pd.Series) -> str:
    procedure_name = clean_text(row.get("procedure_name"))
    domain = clean_text(row.get("domain"))

    return (
        f"Tên thủ tục hành chính: {procedure_name}\n"
        f"Lĩnh vực: {domain}"
    )


def build_required_documents_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    documents = clean_text(row.get("required_documents"))

    return (
        f"{header}\n\n"
        f"INTENT: required_documents\n"
        f"Thành phần hồ sơ:\n"
        f"{documents}"
    )


def build_fee_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    fee = clean_text(row.get("fee"))

    return (
        f"{header}\n\n"
        f"INTENT: fee\n"
        f"Lệ phí:\n"
        f"{fee}"
    )


def build_location_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    location = clean_text(row.get("location"))

    return (
        f"{header}\n\n"
        f"INTENT: location\n"
        f"Địa điểm tiếp nhận hồ sơ trực tiếp:\n"
        f"{location}"
    )


def build_processing_time_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    processing_time = clean_text(row.get("processing_time"))

    return (
        f"{header}\n\n"
        f"INTENT: processing_time\n"
        f"Thời gian giải quyết:\n"
        f"{processing_time}"
    )


def build_legal_basis_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    legal_basis = clean_text(row.get("legal_basis"))

    return (
        f"{header}\n\n"
        f"INTENT: legal_basis\n"
        f"Căn cứ pháp lý:\n"
        f"{legal_basis}"
    )


def build_forms_chunk(row: pd.Series) -> str:
    header = build_procedure_header(row)
    forms = clean_text(row.get("forms"))

    return (
        f"{header}\n\n"
        f"INTENT: forms\n"
        f"Biểu mẫu:\n"
        f"{forms}"
    )


# =============================================================================
# GENERAL INFORMATION
# =============================================================================

def build_general_information_chunk(row: pd.Series) -> str:
    """
    Chunk tổng hợp những thông tin không thuộc intent chuyên biệt.
    """

    procedure_name = clean_text(row.get("procedure_name"))
    domain = clean_text(row.get("domain"))
    agency = clean_text(row.get("executing_agency"))
    method = clean_text(row.get("submission_method"))
    notes = clean_text(row.get("notes"))
    results = clean_text(row.get("results"))
    conditions = clean_text(row.get("conditions"))
    status = clean_text(row.get("status"))

    parts = [
        f"Tên thủ tục hành chính: {procedure_name}",
        f"Lĩnh vực: {domain}",
    ]

    if agency:
        parts.append(f"Cơ quan thực hiện: {agency}")

    if method:
        parts.append(f"Hình thức nộp: {method}")

    if conditions:
        parts.append(f"Điều kiện thực hiện: {conditions}")

    if results:
        parts.append(f"Kết quả thực hiện: {results}")

    if notes:
        parts.append(f"Ghi chú: {notes}")

    if status:
        parts.append(f"Trạng thái: {status}")

    return (
        "\n\n".join(parts)
        + "\n\n"
        + "INTENT: general_information"
    )


# =============================================================================
# CHUNK CREATION
# =============================================================================

def create_chunks(df: pd.DataFrame) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for _, row in df.iterrows():

        procedure_id = clean_text(row.get("procedure_id"))
        procedure_name = clean_text(row.get("procedure_name"))
        domain = clean_text(row.get("domain"))

        if not procedure_id or not procedure_name:
            continue

        # ---------------------------------------------------------------------
        # required_documents
        # ---------------------------------------------------------------------

        value = clean_text(row.get("required_documents"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "required_documents",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "required_documents",
                    "chunk_type": "required_documents",
                    "text": build_required_documents_chunk(row),
                    "search_text": normalize_for_search(
                        build_required_documents_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # fee
        # ---------------------------------------------------------------------

        value = clean_text(row.get("fee"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "fee",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "fee",
                    "chunk_type": "fee",
                    "text": build_fee_chunk(row),
                    "search_text": normalize_for_search(
                        build_fee_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # location
        # ---------------------------------------------------------------------

        value = clean_text(row.get("location"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "location",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "location",
                    "chunk_type": "location",
                    "text": build_location_chunk(row),
                    "search_text": normalize_for_search(
                        build_location_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # processing_time
        # ---------------------------------------------------------------------

        value = clean_text(row.get("processing_time"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "processing_time",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "processing_time",
                    "chunk_type": "processing_time",
                    "text": build_processing_time_chunk(row),
                    "search_text": normalize_for_search(
                        build_processing_time_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # legal_basis
        # ---------------------------------------------------------------------

        value = clean_text(row.get("legal_basis"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "legal_basis",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "legal_basis",
                    "chunk_type": "legal_basis",
                    "text": build_legal_basis_chunk(row),
                    "search_text": normalize_for_search(
                        build_legal_basis_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # forms
        # ---------------------------------------------------------------------

        value = clean_text(row.get("forms"))

        if value:
            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        procedure_id,
                        "forms",
                    ),
                    "procedure_id": procedure_id,
                    "procedure_name": procedure_name,
                    "domain": domain,
                    "intent": "forms",
                    "chunk_type": "forms",
                    "text": build_forms_chunk(row),
                    "search_text": normalize_for_search(
                        build_forms_chunk(row)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # general_information
        # ---------------------------------------------------------------------

        general_text = build_general_information_chunk(row)

        chunks.append(
            {
                "chunk_id": make_chunk_id(
                    procedure_id,
                    "general_information",
                ),
                "procedure_id": procedure_id,
                "procedure_name": procedure_name,
                "domain": domain,
                "intent": "general_information",
                "chunk_type": "general_information",
                "text": general_text,
                "search_text": normalize_for_search(general_text),
            }
        )

    return chunks


# =============================================================================
# VALIDATION
# =============================================================================

def validate_chunks(
    chunks: list[dict[str, Any]],
    procedures_count: int,
) -> None:

    print()
    print("=" * 80)
    print("CHUNKING VALIDATION")
    print("=" * 80)

    print(f"Procedures input : {procedures_count}")
    print(f"Chunks generated : {len(chunks)}")

    intent_counts: dict[str, int] = {}

    for chunk in chunks:
        intent = chunk["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print()
    print("Chunks by intent:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent:<25}: {count}")

    procedure_ids = {
        chunk["procedure_id"]
        for chunk in chunks
    }

    print()
    print(f"Procedures represented in chunks: {len(procedure_ids)}")

    missing_ids = []

    for i in range(1, procedures_count + 1):
        expected_id = f"proc_{i:04d}"

        if expected_id not in procedure_ids:
            missing_ids.append(expected_id)

    if missing_ids:
        print()
        print("WARNING - missing procedure IDs:")

        for procedure_id in missing_ids:
            print(f"  - {procedure_id}")

    else:
        print("All procedure IDs are represented.")

    # Check duplicate chunk IDs
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    duplicates = {
        chunk_id
        for chunk_id in chunk_ids
        if chunk_ids.count(chunk_id) > 1
    }

    if duplicates:
        raise ValueError(
            f"Duplicate chunk IDs detected: {duplicates}"
        )

    print("Chunk IDs: unique")


# =============================================================================
# SAVE
# =============================================================================

def save_chunks(chunks: list[dict[str, Any]]) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(chunks)

    # CSV
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # JSONL
    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("Saved:")
    print(f"  CSV   : {OUTPUT_CSV}")
    print(f"  JSONL : {OUTPUT_JSONL}")


# =============================================================================
# SAMPLE
# =============================================================================

def print_samples(chunks: list[dict[str, Any]]) -> None:

    print()
    print("=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)

    # Lấy mỗi intent một sample
    seen_intents = set()

    for chunk in chunks:

        intent = chunk["intent"]

        if intent in seen_intents:
            continue

        seen_intents.add(intent)

        print()
        print("-" * 80)
        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Procedure: {chunk['procedure_name']}")
        print(f"Intent   : {chunk['intent']}")
        print()
        print(chunk["text"])

        if len(seen_intents) >= 7:
            break


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - INTENT CHUNKING / PHASE 2")
    print("=" * 80)

    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{INPUT_FILE}"
        )

    # -------------------------------------------------------------------------
    # Load canonical dataset
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    print()
    print("CANONICAL DATASET")
    print("-" * 80)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # -------------------------------------------------------------------------
    # Validate required columns
    # -------------------------------------------------------------------------

    required_columns = {
        "procedure_id",
        "procedure_name",
        "domain",
        "required_documents",
        "processing_time",
        "fee",
        "location",
        "forms",
        "legal_basis",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing canonical columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
        )

    # -------------------------------------------------------------------------
    # Load domain profile if available
    # -------------------------------------------------------------------------

    if DOMAIN_PROFILE_FILE.exists():

        with open(
            DOMAIN_PROFILE_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            domain_profile = json.load(f)

        print(
            f"Domain profile loaded: "
            f"{DOMAIN_PROFILE_FILE}"
        )

    else:

        domain_profile = None

        print(
            "WARNING: domain_profile.json not found."
        )

    # -------------------------------------------------------------------------
    # Create chunks
    # -------------------------------------------------------------------------

    chunks = create_chunks(df)

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_chunks(
        chunks=chunks,
        procedures_count=len(df),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_chunks(chunks)

    # -------------------------------------------------------------------------
    # Samples
    # -------------------------------------------------------------------------

    print_samples(chunks)

    print()
    print("=" * 80)
    print("PHASE 2 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()