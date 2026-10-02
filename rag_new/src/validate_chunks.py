from __future__ import annotations

from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.csv"
)

MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 2500


EXPECTED_INTENTS = {
    "required_documents",
    "fee",
    "location",
    "processing_time",
    "legal_basis",
    "general_information",
}


# =============================================================================
# UTILS
# =============================================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_columns(df: pd.DataFrame) -> list[str]:

    required_columns = {
        "chunk_id",
        "procedure_id",
        "procedure_name",
        "domain",
        "intent",
        "chunk_type",
        "text",
        "search_text",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    return missing


def validate_duplicate_chunk_ids(
    df: pd.DataFrame,
) -> list[str]:

    duplicates = (
        df[df["chunk_id"].duplicated(keep=False)]
        ["chunk_id"]
        .unique()
        .tolist()
    )

    return duplicates


def validate_empty_fields(
    df: pd.DataFrame,
) -> list[dict]:

    problems = []

    for index, row in df.iterrows():

        for column in [
            "chunk_id",
            "procedure_id",
            "procedure_name",
            "intent",
            "text",
            "search_text",
        ]:

            value = clean_text(row[column])

            if not value:

                problems.append(
                    {
                        "row": index + 2,
                        "column": column,
                        "procedure_id": row.get(
                            "procedure_id",
                            "",
                        ),
                        "chunk_id": row.get(
                            "chunk_id",
                            "",
                        ),
                    }
                )

    return problems


def validate_unknown_intents(
    df: pd.DataFrame,
) -> list[str]:

    intents = set(
        df["intent"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        intents - EXPECTED_INTENTS
    )


def validate_text_lengths(
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:

    too_short = []
    too_long = []

    for index, row in df.iterrows():

        text = clean_text(row["text"])

        length = len(text)

        info = {
            "row": index + 2,
            "chunk_id": row["chunk_id"],
            "procedure_id": row["procedure_id"],
            "intent": row["intent"],
            "length": length,
        }

        if length < MIN_TEXT_LENGTH:
            too_short.append(info)

        if length > MAX_TEXT_LENGTH:
            too_long.append(info)

    return too_short, too_long


def validate_intent_distribution(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["intent"]
        .value_counts()
        .sort_index()
    )


def validate_procedure_coverage(
    df: pd.DataFrame,
) -> tuple[set[str], list[str]]:

    procedure_ids = sorted(
        df["procedure_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    expected = {
        f"proc_{i:04d}"
        for i in range(1, 43)
    }

    actual = set(procedure_ids)

    missing = sorted(
        expected - actual
    )

    return actual, missing


def validate_intent_content(
    df: pd.DataFrame,
) -> list[dict]:

    """
    Kiểm tra chunk có đúng nội dung tương ứng với intent hay không.
    """

    problems = []

    intent_markers = {
        "required_documents": [
            "INTENT: required_documents",
            "Thành phần hồ sơ:",
        ],
        "fee": [
            "INTENT: fee",
            "Lệ phí:",
        ],
        "location": [
            "INTENT: location",
            "Địa điểm tiếp nhận hồ sơ trực tiếp:",
        ],
        "processing_time": [
            "INTENT: processing_time",
            "Thời gian giải quyết:",
        ],
        "legal_basis": [
            "INTENT: legal_basis",
            "Căn cứ pháp lý:",
        ],
        "general_information": [
            "INTENT: general_information",
        ],
    }

    for index, row in df.iterrows():

        intent = clean_text(row["intent"])
        text = clean_text(row["text"])

        markers = intent_markers.get(intent)

        if not markers:
            continue

        for marker in markers:

            if marker not in text:

                problems.append(
                    {
                        "row": index + 2,
                        "chunk_id": row["chunk_id"],
                        "intent": intent,
                        "missing_marker": marker,
                    }
                )

    return problems


def validate_search_text(
    df: pd.DataFrame,
) -> list[dict]:

    problems = []

    for index, row in df.iterrows():

        procedure_name = clean_text(
            row["procedure_name"]
        )

        intent = clean_text(
            row["intent"]
        )

        search_text = clean_text(
            row["search_text"]
        ).lower()

        if procedure_name.lower() not in search_text:

            problems.append(
                {
                    "row": index + 2,
                    "chunk_id": row["chunk_id"],
                    "problem": "procedure_name_missing",
                }
            )

        if intent.lower() not in search_text:

            problems.append(
                {
                    "row": index + 2,
                    "chunk_id": row["chunk_id"],
                    "problem": "intent_missing",
                }
            )

    return problems


# =============================================================================
# REPORT
# =============================================================================

def print_section(title: str):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():

    print("=" * 80)
    print("LEGAL RAG - CHUNK QUALITY VALIDATION / PHASE 2.5")
    print("=" * 80)

    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"chunks.csv not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    print_section("DATASET")

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # -------------------------------------------------------------------------
    # 1. Columns
    # -------------------------------------------------------------------------

    print_section("1. COLUMN VALIDATION")

    missing_columns = validate_columns(df)

    if missing_columns:

        print("FAIL")
        print("Missing columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise SystemExit(1)

    else:

        print("PASS")
        print("All required columns exist.")

    # -------------------------------------------------------------------------
    # 2. Duplicate chunk IDs
    # -------------------------------------------------------------------------

    print_section("2. DUPLICATE CHUNK ID")

    duplicate_ids = validate_duplicate_chunk_ids(df)

    if duplicate_ids:

        print("FAIL")

        for chunk_id in duplicate_ids:
            print(f"  - {chunk_id}")

    else:

        print("PASS")
        print("All chunk_id values are unique.")

    # -------------------------------------------------------------------------
    # 3. Empty fields
    # -------------------------------------------------------------------------

    print_section("3. EMPTY FIELD VALIDATION")

    empty_fields = validate_empty_fields(df)

    if empty_fields:

        print(f"WARNING: {len(empty_fields)} empty fields")

        for item in empty_fields[:20]:

            print(
                f"  row={item['row']} "
                f"column={item['column']} "
                f"chunk={item['chunk_id']}"
            )

    else:

        print("PASS")
        print("No required field is empty.")

    # -------------------------------------------------------------------------
    # 4. Intent
    # -------------------------------------------------------------------------

    print_section("4. INTENT VALIDATION")

    unknown_intents = validate_unknown_intents(df)

    if unknown_intents:

        print("FAIL")

        for intent in unknown_intents:
            print(f"  - {intent}")

    else:

        print("PASS")

        distribution = validate_intent_distribution(df)

        for intent, count in distribution.items():

            print(
                f"  {intent:<25}: {count}"
            )

    # -------------------------------------------------------------------------
    # 5. Procedure coverage
    # -------------------------------------------------------------------------

    print_section("5. PROCEDURE COVERAGE")

    actual_procedures, missing_procedures = (
        validate_procedure_coverage(df)
    )

    print(
        f"Procedures represented: "
        f"{len(actual_procedures)}"
    )

    if missing_procedures:

        print("FAIL")

        for procedure_id in missing_procedures:
            print(f"  Missing: {procedure_id}")

    else:

        print("PASS")
        print("All 42 procedures are represented.")

    # -------------------------------------------------------------------------
    # 6. Text length
    # -------------------------------------------------------------------------

    print_section("6. TEXT LENGTH")

    too_short, too_long = validate_text_lengths(df)

    print(
        f"Minimum allowed : {MIN_TEXT_LENGTH}"
    )

    print(
        f"Maximum allowed : {MAX_TEXT_LENGTH}"
    )

    if too_short:

        print()
        print(
            f"WARNING: {len(too_short)} "
            f"chunks are too short."
        )

        for item in too_short[:20]:

            print(
                f"  {item['chunk_id']} "
                f"length={item['length']}"
            )

    else:

        print("No chunks below minimum length.")

    if too_long:

        print()
        print(
            f"WARNING: {len(too_long)} "
            f"chunks are too long."
        )

        for item in too_long[:20]:

            print(
                f"  {item['chunk_id']} "
                f"length={item['length']}"
            )

    else:

        print("No chunks above maximum length.")

    # -------------------------------------------------------------------------
    # 7. Intent content
    # -------------------------------------------------------------------------

    print_section("7. INTENT CONTENT VALIDATION")

    intent_problems = validate_intent_content(df)

    if intent_problems:

        print(
            f"WARNING: {len(intent_problems)} "
            f"intent content problems."
        )

        for item in intent_problems[:20]:

            print(
                f"  row={item['row']} "
                f"chunk={item['chunk_id']} "
                f"missing={item['missing_marker']}"
            )

    else:

        print("PASS")
        print("Intent markers are consistent.")

    # -------------------------------------------------------------------------
    # 8. Search text
    # -------------------------------------------------------------------------

    print_section("8. SEARCH TEXT VALIDATION")

    search_problems = validate_search_text(df)

    if search_problems:

        print(
            f"WARNING: {len(search_problems)} "
            f"search_text problems."
        )

        for item in search_problems[:20]:

            print(
                f"  row={item['row']} "
                f"chunk={item['chunk_id']} "
                f"problem={item['problem']}"
            )

    else:

        print("PASS")
        print(
            "Every search_text contains "
            "procedure name and intent."
        )

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    print_section("FINAL RESULT")

    hard_fail = (
        bool(missing_columns)
        or bool(duplicate_ids)
        or bool(unknown_intents)
        or bool(missing_procedures)
    )

    if hard_fail:

        print("❌ PHASE 2.5 FAILED")
        print()
        print(
            "Do NOT continue to embedding."
        )

        raise SystemExit(1)

    else:

        print("✅ PHASE 2.5 PASSED")
        print()
        print(
            "Chunk dataset is structurally ready "
            "for the next phase."
        )

        if too_long:

            print()
            print(
                "NOTE: Some chunks are long."
            )

            print(
                "Before embedding, review the long "
                "chunks and decide whether they "
                "need semantic sub-chunking."
            )


if __name__ == "__main__":
    main()