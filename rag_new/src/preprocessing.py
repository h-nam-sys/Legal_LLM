from pathlib import Path
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "administrative_procedures.xlsx"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_clean.parquet"
)


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_MAP = {
    "Tên thủ tục hành chính": "procedure_name",
    "Lĩnh vực": "field",
    "Hình thức nộp": "submission_method",
    "Thành phần hồ sơ": "required_documents",
    "Thời gian giải quyết": "processing_time",
    "Lệ phí": "fee",
    "Địa điểm tiếp nhận hồ sơ trực tiếp (nếu có)": "location",
    "Ghi chú": "notes",
    "Trong trường hợp hồ sơ bao gồm các biểu mẫu vui lòng dính kèm file mẫu": "form_link",
    "trạng thái": "status",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value) -> str:
    """
    Clean one text value.

    - Convert NaN to empty string
    - Normalize whitespace
    - Preserve meaningful line breaks
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize Unicode
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces at beginning/end of lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Select useful columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in COLUMN_MAP
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing_columns)
        )

    df = df[list(COLUMN_MAP.keys())].copy()

    # Rename
    df = df.rename(columns=COLUMN_MAP)

    print()
    print(f"Rows before cleaning: {len(df):,}")

    # --------------------------------------------------------
    # 2. Clean text columns
    # --------------------------------------------------------

    for column in df.columns:
        df[column] = df[column].apply(clean_text)

    # --------------------------------------------------------
    # 3. Remove completely empty rows
    # --------------------------------------------------------

    text_columns = list(df.columns)

    empty_rows = (
        df[text_columns]
        .apply(
            lambda row: all(
                value == ""
                for value in row
            ),
            axis=1
        )
    )

    removed_empty = empty_rows.sum()

    if removed_empty:
        print(
            f"Empty rows removed: {removed_empty}"
        )

    df = df[~empty_rows].copy()

    # --------------------------------------------------------
    # 4. Remove rows without procedure name
    # --------------------------------------------------------

    missing_procedure = (
        df["procedure_name"]
        .eq("")
    )

    removed_missing_name = missing_procedure.sum()

    if removed_missing_name:
        print(
            "Rows without procedure name removed: "
            f"{removed_missing_name}"
        )

    df = df[~missing_procedure].copy()

    # --------------------------------------------------------
    # 5. Remove duplicated procedures
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "procedure_name",
            "field",
            "submission_method",
            "required_documents",
            "processing_time",
            "fee",
            "location",
        ],
        keep="first",
    )

    duplicate_count = duplicates.sum()

    if duplicate_count:
        print(
            f"Duplicate rows removed: "
            f"{duplicate_count}"
        )

    df = df[~duplicates].copy()

    # --------------------------------------------------------
    # 6. Create stable document ID
    # --------------------------------------------------------

    df.insert(
        0,
        "document_id",
        range(1, len(df) + 1)
    )

    # --------------------------------------------------------
    # 7. Create RAG text
    # --------------------------------------------------------

    df["rag_text"] = df.apply(
        build_rag_text,
        axis=1
    )

    # --------------------------------------------------------
    # 8. Length
    # --------------------------------------------------------

    df["text_length"] = (
        df["rag_text"]
        .str.len()
    )

    print()
    print(f"Rows after cleaning: {len(df):,}")

    return df


# ============================================================
# BUILD RAG TEXT
# ============================================================

def build_rag_text(row) -> str:

    sections = []

    sections.append(
        f"Tên thủ tục hành chính: "
        f"{row['procedure_name']}"
    )

    sections.append(
        f"Lĩnh vực: "
        f"{row['field']}"
    )

    sections.append(
        f"Hình thức nộp: "
        f"{row['submission_method']}"
    )

    sections.append(
        f"Thành phần hồ sơ:\n"
        f"{row['required_documents']}"
    )

    sections.append(
        f"Thời gian giải quyết: "
        f"{row['processing_time']}"
    )

    sections.append(
        f"Lệ phí: "
        f"{row['fee']}"
    )

    sections.append(
        f"Địa điểm tiếp nhận hồ sơ trực tiếp: "
        f"{row['location']}"
    )

    if row["notes"]:
        sections.append(
            f"Ghi chú: "
            f"{row['notes']}"
        )

    if row["form_link"]:
        sections.append(
            f"Biểu mẫu: "
            f"{row['form_link']}"
        )

    return "\n\n".join(
        section
        for section in sections
        if section.strip()
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame):

    print()
    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)

    print()
    print("Rows:")
    print(f"  {len(df):,}")

    print()
    print("Columns:")

    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Missing procedure names
    # --------------------------------------------------------

    missing_names = (
        df["procedure_name"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print()
    print(
        f"Missing procedure names: "
        f"{missing_names}"
    )

    # --------------------------------------------------------
    # Empty RAG text
    # --------------------------------------------------------

    empty_rag = (
        df["rag_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty RAG texts: "
        f"{empty_rag}"
    )

    # --------------------------------------------------------
    # Duplicate document IDs
    # --------------------------------------------------------

    duplicate_ids = (
        df["document_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate document IDs: "
        f"{duplicate_ids}"
    )

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    lengths = df["rag_text"].str.len()

    print()
    print("RAG text statistics:")

    print(
        f"  Min    : {lengths.min():,}"
    )

    print(
        f"  Max    : {lengths.max():,}"
    )

    print(
        f"  Mean   : {lengths.mean():.1f}"
    )

    print(
        f"  Median : {lengths.median():.1f}"
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    print()
    print("Status distribution:")

    print(
        df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Field distribution
    # --------------------------------------------------------

    print()
    print("Field distribution:")

    print(
        df["field"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # --------------------------------------------------------
    # Submission method
    # --------------------------------------------------------

    print()
    print("Submission method distribution:")

    print(
        df["submission_method"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Quality check
    # --------------------------------------------------------

    print()
    print("QUALITY CHECK")

    if missing_names == 0:
        print("[OK] No missing procedure names")
    else:
        print(
            f"[ERROR] {missing_names} missing procedure names"
        )

    if empty_rag == 0:
        print("[OK] No empty RAG texts")
    else:
        print(
            f"[ERROR] {empty_rag} empty RAG texts"
        )

    if duplicate_ids == 0:
        print("[OK] Document IDs are unique")
    else:
        print(
            f"[ERROR] {duplicate_ids} duplicate IDs"
        )


# ============================================================
# SAVE
# ============================================================

def save_dataset(df: pd.DataFrame):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print()
    print(f"Output file:")
    print(OUTPUT_FILE)

    print()
    print(
        f"Rows: {len(df):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    print()
    print("Loading Excel...")

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="Câu trả lời biểu mẫu 1"
    )

    print(
        f"Raw rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    cleaned_df = clean_dataset(df)

    # --------------------------------------------------------
    # 3. Validate
    # --------------------------------------------------------

    validate_dataset(
        cleaned_df
    )

    # --------------------------------------------------------
    # 4. Preview
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE PROCESSED DOCUMENTS")
    print("=" * 70)

    for _, row in cleaned_df.head(5).iterrows():

        print()
        print("-" * 70)

        print(
            f"Document ID: "
            f"{row['document_id']}"
        )

        print(
            f"Procedure: "
            f"{row['procedure_name']}"
        )

        print()
        print("RAG TEXT:")

        print(
            row["rag_text"]
        )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    save_dataset(
        cleaned_df
    )

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()