from pathlib import Path
import re
import unicodedata
import numpy as np
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

SHEET_NAME = "Câu trả lời biểu mẫu 1"


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

# Columns used as actual RAG evidence.
RAG_COLUMNS = [
    "procedure_name",
    "field",
    "submission_method",
    "required_documents",
    "processing_time",
    "fee",
    "location",
    "notes",
]

# Metadata kept for audit / downstream use, but not embedded into RAG text.
METADATA_COLUMNS = [
    "form_link",
    "status",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value) -> str:
    """Clean one text value while preserving meaningful line breaks."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Unicode / invisible characters
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove whitespace around line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse horizontal whitespace only.
    # Meaningful line breaks in required_documents remain intact.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_key(value) -> str:
    """Create a stable accent-insensitive key for matching / grouping."""

    text = clean_text(value)

    if not text:
        return ""

    text = text.lower()
    text = text.replace("đ", "d")

    # Remove Vietnamese diacritics.
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )
    text = unicodedata.normalize("NFC", text)

    # Normalize punctuation to spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def procedure_key(value) -> str:
    """Normalize procedure names for matching without changing display names."""

    key = normalize_key(value)

    # These punctuation/format differences should not create new procedures.
    key = key.rstrip(" ._-:")
    key = re.sub(r"\s+", " ", key).strip()

    return key


# ============================================================
# FIELD NORMALIZATION
# ============================================================

NO_FEE_VALUES = {
    "khong",
    "khong dong",
    "khong co",
    "0",
    "0 dong",
}

FEE_EXACT_MAP = {
    "khong thu phi": "Không thu phí",
    "theo quy dinh": "Theo quy định",
}

LOCATION_REPLACEMENTS = {
    # Confirmed typo in source data.
    "ch1inh": "chính",
}

LOCATION_ALIASES = {
    "ttpvhcc": (
        "Trung tâm Phục vụ hành chính công "
        "phường Tăng Nhơn Phú"
    ),
    "ttpvhcc phuong tang nhon phu": (
        "Trung tâm Phục vụ hành chính công "
        "phường Tăng Nhơn Phú"
    ),
}


def normalize_fee(value) -> str:
    """Normalize obvious no-fee variants while preserving detailed legal fees."""

    text = clean_text(value)

    if not text:
        return ""

    key = normalize_key(text)

    # Obvious no-fee variants.
    if key in NO_FEE_VALUES:
        return "Không thu phí"

    # Exact known phrases.
    if key in FEE_EXACT_MAP:
        return FEE_EXACT_MAP[key]

    # Preserve detailed fee information exactly after whitespace cleanup.
    return text


def normalize_location(value) -> str:
    """Normalize location text without inventing or merging distinct offices."""

    text = clean_text(value)

    if not text:
        return ""

    # Fix only confirmed source typo(s).
    for wrong, correct in LOCATION_REPLACEMENTS.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Expand only explicitly known abbreviations.
    key = normalize_key(text)
    if key in LOCATION_ALIASES:
        text = LOCATION_ALIASES[key]

    return text


def normalize_submission_method(value) -> str:
    """Normalize common submission-method variants without changing semantics."""

    text = clean_text(value)
    if not text:
        return ""

    key = normalize_key(text)

    aliases = {
        "truc tiep": "Trực tiếp",
        "truc tuyen": "Trực tuyến",
        "ca hai": "Cả hai",
        "cả hai": "Cả hai",
    }

    return aliases.get(key, text)


def normalize_field(value) -> str:
    """Normalize whitespace and leading/trailing case only; keep source wording."""

    text = clean_text(value)
    return text


def normalize_processing_time(value) -> str:
    """Normalize whitespace only; do not infer missing units or values."""

    return clean_text(value)


def normalize_notes(value) -> str:
    return clean_text(value)


def normalize_form_link(value) -> str:
    return clean_text(value)


def normalize_status(value) -> str:
    return clean_text(value)


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def empty_strings_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty / whitespace-only strings to pandas NaN."""

    df = df.copy()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def value_present(value) -> bool:
    """Return True when a cell contains usable text."""

    return pd.notna(value) and bool(str(value).strip())


# ============================================================
# DUPLICATE HANDLING
# ============================================================

def make_duplicate_key(df: pd.DataFrame) -> pd.Series:
    """Create an exact normalized-content key for safe duplicate removal."""

    parts = []
    for column in RAG_COLUMNS:
        if column == "required_documents":
            values = df[column].fillna("").astype(str).map(clean_text)
        else:
            values = df[column].fillna("").astype(str).map(normalize_key)
        parts.append(values)

    result = parts[0].astype(str)
    for values in parts[1:]:
        result = result + "||" + values.astype(str)

    return result


# ============================================================
# BUILD RAG TEXT
# ============================================================

def build_rag_text(row) -> str:
    """Build embedding text while omitting missing fields."""

    sections = []

    if value_present(row["procedure_name"]):
        sections.append(
            f"Tên thủ tục hành chính: {row['procedure_name']}"
        )

    if value_present(row["field"]):
        sections.append(
            f"Lĩnh vực: {row['field']}"
        )

    if value_present(row["submission_method"]):
        sections.append(
            f"Hình thức nộp: {row['submission_method']}"
        )

    if value_present(row["required_documents"]):
        sections.append(
            "Thành phần hồ sơ:\n"
            f"{row['required_documents']}"
        )

    if value_present(row["processing_time"]):
        sections.append(
            f"Thời gian giải quyết: {row['processing_time']}"
        )

    if value_present(row["fee"]):
        sections.append(
            f"Lệ phí: {row['fee']}"
        )

    if value_present(row["location"]):
        sections.append(
            "Địa điểm tiếp nhận hồ sơ trực tiếp: "
            f"{row['location']}"
        )

    if value_present(row["notes"]):
        sections.append(
            f"Ghi chú: {row['notes']}"
        )

    return "\n\n".join(sections).strip()


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("DATA CLEANING / PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Validate source columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in COLUMN_MAP
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns:\n" + "\n".join(missing_columns)
        )

    df = df[list(COLUMN_MAP.keys())].copy()
    df = df.rename(columns=COLUMN_MAP)

    print(f"Rows before cleaning: {len(df):,}")

    # --------------------------------------------------------
    # 2. Basic cleaning for all columns
    # --------------------------------------------------------

    for column in df.columns:
        df[column] = df[column].apply(clean_text)

    # Empty strings become NaN.
    df = empty_strings_to_nan(df)

    # --------------------------------------------------------
    # 3. Remove completely empty rows
    # --------------------------------------------------------

    empty_rows = df.isna().all(axis=1)
    removed_empty = int(empty_rows.sum())

    if removed_empty:
        print(f"Completely empty rows removed: {removed_empty}")

    df = df.loc[~empty_rows].copy()

    # --------------------------------------------------------
    # 4. Remove rows without procedure name
    # --------------------------------------------------------

    missing_procedure = df["procedure_name"].isna()
    removed_missing_name = int(missing_procedure.sum())

    if removed_missing_name:
        print(f"Rows without procedure name removed: {removed_missing_name}")

    df = df.loc[~missing_procedure].copy()

    # --------------------------------------------------------
    # 5. Normalize each field
    # --------------------------------------------------------

    df["procedure_name"] = df["procedure_name"].apply(clean_text)
    df["procedure_key"] = df["procedure_name"].apply(procedure_key)

    df["field"] = df["field"].apply(normalize_field)
    df["submission_method"] = df["submission_method"].apply(
        normalize_submission_method
    )
    df["required_documents"] = df["required_documents"].apply(clean_text)
    df["processing_time"] = df["processing_time"].apply(
        normalize_processing_time
    )

    # Keep fee_raw for auditability, then normalize fee used by RAG.
    df["fee_raw"] = df["fee"].apply(clean_text)
    df["fee"] = df["fee"].apply(normalize_fee)

    # Keep location_raw for auditability, then normalize location used by RAG.
    df["location_raw"] = df["location"].apply(clean_text)
    df["location"] = df["location"].apply(normalize_location)
    df["location_key"] = df["location"].apply(normalize_key)

    df["notes"] = df["notes"].apply(normalize_notes)
    df["form_link"] = df["form_link"].apply(normalize_form_link)
    df["status"] = df["status"].apply(normalize_status)

    # Convert newly empty strings back to NaN.
    df = empty_strings_to_nan(df)

    # --------------------------------------------------------
    # 6. Safe exact duplicate removal
    # --------------------------------------------------------
    # We remove only records whose normalized RAG content is identical.
    # Records with the same procedure name but different legal content
    # are deliberately kept as separate records.

    df["_duplicate_key"] = make_duplicate_key(df)

    duplicates = df["_duplicate_key"].duplicated(keep="first")
    duplicate_count = int(duplicates.sum())

    if duplicate_count:
        print(f"Exact duplicate rows removed: {duplicate_count}")

    df = df.loc[~duplicates].copy()
    df = df.drop(columns=["_duplicate_key"])

    # --------------------------------------------------------
    # 7. Stable document ID
    # --------------------------------------------------------

    df = df.reset_index(drop=True)
    df.insert(0, "document_id", range(1, len(df) + 1))

    # --------------------------------------------------------
    # 8. Build RAG text
    # --------------------------------------------------------

    df["rag_text"] = df.apply(build_rag_text, axis=1)

    # --------------------------------------------------------
    # 9. Length / quality metadata
    # --------------------------------------------------------

    df["text_length"] = df["rag_text"].fillna("").str.len()

    # Ensure empty text becomes NaN only after we have checked it.
    df.loc[df["rag_text"].fillna("").str.strip().eq(""), "rag_text"] = np.nan

    print()
    print(f"Rows after preprocessing: {len(df):,}")
    print(
        "Unique procedure display names: "
        f"{df['procedure_name'].nunique(dropna=True):,}"
    )
    print(
        "Unique procedure keys: "
        f"{df['procedure_key'].nunique(dropna=True):,}"
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)

    print()
    print(f"Rows: {len(df):,}")

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Missing value report
    # --------------------------------------------------------

    print()
    print("Missing values by column:")
    missing_report = df.isna().sum().sort_values(ascending=False)
    for column, count in missing_report.items():
        ratio = (count / len(df) * 100) if len(df) else 0.0
        print(f"  {column:22s}: {count:3d} ({ratio:5.1f}%)")

    # --------------------------------------------------------
    # Procedure names
    # --------------------------------------------------------

    missing_names = int(df["procedure_name"].isna().sum())
    empty_keys = int(df["procedure_key"].fillna("").eq("").sum())

    print()
    print(f"Missing procedure names: {missing_names}")
    print(f"Empty procedure keys  : {empty_keys}")

    # --------------------------------------------------------
    # RAG text
    # --------------------------------------------------------

    empty_rag = int(
        df["rag_text"].isna().sum()
        + df["rag_text"].fillna("").str.strip().eq("").sum()
    )

    print(f"Empty RAG texts       : {empty_rag}")

    # --------------------------------------------------------
    # Document IDs
    # --------------------------------------------------------

    duplicate_ids = int(df["document_id"].duplicated().sum())
    print(f"Duplicate document IDs: {duplicate_ids}")

    # --------------------------------------------------------
    # Procedure groups with multiple records
    # --------------------------------------------------------

    print()
    print("Procedure groups with >1 record:")
    procedure_counts = (
        df.groupby("procedure_key", dropna=True)
        .size()
        .sort_values(ascending=False)
    )

    multi = procedure_counts[procedure_counts > 1]

    if multi.empty:
        print("  None")
    else:
        for key, count in multi.items():
            display_name = (
                df.loc[df["procedure_key"] == key, "procedure_name"]
                .dropna()
                .iloc[0]
            )
            print(f"  {count}x - {display_name}")

    # --------------------------------------------------------
    # Fee distribution
    # --------------------------------------------------------

    print()
    print("Normalized fee distribution:")
    print(
        df["fee"]
        .fillna("<NaN>")
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # --------------------------------------------------------
    # Location distribution
    # --------------------------------------------------------

    print()
    print("Normalized location distribution:")
    print(
        df["location"]
        .fillna("<NaN>")
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    lengths = df["rag_text"].fillna("").str.len()

    print()
    print("RAG text statistics:")
    print(f"  Min    : {lengths.min():,}")
    print(f"  Max    : {lengths.max():,}")
    print(f"  Mean   : {lengths.mean():.1f}")
    print(f"  Median : {lengths.median():.1f}")

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    print()
    print("QUALITY CHECK")

    if missing_names == 0:
        print("[OK] No missing procedure names")
    else:
        print(f"[ERROR] {missing_names} missing procedure names")

    if empty_keys == 0:
        print("[OK] No empty procedure keys")
    else:
        print(f"[ERROR] {empty_keys} empty procedure keys")

    if empty_rag == 0:
        print("[OK] No empty RAG texts")
    else:
        print(f"[ERROR] {empty_rag} empty RAG texts")

    if duplicate_ids == 0:
        print("[OK] Document IDs are unique")
    else:
        print(f"[ERROR] {duplicate_ids} duplicate IDs")

    # Confirm NaN is not literally embedded in RAG text.
    nan_literal_count = int(
        df["rag_text"]
        .fillna("")
        .str.lower()
        .str.contains(r"\bnan\b", regex=True)
        .sum()
    )

    if nan_literal_count == 0:
        print("[OK] No literal 'nan' in RAG text")
    else:
        print(f"[ERROR] Literal 'nan' found in {nan_literal_count} RAG texts")


# ============================================================
# SAVE
# ============================================================

def save_dataset(df: pd.DataFrame) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)
    print()
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows       : {len(df):,}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
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
        sheet_name=SHEET_NAME,
    )

    print(f"Raw rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Clean + normalize
    # --------------------------------------------------------

    cleaned_df = clean_dataset(df)

    # --------------------------------------------------------
    # 3. Validate
    # --------------------------------------------------------

    validate_dataset(cleaned_df)

    # --------------------------------------------------------
    # 4. Preview
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE PROCESSED DOCUMENTS")
    print("=" * 70)

    for _, row in cleaned_df.iterrows():
        print()
        print("-" * 70)
        print(f"Document ID: {row['document_id']}")
        print(f"Procedure   : {row['procedure_name']}")
        print(f"Procedure key: {row['procedure_key']}")
        print()
        print("RAG TEXT:")
        print(row["rag_text"] if pd.notna(row["rag_text"]) else "<NaN>")

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    save_dataset(cleaned_df)

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
