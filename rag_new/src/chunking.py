from pathlib import Path
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_clean.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_chunks.parquet"
)

# Maximum characters for one chunk
MAX_CHUNK_LENGTH = 1200

# Minimum useful length for a chunk
MIN_CHUNK_LENGTH = 20


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text) -> str:
    """
    Normalize text while preserving meaningful line breaks.

    Important:
    We DO NOT split text by newline here.
    Newlines may simply be wrapped lines from Excel.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode cleanup
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces around line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse repeated spaces inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str):
    """
    Split Vietnamese legal text into sentences as safely as possible.

    We split after:
        .
        ?
        !
        ;
        :

    followed by whitespace/newline.

    This is only a fallback for long chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    # First preserve paragraphs
    paragraphs = re.split(r"\n{2,}", text)

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Split sentence-like boundaries.
        # Do not aggressively split every newline.
        parts = re.split(
            r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỵ0-9Đ])",
            paragraph
        )

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


# ============================================================
# EXPLICIT LIST DETECTION
# ============================================================

ITEM_PATTERN = re.compile(
    r"""
    ^
    (?:
        [-•▪◦*]\s+
        |
        \d+[.)]\s+
        |
        [a-zA-ZÀ-ỴđĐ][.)]\s+
    )
    """,
    re.VERBOSE
)


def is_list_item(line: str) -> bool:
    """
    Check whether a line clearly starts a new list item.
    """

    if not line:
        return False

    return bool(ITEM_PATTERN.match(line.strip()))


def remove_list_marker(line: str) -> str:
    """
    Remove bullet / numbering from a list item.
    """

    return ITEM_PATTERN.sub("", line.strip()).strip()


# ============================================================
# REQUIRED DOCUMENTS
# ============================================================

def split_required_documents(text: str):
    """
    Split 'required_documents' into logical document items.

    IMPORTANT CHANGE FROM OLD VERSION:

    We DO NOT do:

        re.split(r"\\n+", text)

    because a newline does not necessarily mean a new
    document item.

    Example:

        Chứng minh nhân dân (hoặc căn cước công dân, hộ chiếu) của
        2 vợ chồng (01 bản photo)

    must remain ONE logical item.

    If explicit bullets/numbers exist, they are used as
    boundaries. Wrapped lines are merged back together.
    """

    text = clean_text(text)

    if not text:
        return []

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if not lines:
        return []

    # --------------------------------------------------------
    # Detect whether the section contains explicit list items
    # --------------------------------------------------------

    has_explicit_items = any(
        is_list_item(line)
        for line in lines
    )

    # --------------------------------------------------------
    # Case 1:
    # Explicit bullet / numbered list exists
    # --------------------------------------------------------

    if has_explicit_items:

        items = []
        current = ""

        for line in lines:

            if is_list_item(line):

                # Save previous item
                if current:
                    items.append(current.strip())

                # Start new item
                current = remove_list_marker(line)

            else:

                # This is a continuation line.
                # Merge it with the previous line.
                if current:
                    current = f"{current} {line}"
                else:
                    current = line

        # Save last item
        if current:
            items.append(current.strip())

        return items

    # --------------------------------------------------------
    # Case 2:
    # No explicit list.
    #
    # Preserve the whole section as ONE logical item.
    # --------------------------------------------------------

    merged = " ".join(lines)

    return [merged.strip()] if merged.strip() else []


# ============================================================
# GENERAL LONG-TEXT SPLITTER
# ============================================================

def split_long_text(text: str, max_length=MAX_CHUNK_LENGTH):
    """
    Split long text while preserving semantic boundaries.

    Priority:

        1. Paragraph
        2. Sentence
        3. Character-level fallback

    Character-level splitting is used only when absolutely
    necessary.
    """

    text = clean_text(text)

    if not text:
        return []

    # Already short enough
    if len(text) <= max_length:
        return [text]

    # --------------------------------------------------------
    # 1. Split by paragraphs
    # --------------------------------------------------------

    paragraphs = [
        p.strip()
        for p in re.split(r"\n{2,}", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # Paragraph itself fits
        if len(paragraph) <= max_length:

            if not current:
                current = paragraph

            elif len(current) + 2 + len(paragraph) <= max_length:
                current += "\n\n" + paragraph

            else:
                chunks.append(current.strip())
                current = paragraph

            continue

        # ----------------------------------------------------
        # Paragraph is too long.
        # Split it by sentences.
        # ----------------------------------------------------

        sentences = split_sentences(paragraph)

        for sentence in sentences:

            if len(sentence) <= max_length:

                if not current:
                    current = sentence

                elif len(current) + 1 + len(sentence) <= max_length:
                    current += " " + sentence

                else:
                    chunks.append(current.strip())
                    current = sentence

            else:
                # ------------------------------------------------
                # Sentence itself is too long.
                # Save current chunk first.
                # ------------------------------------------------

                if current:
                    chunks.append(current.strip())
                    current = ""

                # Character-level fallback
                for start in range(
                    0,
                    len(sentence),
                    max_length
                ):
                    piece = sentence[
                        start:start + max_length
                    ].strip()

                    if piece:
                        chunks.append(piece)

    # Save final chunk
    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# ADD CHUNK
# ============================================================

def add_chunk(
    chunks,
    document_id,
    procedure_name,
    field,
    submission_method,
    chunk_type,
    text
):
    """
    Add one chunk to the final chunk list.
    """

    text = clean_text(text)

    if not text:
        return

    if len(text) < MIN_CHUNK_LENGTH:
        return

    chunks.append(
        {
            "document_id": document_id,
            "procedure_name": procedure_name,
            "field": field,
            "submission_method": submission_method,
            "chunk_type": chunk_type,
            "text": text,
        }
    )


# ============================================================
# CREATE CHUNKS FOR ONE DOCUMENT
# ============================================================

def create_chunks_for_document(row):
    """
    Create semantic chunks for one administrative procedure.
    """

    document_id = row["document_id"]

    procedure_name = clean_text(
        row.get("procedure_name", "")
    )

    field = clean_text(
        row.get("field", "")
    )

    submission_method = clean_text(
        row.get("submission_method", "")
    )

    chunks = []

    # ========================================================
    # 1. REQUIRED DOCUMENTS
    # ========================================================

    required_documents = clean_text(
        row.get("required_documents", "")
    )

    if required_documents:

        document_items = split_required_documents(
            required_documents
        )

        for item in document_items:

            logical_text = (
                f"Tên thủ tục hành chính: "
                f"{procedure_name}\n\n"
                f"Thành phần hồ sơ:\n"
                f"{item}"
            )

            # Normally one item = one chunk.
            # If an item is too long, split semantically.
            pieces = split_long_text(
                logical_text,
                MAX_CHUNK_LENGTH
            )

            for piece in pieces:

                add_chunk(
                    chunks=chunks,
                    document_id=document_id,
                    procedure_name=procedure_name,
                    field=field,
                    submission_method=submission_method,
                    chunk_type="required_documents",
                    text=piece,
                )

    # ========================================================
    # 2. PROCESSING TIME
    # ========================================================

    processing_time = clean_text(
        row.get("processing_time", "")
    )

    if processing_time:

        text = (
            f"Tên thủ tục hành chính: "
            f"{procedure_name}\n\n"
            f"Thời gian giải quyết: "
            f"{processing_time}"
        )

        for piece in split_long_text(
            text,
            MAX_CHUNK_LENGTH
        ):

            add_chunk(
                chunks=chunks,
                document_id=document_id,
                procedure_name=procedure_name,
                field=field,
                submission_method=submission_method,
                chunk_type="processing_time",
                text=piece,
            )

    # ========================================================
    # 3. FEE
    # ========================================================

    fee = clean_text(
        row.get("fee", "")
    )

    if fee:

        text = (
            f"Tên thủ tục hành chính: "
            f"{procedure_name}\n\n"
            f"Lệ phí: "
            f"{fee}"
        )

        for piece in split_long_text(
            text,
            MAX_CHUNK_LENGTH
        ):

            add_chunk(
                chunks=chunks,
                document_id=document_id,
                procedure_name=procedure_name,
                field=field,
                submission_method=submission_method,
                chunk_type="fee",
                text=piece,
            )

    # ========================================================
    # 4. LOCATION
    # ========================================================

    location = clean_text(
        row.get("location", "")
    )

    if location:

        text = (
            f"Tên thủ tục hành chính: "
            f"{procedure_name}\n\n"
            f"Địa điểm tiếp nhận hồ sơ trực tiếp: "
            f"{location}"
        )

        for piece in split_long_text(
            text,
            MAX_CHUNK_LENGTH
        ):

            add_chunk(
                chunks=chunks,
                document_id=document_id,
                procedure_name=procedure_name,
                field=field,
                submission_method=submission_method,
                chunk_type="location",
                text=piece,
            )

    # ========================================================
    # 5. NOTES
    # ========================================================

    notes = clean_text(
        row.get("notes", "")
    )

    if notes:

        text = (
            f"Tên thủ tục hành chính: "
            f"{procedure_name}\n\n"
            f"Ghi chú:\n"
            f"{notes}"
        )

        for piece in split_long_text(
            text,
            MAX_CHUNK_LENGTH
        ):

            add_chunk(
                chunks=chunks,
                document_id=document_id,
                procedure_name=procedure_name,
                field=field,
                submission_method=submission_method,
                chunk_type="notes",
                text=piece,
            )

    # ========================================================
    # 6. FORM LINK
    # ========================================================

    form_link = clean_text(
        row.get("form_link", "")
    )

    if form_link:

        text = (
            f"Tên thủ tục hành chính: "
            f"{procedure_name}\n\n"
            f"Biểu mẫu: "
            f"{form_link}"
        )

        add_chunk(
            chunks=chunks,
            document_id=document_id,
            procedure_name=procedure_name,
            field=field,
            submission_method=submission_method,
            chunk_type="form_link",
            text=text,
        )

    # ========================================================
    # 7. GENERAL INFORMATION
    # ========================================================

    general_sections = []

    if procedure_name:
        general_sections.append(
            f"Tên thủ tục hành chính: {procedure_name}"
        )

    if field:
        general_sections.append(
            f"Lĩnh vực: {field}"
        )

    if submission_method:
        general_sections.append(
            f"Hình thức nộp: {submission_method}"
        )

    if processing_time:
        general_sections.append(
            f"Thời gian giải quyết: {processing_time}"
        )

    if fee:
        general_sections.append(
            f"Lệ phí: {fee}"
        )

    if location:
        general_sections.append(
            f"Địa điểm tiếp nhận hồ sơ trực tiếp: {location}"
        )

    if general_sections:

        general_text = "\n\n".join(
            general_sections
        )

        for piece in split_long_text(
            general_text,
            MAX_CHUNK_LENGTH
        ):

            add_chunk(
                chunks=chunks,
                document_id=document_id,
                procedure_name=procedure_name,
                field=field,
                submission_method=submission_method,
                chunk_type="general_information",
                text=piece,
            )

    return chunks


# ============================================================
# CREATE ALL CHUNKS
# ============================================================

def create_all_chunks(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SEMANTIC CHUNKING")
    print("=" * 70)

    all_chunks = []

    print()
    print(
        f"Documents to process: {len(df):,}"
    )

    for _, row in df.iterrows():

        document_chunks = create_chunks_for_document(
            row
        )

        all_chunks.extend(
            document_chunks
        )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    chunks_df = pd.DataFrame(
        all_chunks
    )

    # --------------------------------------------------------
    # Stable chunk ID
    # --------------------------------------------------------

    if not chunks_df.empty:

        chunks_df.insert(
            0,
            "chunk_id",
            [
                f"doc_{document_id}_chunk_{index}"
                for document_id, index in zip(
                    chunks_df["document_id"],
                    chunks_df.groupby(
                        "document_id"
                    ).cumcount() + 1
                )
            ]
        )

    print()
    print(
        f"Total chunks created: "
        f"{len(chunks_df):,}"
    )

    return chunks_df


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(chunks_df: pd.DataFrame):

    print()
    print("=" * 70)
    print("CHUNKING VALIDATION")
    print("=" * 70)

    if chunks_df.empty:

        print()
        print("[ERROR] No chunks were created.")

        return

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    lengths = chunks_df["text"].str.len()

    print()
    print("Chunk length statistics:")

    print(
        f"  Min      : {lengths.min():,}"
    )

    print(
        f"  Max      : {lengths.max():,}"
    )

    print(
        f"  Mean     : {lengths.mean():.1f}"
    )

    print(
        f"  Median   : {lengths.median():.1f}"
    )

    print(
        f"  > {MAX_CHUNK_LENGTH}: "
        f"{(lengths > MAX_CHUNK_LENGTH).sum()}"
    )

    # --------------------------------------------------------
    # Chunk type
    # --------------------------------------------------------

    print()
    print("Chunk type distribution:")

    print(
        chunks_df["chunk_type"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Document statistics
    # --------------------------------------------------------

    print()
    print("Document statistics:")

    print(
        f"  Documents : "
        f"{chunks_df['document_id'].nunique():,}"
    )

    print(
        f"  Chunks     : "
        f"{len(chunks_df):,}"
    )

    # --------------------------------------------------------
    # Duplicate chunk IDs
    # --------------------------------------------------------

    duplicate_ids = (
        chunks_df["chunk_id"]
        .duplicated()
        .sum()
    )

    print()
    print(
        f"Duplicate chunk IDs: "
        f"{duplicate_ids}"
    )

    # --------------------------------------------------------
    # Empty chunks
    # --------------------------------------------------------

    empty_chunks = (
        chunks_df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty chunks: "
        f"{empty_chunks}"
    )

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    print()
    print("QUALITY CHECK")

    if duplicate_ids == 0:
        print("[OK] Chunk IDs are unique")
    else:
        print(
            f"[ERROR] {duplicate_ids} duplicate chunk IDs"
        )

    if empty_chunks == 0:
        print("[OK] No empty chunks")
    else:
        print(
            f"[ERROR] {empty_chunks} empty chunks"
        )

    if (lengths <= MAX_CHUNK_LENGTH).all():
        print(
            f"[OK] All chunks <= {MAX_CHUNK_LENGTH} characters"
        )
    else:
        print(
            "[WARNING] Some chunks exceed "
            f"{MAX_CHUNK_LENGTH} characters"
        )


# ============================================================
# SAMPLE OUTPUT
# ============================================================

def show_sample_chunks(chunks_df: pd.DataFrame):

    print()
    print("=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    if chunks_df.empty:
        print("No chunks.")
        return

    # Show one sample for each chunk type
    for chunk_type in chunks_df["chunk_type"].unique():

        subset = chunks_df[
            chunks_df["chunk_type"] == chunk_type
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        print()
        print("-" * 70)

        print(
            f"Chunk ID   : {row['chunk_id']}"
        )

        print(
            f"Document ID: {row['document_id']}"
        )

        print(
            f"Procedure  : {row['procedure_name']}"
        )

        print(
            f"Type       : {row['chunk_type']}"
        )

        print(
            f"Length     : {len(row['text'])}"
        )

        print()
        print("TEXT:")
        print(row["text"])


# ============================================================
# SAVE
# ============================================================

def save_chunks(chunks_df: pd.DataFrame):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print()
    print(
        f"Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        f"Chunks: {len(chunks_df):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("SEMANTIC CHUNKING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load processed dataset
    # --------------------------------------------------------

    print()
    print("Loading processed dataset...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Documents loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # 3. Create semantic chunks
    # --------------------------------------------------------

    chunks_df = create_all_chunks(
        df
    )

    # --------------------------------------------------------
    # 4. Validate
    # --------------------------------------------------------

    validate_chunks(
        chunks_df
    )

    # --------------------------------------------------------
    # 5. Show samples
    # --------------------------------------------------------

    show_sample_chunks(
        chunks_df
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    save_chunks(
        chunks_df
    )

    print()
    print("=" * 70)
    print("CHUNKING COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()