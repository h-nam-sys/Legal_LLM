from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "administrative_procedures_clean.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "administrative_procedures_chunks.parquet"
)

MIN_CHUNK_LENGTH = 80
MAX_CHUNK_LENGTH = 1200


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean text while preserving the original meaning.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# SPLIT REQUIRED DOCUMENTS
# ============================================================

def split_required_documents(text):
    """
    Split the required-document section into logical pieces.

    Priority:
    1. Lines beginning with '-'
    2. Otherwise split by sentences/paragraphs.
    """

    text = clean_text(text)

    if not text:
        return []

    # Try bullet-list structure first
    bullets = re.split(r"\n\s*-\s*", text)

    if len(bullets) > 1:
        chunks = []

        for item in bullets:
            item = item.strip()

            if item:
                chunks.append(item)

        return chunks

    # Otherwise preserve paragraphs
    paragraphs = re.split(r"\n+", text)

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]

    return paragraphs


# ============================================================
# SPLIT LONG TEXT
# ============================================================

def split_long_text(text, max_length=MAX_CHUNK_LENGTH):
    """
    Split a long text into smaller chunks.

    Prefer sentence boundaries instead of cutting words.
    """

    text = clean_text(text)

    if len(text) <= max_length:
        return [text]

    sentences = re.split(
        r"(?<=[.!?;:])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            sentence
            if not current
            else current + " " + sentence
        )

        if len(candidate) <= max_length:
            current = candidate

        else:
            if current:
                chunks.append(current)

            # Extremely long sentence
            if len(sentence) > max_length:

                for i in range(0, len(sentence), max_length):
                    piece = sentence[i:i + max_length].strip()

                    if piece:
                        chunks.append(piece)

                current = ""

            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# CREATE CHUNKS FOR ONE DOCUMENT
# ============================================================

def create_chunks(row):
    """
    Convert one administrative procedure into RAG chunks.

    Each chunk keeps document-level metadata.
    """

    document_id = row["document_id"]
    procedure_name = clean_text(row["procedure_name"])
    field = clean_text(row["field"])
    submission_method = clean_text(row["submission_method"])
    required_documents = clean_text(row["required_documents"])
    processing_time = clean_text(row["processing_time"])
    fee = clean_text(row["fee"])
    location = clean_text(row["location"])
    notes = clean_text(row["notes"])
    form_link = clean_text(row["form_link"])
    status = clean_text(row["status"])

    chunks = []

    # --------------------------------------------------------
    # CHUNK 1: GENERAL INFORMATION
    # --------------------------------------------------------

    general_text = (
        f"Tên thủ tục hành chính: {procedure_name}\n\n"
        f"Lĩnh vực: {field}\n\n"
        f"Hình thức nộp: {submission_method}\n\n"
        f"Thời gian giải quyết: {processing_time}\n\n"
        f"Lệ phí: {fee}\n\n"
        f"Địa điểm tiếp nhận hồ sơ trực tiếp: {location}"
    )

    general_text = clean_text(general_text)

    if len(general_text) >= MIN_CHUNK_LENGTH:

        chunks.append({
            "document_id": document_id,
            "procedure_name": procedure_name,
            "field": field,
            "submission_method": submission_method,
            "chunk_type": "general_information",
            "chunk_index": len(chunks),
            "text": general_text,
            "processing_time": processing_time,
            "fee": fee,
            "location": location,
            "notes": notes,
            "form_link": form_link,
            "status": status,
        })

    # --------------------------------------------------------
    # CHUNK 2+: REQUIRED DOCUMENTS
    # --------------------------------------------------------

    document_parts = split_required_documents(
        required_documents
    )

    for part in document_parts:

        part_chunks = split_long_text(part)

        for sub_chunk in part_chunks:

            sub_chunk = clean_text(sub_chunk)

            if len(sub_chunk) < MIN_CHUNK_LENGTH:
                continue

            text = (
                f"Tên thủ tục hành chính: {procedure_name}\n\n"
                f"Thành phần hồ sơ:\n{sub_chunk}"
            )

            chunks.append({
                "document_id": document_id,
                "procedure_name": procedure_name,
                "field": field,
                "submission_method": submission_method,
                "chunk_type": "required_documents",
                "chunk_index": len(chunks),
                "text": text,
                "processing_time": processing_time,
                "fee": fee,
                "location": location,
                "notes": notes,
                "form_link": form_link,
                "status": status,
            })

    # --------------------------------------------------------
    # OPTIONAL: NOTES
    # --------------------------------------------------------

    if notes and len(notes) >= MIN_CHUNK_LENGTH:

        notes_text = (
            f"Tên thủ tục hành chính: {procedure_name}\n\n"
            f"Ghi chú: {notes}"
        )

        chunks.append({
            "document_id": document_id,
            "procedure_name": procedure_name,
            "field": field,
            "submission_method": submission_method,
            "chunk_type": "notes",
            "chunk_index": len(chunks),
            "text": notes_text,
            "processing_time": processing_time,
            "fee": fee,
            "location": location,
            "notes": notes,
            "form_link": form_link,
            "status": status,
        })

    return chunks


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(chunks_df, source_df):

    print("\n" + "=" * 80)
    print("CHUNKING VALIDATION")
    print("=" * 80)

    print(f"\nSource documents       : {len(source_df):,}")
    print(f"Documents with chunks  : {chunks_df['document_id'].nunique():,}")
    print(f"Total chunks           : {len(chunks_df):,}")

    missing_documents = set(
        source_df["document_id"]
    ) - set(
        chunks_df["document_id"]
    )

    print(
        f"Documents without chunks: "
        f"{len(missing_documents):,}"
    )

    if missing_documents:
        print(
            f"Missing document IDs: "
            f"{sorted(missing_documents)}"
        )

    # Empty chunks
    empty_chunks = (
        chunks_df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nEmpty chunks: {empty_chunks}")

    # Length
    lengths = chunks_df["text"].str.len()

    print("\nChunk length statistics:")
    print(f"  Min    : {lengths.min()}")
    print(f"  Max    : {lengths.max()}")
    print(f"  Mean   : {lengths.mean():.1f}")
    print(f"  Median : {lengths.median():.1f}")

    too_long = (lengths > MAX_CHUNK_LENGTH).sum()
    too_short = (lengths < MIN_CHUNK_LENGTH).sum()

    print(f"\nChunks > {MAX_CHUNK_LENGTH}: {too_long}")
    print(f"Chunks < {MIN_CHUNK_LENGTH}: {too_short}")

    # Duplicate IDs
    duplicate_ids = chunks_df["chunk_id"].duplicated().sum()

    print(f"\nDuplicated chunk IDs: {duplicate_ids}")

    # Duplicate text
    duplicate_texts = chunks_df["text"].duplicated().sum()

    print(f"Duplicated texts: {duplicate_texts}")

    # Document distribution
    print("\nChunks per document:")
    print(
        chunks_df
        .groupby("document_id")
        .size()
        .describe()
    )

    # Chunk type
    print("\nChunk type distribution:")
    print(
        chunks_df["chunk_type"]
        .value_counts()
    )

    # Quality checks
    print("\n" + "-" * 80)
    print("QUALITY CHECK")
    print("-" * 80)

    if empty_chunks == 0:
        print("[OK] No empty chunks")
    else:
        print("[WARNING] Empty chunks detected")

    if too_long == 0:
        print(
            f"[OK] All chunks <= {MAX_CHUNK_LENGTH} chars"
        )
    else:
        print(
            f"[WARNING] {too_long} chunks exceed "
            f"{MAX_CHUNK_LENGTH} chars"
        )

    if too_short == 0:
        print(
            f"[OK] No chunks < {MIN_CHUNK_LENGTH} chars"
        )
    else:
        print(
            f"[WARNING] {too_short} chunks < "
            f"{MIN_CHUNK_LENGTH} chars"
        )

    if duplicate_ids == 0:
        print("[OK] Chunk IDs are unique")
    else:
        print("[WARNING] Duplicate chunk IDs")

    if len(missing_documents) == 0:
        print("[OK] All documents produced chunks")
    else:
        print(
            f"[WARNING] {len(missing_documents)} "
            f"documents produced no chunks"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("RAG CHUNKING")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading preprocessed dataset...")
    print(f"Input file: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Documents loaded: {len(df):,}")

    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING CHUNKS")
    print("=" * 80)

    all_chunks = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Chunking"
    ):

        try:

            document_chunks = create_chunks(row)

            all_chunks.extend(document_chunks)

        except Exception as e:

            print(
                f"\n[ERROR] Document "
                f"{row['document_id']}: {e}"
            )

    chunks_df = pd.DataFrame(all_chunks)

    # --------------------------------------------------------
    # CREATE CHUNK ID
    # --------------------------------------------------------

    chunks_df.insert(
        0,
        "chunk_id",
        [
            f"doc_{doc_id}_chunk_{i}"
            for doc_id, i in zip(
                chunks_df["document_id"],
                chunks_df["chunk_index"]
            )
        ]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_chunks(
        chunks_df,
        df
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)

    for _, row in chunks_df.head(10).iterrows():

        print("\n" + "-" * 80)

        print(f"Chunk ID      : {row['chunk_id']}")
        print(f"Document ID   : {row['document_id']}")
        print(f"Procedure     : {row['procedure_name']}")
        print(f"Chunk type    : {row['chunk_type']}")
        print(f"Length        : {len(row['text'])}")

        print("\nTEXT:")
        print(row["text"])

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(f"\nOutput file:")
    print(OUTPUT_FILE)

    print(f"\nRows: {len(chunks_df):,}")

    print("\n" + "=" * 80)
    print("CHUNKING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()