# ============================================================
# VIETNAMESE LEGAL RAG
# LEGAL STRUCTURE-AWARE CHUNKING
# ============================================================

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\legal-rag")

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "vietnamese-legal-documents"
    / "data"
)

# Your actual dataset currently contains content.parquet.
# The automatic detection below will also find it.
INPUT_FILE = DATA_DIR / "content.parquet"

OUTPUT_FILE = DATA_DIR / "chunks.parquet"

# ------------------------------------------------------------
# TEST MODE
# ------------------------------------------------------------
# True  -> process only TEST_DOCUMENTS
# False -> process the entire dataset
TEST_MODE = True
TEST_DOCUMENTS = 20

# ------------------------------------------------------------
# CHUNK SIZE
# ------------------------------------------------------------

MAX_CHUNK_SIZE = 2500

# Chunks shorter than this are considered small.
MIN_CHUNK_SIZE = 30

# Try to merge small neighboring chunks.
MERGE_SMALL_CHUNKS = True

# Only merge when resulting chunk is below this size.
MERGE_TARGET_SIZE = 1800

# ------------------------------------------------------------
# OUTPUT / DEBUG
# ------------------------------------------------------------

SHOW_SAMPLES = 10

# Save a CSV containing documents that failed to produce chunks.
SAVE_FAILED_DOCUMENTS = True

FAILED_DOCUMENTS_FILE = DATA_DIR / "failed_documents.csv"


# ============================================================
# REGEX PATTERNS
# ============================================================

# ------------------------------------------------------------
# ARTICLE
#
# Examples:
#   Điều 1.
#   Điều 12.
#   Điều 12:
#   Điều 12a.
#   ĐIỀU 5
#
# We intentionally require "Điều" + number.
# ------------------------------------------------------------

ARTICLE_RE = re.compile(
    r"(?im)^\s*Điều\s+(\d{1,3}[A-Za-z]?)\s*[\.:]?\s*"
)


# ------------------------------------------------------------
# CLAUSE
#
# Examples:
#   1.
#   2.
#   10.
#
# Clause is only recognized at line start.
# ------------------------------------------------------------

CLAUSE_RE = re.compile(
    r"(?m)^\s*(\d{1,3})\.\s+"
)


# ------------------------------------------------------------
# POINT
#
# Examples:
#   a)
#   b)
#   c)
#   đ)
#   a.
#
# Vietnamese legal documents may use "đ)".
# ------------------------------------------------------------

POINT_RE = re.compile(
    r"(?im)^\s*([a-zđ])[\)\.]\s+"
)


# ------------------------------------------------------------
# INLINE POINT
#
# Examples:
#   ... gồm:
#   a) ...
#   b) ...
#
# Only split after ":" or ";".
# ------------------------------------------------------------

INLINE_POINT_RE = re.compile(
    r"(?i)(?<=[\:;])\s+([a-zđ])[\)\.]\s+"
)


# ------------------------------------------------------------
# SENTENCE
#
# Conservative sentence boundary.
# ------------------------------------------------------------

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÀ-ỸĐ])"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode using NFKC.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    return unicodedata.normalize("NFKC", text)


def normalize_spaces(text: str) -> str:
    """
    Normalize whitespace while preserving line breaks.
    """

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove whitespace around newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fix_split_legal_markers(text: str) -> str:
    """
    Repair OCR / HTML extraction cases such as:

        Điều
        12.

    ->

        Điều 12.

    Also:

        Khoản
        1.

    ->

        Khoản 1.

    And:

        Điểm
        a)

    ->

        Điểm a)
    """

    if not text:
        return ""

    # Điều\n12
    text = re.sub(
        r"\b(Điều)\s*\n+\s*(\d{1,3}[A-Za-z]?)",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    # Khoản\n1
    text = re.sub(
        r"\b(Khoản)\s*\n+\s*(\d{1,3})",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    # Điểm\na
    text = re.sub(
        r"\b(Điểm)\s*\n+\s*([a-zđ])\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    return text


# ============================================================
# HTML CLEANING
# ============================================================

def clean_html(html: str) -> str:
    """
    Convert HTML into clean plain text.

    Important:
    We preserve block-level boundaries because legal structure
    depends heavily on line boundaries.
    """

    if html is None:
        return ""

    if not isinstance(html, str):
        html = str(html)

    if not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove irrelevant HTML
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Block-level elements
    block_tags = [
        "p",
        "div",
        "section",
        "article",
        "li",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
    ]

    for tag in soup.find_all(block_tags):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # Explicit line breaks
    for tag in soup.find_all("br"):
        tag.replace_with("\n")

    text = soup.get_text(" ")

    text = normalize_unicode(text)

    text = fix_split_legal_markers(text)

    text = normalize_spaces(text)

    return text.strip()


# ============================================================
# STRUCTURAL LINE PREPARATION
# ============================================================

def prepare_structure_lines(text: str) -> str:
    """
    Re-introduce line breaks before strong legal markers.

    We do NOT split every "1." globally because dates and
    ordinary numbering can contain numbers followed by dots.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # Article
    # --------------------------------------------------------

    text = re.sub(
        r"\s+(?=Điều\s+\d{1,3}[A-Za-z]?\s*[\.:]?\s*)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Article at beginning
    text = re.sub(
        r"(?i)(?<!\n)(Điều\s+\d{1,3}[A-Za-z]?\s*[\.:]?)",
        r"\n\1",
        text,
    )

    # --------------------------------------------------------
    # Clause
    #
    # Only split when a number is followed by "." and then
    # an uppercase Vietnamese character.
    #
    # This avoids splitting:
    #   ngày 07.09.2016
    #   1.5 tỷ
    # --------------------------------------------------------

    text = re.sub(
        r"\s+(?=(\d{1,3})\.\s+(?=[A-ZÀ-ỸĐ]))",
        "\n",
        text,
    )

    # --------------------------------------------------------
    # Inline points
    # --------------------------------------------------------

    text = INLINE_POINT_RE.sub(
        lambda m: "\n" + m.group(1).lower() + ") ",
        text,
    )

    # --------------------------------------------------------
    # Normalize lines
    # --------------------------------------------------------

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return text.strip()


def remove_noise_lines(text: str) -> str:
    """
    Remove obvious extraction artifacts.
    """

    if not text:
        return ""

    lines = text.splitlines()

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Repeated separators
        if re.fullmatch(r"[_=\-]{5,}", line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def normalize_document(text: str) -> str:
    """
    Complete document normalization pipeline.
    """

    text = clean_html(text)

    if not text:
        return ""

    text = remove_noise_lines(text)

    text = fix_split_legal_markers(text)

    # Normalize spaces inside lines only
    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Prepare strong structural boundaries
    text = prepare_structure_lines(text)

    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# ARTICLE SPLITTING
# ============================================================

def split_articles(
    text: str,
) -> List[Tuple[Optional[str], str]]:
    """
    Split document into articles.

    Returns:
        [
            (article_number, article_text),
            ...
        ]

    If no article is detected, the entire document is
    returned as one item.
    """

    if not text:
        return []

    matches = list(ARTICLE_RE.finditer(text))

    if not matches:
        return [(None, text.strip())]

    articles = []

    # --------------------------------------------------------
    # Prefix before first Article
    # --------------------------------------------------------

    prefix = text[: matches[0].start()].strip()

    if prefix:
        articles.append((None, prefix))

    # --------------------------------------------------------
    # Articles
    # --------------------------------------------------------

    for i, match in enumerate(matches):

        article_number = match.group(1)

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        article_text = text[start:end].strip()

        if article_text:
            articles.append(
                (
                    article_number,
                    article_text,
                )
            )

    return articles


# ============================================================
# CLAUSE SPLITTING
# ============================================================

def split_clauses(
    article_text: str,
) -> List[Tuple[Optional[str], str]]:
    """
    Split an article into clauses.

    Clause markers are recognized conservatively.
    """

    if not article_text:
        return []

    text = article_text.strip()

    # Ensure likely clause markers start a line.
    text = re.sub(
        r"\s+(?=(\d{1,3})\.\s+(?=[A-ZÀ-ỸĐ]))",
        "\n",
        text,
    )

    matches = list(CLAUSE_RE.finditer(text))

    if not matches:
        return [(None, text)]

    clauses = []

    # --------------------------------------------------------
    # Prefix before first clause
    # --------------------------------------------------------

    prefix = text[: matches[0].start()].strip()

    if prefix:
        clauses.append(
            (
                None,
                prefix,
            )
        )

    # --------------------------------------------------------
    # Clauses
    # --------------------------------------------------------

    for i, match in enumerate(matches):

        clause_number = match.group(1)

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        clause_text = text[start:end].strip()

        if clause_text:
            clauses.append(
                (
                    clause_number,
                    clause_text,
                )
            )

    return clauses


# ============================================================
# POINT SPLITTING
# ============================================================

def split_points(
    clause_text: str,
) -> List[Tuple[Optional[str], str]]:
    """
    Split clause into legal points.

    Supports:

        a)
        b)
        c)
        đ)

    and inline points after ":" / ";".
    """

    if not clause_text:
        return []

    text = clause_text.strip()

    # --------------------------------------------------------
    # Convert inline points into line-start points.
    #
    # Example:
    #
    # Điều kiện gồm: a) ... b) ... c) ...
    #
    # ->
    #
    # Điều kiện gồm:
    # a) ...
    # b) ...
    # c) ...
    # --------------------------------------------------------

    text = INLINE_POINT_RE.sub(
        lambda m: "\n" + m.group(1).lower() + ") ",
        text,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    text = re.sub(r"\n{3,}", "\n\n", text)

    matches = list(POINT_RE.finditer(text))

    if not matches:
        return [(None, text)]

    points = []

    # --------------------------------------------------------
    # Prefix before first point
    # --------------------------------------------------------

    prefix = text[: matches[0].start()].strip()

    if prefix:
        points.append(
            (
                None,
                prefix,
            )
        )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    for i, match in enumerate(matches):

        point_number = match.group(1).lower()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        point_text = text[start:end].strip()

        if point_text:
            points.append(
                (
                    point_number,
                    point_text,
                )
            )

    return points


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitting.

    Existing line boundaries are always respected first.
    """

    if not text:
        return []

    text = text.strip()

    paragraphs = [
        p.strip()
        for p in text.splitlines()
        if p.strip()
    ]

    sentences = []

    for paragraph in paragraphs:

        if len(paragraph) <= MAX_CHUNK_SIZE:
            sentences.append(paragraph)
            continue

        # Split long paragraph at sentence boundaries.
        parts = SENTENCE_RE.split(paragraph)

        for part in parts:

            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


# ============================================================
# HARD SPLIT
# ============================================================

def hard_split(
    text: str,
    max_length: int,
) -> List[str]:
    """
    Final fallback for extremely long text.

    Never cuts a word unless no reasonable space exists.
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= max_length:
        return [text]

    chunks = []

    remaining = text

    while len(remaining) > max_length:

        cut = remaining.rfind(
            " ",
            0,
            max_length + 1,
        )

        # If no suitable space is found,
        # use hard character boundary.
        if cut < int(max_length * 0.5):
            cut = max_length

        piece = remaining[:cut].strip()

        if piece:
            chunks.append(piece)

        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


# ============================================================
# ENFORCE MAX LENGTH
# ============================================================

def enforce_max_length(
    text: str,
) -> List[str]:
    """
    Ensure all chunks are <= MAX_CHUNK_SIZE.

    Priority:

        1. Existing legal structure
        2. Sentence boundaries
        3. Hard split
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    sentence_parts = split_sentences(text)

    if not sentence_parts:
        return hard_split(
            text,
            MAX_CHUNK_SIZE,
        )

    final_chunks = []

    current = ""

    for sentence in sentence_parts:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Single sentence exceeds max.
        if len(sentence) > MAX_CHUNK_SIZE:

            if current:
                final_chunks.append(
                    current.strip()
                )
                current = ""

            final_chunks.extend(
                hard_split(
                    sentence,
                    MAX_CHUNK_SIZE,
                )
            )

            continue

        # Start first chunk.
        if not current:
            current = sentence
            continue

        candidate = (
            current
            + " "
            + sentence
        )

        if len(candidate) <= MAX_CHUNK_SIZE:
            current = candidate

        else:
            final_chunks.append(
                current.strip()
            )

            current = sentence

    if current:
        final_chunks.append(
            current.strip()
        )

    # Final safety
    safe_chunks = []

    for chunk in final_chunks:

        if len(chunk) <= MAX_CHUNK_SIZE:
            safe_chunks.append(chunk)

        else:
            safe_chunks.extend(
                hard_split(
                    chunk,
                    MAX_CHUNK_SIZE,
                )
            )

    return safe_chunks


# ============================================================
# SMALL CHUNK MERGING
# ============================================================

def merge_small_chunks(
    chunks: List[Dict],
) -> List[Dict]:
    """
    Merge small neighboring chunks.

    IMPORTANT:
    We do NOT require "part" to be equal because "part"
    is the chunk sequence number and therefore changes
    for every chunk.
    """

    if not chunks:
        return chunks

    result = []

    for chunk in chunks:

        if not result:
            result.append(chunk)
            continue

        previous = result[-1]

        # ----------------------------------------------------
        # Same legal structure
        # ----------------------------------------------------

        same_structure = (
            previous["document_id"]
            == chunk["document_id"]
            and previous["article"]
            == chunk["article"]
            and previous["clause"]
            == chunk["clause"]
            and previous["point"]
            == chunk["point"]
        )

        combined_length = (
            len(previous["text"])
            + 1
            + len(chunk["text"])
        )

        # ----------------------------------------------------
        # Merge if previous chunk is small
        # ----------------------------------------------------

        if (
            MERGE_SMALL_CHUNKS
            and same_structure
            and len(previous["text"]) < MIN_CHUNK_SIZE
            and combined_length <= MERGE_TARGET_SIZE
        ):
            previous["text"] = (
                previous["text"].rstrip()
                + " "
                + chunk["text"].lstrip()
            )

            previous["length"] = len(
                previous["text"]
            )

            continue

        # ----------------------------------------------------
        # Merge if current chunk is small
        # ----------------------------------------------------

        if (
            MERGE_SMALL_CHUNKS
            and same_structure
            and len(chunk["text"]) < MIN_CHUNK_SIZE
            and combined_length <= MERGE_TARGET_SIZE
        ):
            previous["text"] = (
                previous["text"].rstrip()
                + " "
                + chunk["text"].lstrip()
            )

            previous["length"] = len(
                previous["text"]
            )

            continue

        result.append(chunk)

    return result


# ============================================================
# CHUNK CREATION
# ============================================================

def create_base_chunk(
    document_id,
    text: str,
    article: Optional[str],
    clause: Optional[str],
    point: Optional[str],
    part: int,
) -> Dict:

    text = text.strip()

    return {
        "id": None,

        "document_id": document_id,

        "chapter": "",

        "section": "",

        "article": (
            str(article)
            if article is not None
            else ""
        ),

        "clause": (
            str(clause)
            if clause is not None
            else ""
        ),

        "point": (
            str(point)
            if point is not None
            else ""
        ),

        "part": part,

        "text": text,

        "length": len(text),
    }


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_document(
    document_id,
    html_content: str,
) -> List[Dict]:
    """
    Main structure-aware chunking function.

    Hierarchy:

        Document
            |
            +-- Article
                    |
                    +-- Clause
                            |
                            +-- Point
                                    |
                                    +-- Sentence chunks

    If the document has no legal structure,
    the full document is still preserved.
    """

    normalized = normalize_document(
        html_content
    )

    # --------------------------------------------------------
    # Empty document
    # --------------------------------------------------------

    if not normalized:
        return []

    # --------------------------------------------------------
    # Articles
    # --------------------------------------------------------

    articles = split_articles(
        normalized
    )

    if not articles:
        articles = [
            (
                None,
                normalized,
            )
        ]

    document_chunks = []

    part_counter = 1

    # ========================================================
    # ARTICLE
    # ========================================================

    for article_number, article_text in articles:

        if not article_text:
            continue

        # ====================================================
        # CLAUSE
        # ====================================================

        clauses = split_clauses(
            article_text
        )

        if not clauses:
            clauses = [
                (
                    None,
                    article_text,
                )
            ]

        for clause_number, clause_text in clauses:

            if not clause_text:
                continue

            # =================================================
            # POINT
            # =================================================

            points = split_points(
                clause_text
            )

            if not points:
                points = [
                    (
                        None,
                        clause_text,
                    )
                ]

            for point_number, point_text in points:

                if not point_text:
                    continue

                # =============================================
                # MAX LENGTH
                # =============================================

                pieces = enforce_max_length(
                    point_text
                )

                for piece in pieces:

                    piece = piece.strip()

                    if not piece:
                        continue

                    chunk = create_base_chunk(
                        document_id=document_id,
                        text=piece,
                        article=article_number,
                        clause=clause_number,
                        point=point_number,
                        part=part_counter,
                    )

                    document_chunks.append(
                        chunk
                    )

                    part_counter += 1

    # ========================================================
    # FALLBACK
    #
    # Very important:
    # Never lose a non-empty document because the structural
    # parser failed.
    # ========================================================

    if not document_chunks:

        fallback_chunks = enforce_max_length(
            normalized
        )

        for piece in fallback_chunks:

            document_chunks.append(
                create_base_chunk(
                    document_id=document_id,
                    text=piece,
                    article=None,
                    clause=None,
                    point=None,
                    part=part_counter,
                )
            )

            part_counter += 1

    # ========================================================
    # MERGE SMALL CHUNKS
    # ========================================================

    document_chunks = merge_small_chunks(
        document_chunks
    )

    return document_chunks


# ============================================================
# DATASET LOADING
# ============================================================

def find_input_file() -> Path:
    """
    Locate the input dataset.

    Priority:
        1. Explicit INPUT_FILE
        2. Common filenames
        3. Any parquet containing id + content_html
    """

    # --------------------------------------------------------
    # Explicit input
    # --------------------------------------------------------

    if INPUT_FILE.exists():
        return INPUT_FILE

    # --------------------------------------------------------
    # Common filenames
    # --------------------------------------------------------

    candidates = [
        DATA_DIR / "content.parquet",
        DATA_DIR / "documents.parquet",
        DATA_DIR / "legal_documents.parquet",
        DATA_DIR / "vietnamese-legal-documents.parquet",
        DATA_DIR / "data.parquet",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # --------------------------------------------------------
    # Automatic search
    # --------------------------------------------------------

    parquet_files = list(
        DATA_DIR.glob("*.parquet")
    )

    parquet_files = [
        p
        for p in parquet_files
        if p.name != OUTPUT_FILE.name
    ]

    for path in parquet_files:

        try:

            df = pd.read_parquet(
                path,
                columns=[
                    "id",
                    "content_html",
                ],
            )

            if (
                "id" in df.columns
                and "content_html" in df.columns
            ):
                return path

        except Exception:
            continue

    raise FileNotFoundError(
        "\nCould not find input dataset.\n"
        f"Expected directory:\n{DATA_DIR}\n\n"
        "Please set INPUT_FILE manually.\n"
    )


def load_dataset() -> pd.DataFrame:
    """
    Load dataset and validate required columns.
    """

    input_file = find_input_file()

    print(
        f"Input file: {input_file}"
    )

    df = pd.read_parquet(
        input_file
    )

    required_columns = {
        "id",
        "content_html",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Documents loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List]:
    """
    Process dataset.

    Returns:
        chunks_df
        failed_documents
    """

    source_df = df.copy()

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:

        print()

        print(
            f"TEST MODE enabled: "
            f"{TEST_DOCUMENTS} documents"
        )

        df = df.head(
            TEST_DOCUMENTS
        ).copy()

    print()

    if TEST_MODE:
        print(
            f"TEST MODE: {len(df)} documents"
        )
    else:
        print(
            f"FULL MODE: {len(df):,} documents"
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    all_chunks: List[Dict] = []

    failed_documents = []

    empty_documents = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Chunking",
    ):

        document_id = row["id"]

        content_html = row[
            "content_html"
        ]

        try:

            # -----------------------------------------------
            # Detect empty source
            # -----------------------------------------------

            if (
                content_html is None
                or pd.isna(content_html)
                or not str(content_html).strip()
            ):

                failed_documents.append(
                    document_id
                )

                empty_documents.append(
                    document_id
                )

                continue

            # -----------------------------------------------
            # Chunk
            # -----------------------------------------------

            chunks = chunk_document(
                document_id=document_id,
                html_content=str(
                    content_html
                ),
            )

            # -----------------------------------------------
            # Safety fallback
            # -----------------------------------------------

            if not chunks:

                failed_documents.append(
                    document_id
                )

                continue

            all_chunks.extend(
                chunks
            )

        except Exception as exc:

            failed_documents.append(
                document_id
            )

            print(
                f"\nWarning: failed document "
                f"{document_id}: {exc}"
            )

    # --------------------------------------------------------
    # Assign global IDs
    # --------------------------------------------------------

    for idx, chunk in enumerate(
        all_chunks
    ):

        chunk["id"] = idx

    chunks_df = pd.DataFrame(
        all_chunks
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()

    print(
        f"Total chunks created: "
        f"{len(chunks_df):,}"
    )

    print(
        f"Documents without chunks: "
        f"{len(failed_documents)}"
    )

    if failed_documents:

        print(
            "Failed document IDs:",
            failed_documents[:20],
        )

    # --------------------------------------------------------
    # Save failed document information
    # --------------------------------------------------------

    if (
        SAVE_FAILED_DOCUMENTS
        and failed_documents
    ):

        failed_rows = source_df[
            source_df["id"].isin(
                failed_documents
            )
        ].copy()

        failed_rows[
            "failure_reason"
        ] = "No chunks generated"

        failed_rows.to_csv(
            FAILED_DOCUMENTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Failed documents saved to:\n"
            f"{FAILED_DOCUMENTS_FILE}"
        )

    return (
        chunks_df,
        failed_documents,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(
    chunks_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:

    print()

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if chunks_df.empty:

        print(
            "ERROR: No chunks created."
        )

        return

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    texts = (
        chunks_df["text"]
        .fillna("")
        .astype(str)
    )

    lengths = texts.str.len()

    empty_chunks = (
        texts.str.strip()
        .eq("")
        .sum()
    )

    average_length = lengths.mean()

    median_length = lengths.median()

    min_length = lengths.min()

    max_length = lengths.max()

    over_max = (
        lengths
        > MAX_CHUNK_SIZE
    ).sum()

    under_min = (
        lengths
        < MIN_CHUNK_SIZE
    ).sum()

    # ========================================================
    # STRUCTURE
    # ========================================================

    article_count = (
        chunks_df["article"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    clause_count = (
        chunks_df["clause"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    point_count = (
        chunks_df["point"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    unique_documents = (
        chunks_df[
            "document_id"
        ].nunique()
    )

    average_chunks_per_document = (
        len(chunks_df)
        / unique_documents
        if unique_documents
        else 0
    )

    # ========================================================
    # DUPLICATE IDS
    # ========================================================

    duplicate_ids = (
        chunks_df["id"]
        .duplicated()
        .sum()
    )

    # ========================================================
    # DUPLICATE TEXTS
    # ========================================================

    duplicate_texts = (
        texts
        .duplicated()
        .sum()
    )

    # Duplicate text WITHIN SAME DOCUMENT
    duplicate_same_document = (
        chunks_df[
            ["document_id", "text"]
        ]
        .duplicated()
        .sum()
    )

    # ========================================================
    # SOURCE COVERAGE
    # ========================================================

    source_documents = set(
        source_df["id"].tolist()
    )

    chunk_documents = set(
        chunks_df[
            "document_id"
        ].tolist()
    )

    missing_documents = (
        source_documents
        - chunk_documents
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Empty chunks: "
        f"{empty_chunks}"
    )

    print(
        f"Average length: "
        f"{average_length:.1f} chars"
    )

    print(
        f"Median length: "
        f"{median_length:.1f} chars"
    )

    print(
        f"Min length: "
        f"{min_length} chars"
    )

    print(
        f"Max length: "
        f"{max_length:,} chars"
    )

    print(
        f"Chunks > {MAX_CHUNK_SIZE}: "
        f"{over_max}"
    )

    print(
        f"Chunks < {MIN_CHUNK_SIZE}: "
        f"{under_min}"
    )

    print(
        f"Chunks with Article: "
        f"{article_count:,} "
        f"({article_count / len(chunks_df) * 100:.2f}%)"
    )

    print(
        f"Chunks with Clause: "
        f"{clause_count:,} "
        f"({clause_count / len(chunks_df) * 100:.2f}%)"
    )

    print(
        f"Chunks with Point: "
        f"{point_count:,} "
        f"({point_count / len(chunks_df) * 100:.2f}%)"
    )

    print(
        f"Unique documents: "
        f"{unique_documents:,}"
    )

    print(
        f"Average chunks/document: "
        f"{average_chunks_per_document:.2f}"
    )

    print(
        f"Duplicated chunk IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Duplicated texts: "
        f"{duplicate_texts}"
    )

    print(
        f"Duplicated text within same document: "
        f"{duplicate_same_document}"
    )

    # ========================================================
    # COVERAGE
    # ========================================================

    print()

    print(
        f"Source documents: "
        f"{len(source_documents):,}"
    )

    print(
        f"Documents producing chunks: "
        f"{len(chunk_documents):,}"
    )

    print(
        f"Documents without chunks: "
        f"{len(missing_documents):,}"
    )

    if missing_documents:

        print(
            "Missing document IDs:",
            list(missing_documents)[:20],
        )

    # ========================================================
    # QUALITY CHECK
    # ========================================================

    print()

    print("QUALITY CHECK")

    if empty_chunks == 0:
        print(
            "[OK] No empty chunks"
        )
    else:
        print(
            f"[WARNING] "
            f"{empty_chunks} empty chunks"
        )

    if over_max == 0:
        print(
            f"[OK] All chunks <= "
            f"{MAX_CHUNK_SIZE} chars"
        )
    else:
        print(
            f"[ERROR] {over_max} chunks "
            f"exceed {MAX_CHUNK_SIZE} chars"
        )

    if duplicate_ids == 0:
        print(
            "[OK] Chunk IDs are unique"
        )
    else:
        print(
            f"[ERROR] {duplicate_ids} "
            f"duplicate IDs"
        )

    if under_min == 0:
        print(
            f"[OK] No chunks < "
            f"{MIN_CHUNK_SIZE} chars"
        )
    else:
        print(
            f"[WARNING] {under_min} chunks "
            f"< {MIN_CHUNK_SIZE} chars"
        )

    if missing_documents == set():
        print(
            "[OK] All test documents "
            "produced chunks"
        )
    else:
        print(
            f"[WARNING] "
            f"{len(missing_documents)} "
            f"documents produced no chunks"
        )

    if point_count > 0:
        print(
            "[OK] Point-level metadata detected"
        )
    else:
        print(
            "[INFO] No point metadata detected"
        )


# ============================================================
# STRUCTURE STATISTICS
# ============================================================

def print_structure_statistics(
    chunks_df: pd.DataFrame,
) -> None:

    print()

    print("=" * 70)
    print("STRUCTURE STATISTICS")
    print("=" * 70)

    if chunks_df.empty:
        return

    # ========================================================
    # ARTICLES
    # ========================================================

    print()

    print("Top Article numbers:")

    article_series = (
        chunks_df["article"]
        .replace("", pd.NA)
        .dropna()
    )

    if not article_series.empty:

        print(
            article_series
            .value_counts()
            .head(20)
            .to_string()
        )

    else:

        print(
            "No article metadata detected."
        )

    # ========================================================
    # CLAUSES
    # ========================================================

    print()

    print("Top Clause numbers:")

    clause_series = (
        chunks_df["clause"]
        .replace("", pd.NA)
        .dropna()
    )

    if not clause_series.empty:

        print(
            clause_series
            .value_counts()
            .head(20)
            .to_string()
        )

    else:

        print(
            "No clause metadata detected."
        )

    # ========================================================
    # POINTS
    # ========================================================

    print()

    print("Point distribution:")

    point_series = (
        chunks_df["point"]
        .replace("", pd.NA)
        .dropna()
    )

    if not point_series.empty:

        print(
            point_series
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "No point metadata detected."
        )


# ============================================================
# SAMPLE OUTPUT
# ============================================================

def print_sample_chunks(
    chunks_df: pd.DataFrame,
    n: int = 10,
) -> None:

    print()

    print("=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    if chunks_df.empty:
        return

    samples = chunks_df.head(n)

    for _, row in samples.iterrows():

        print()

        print("-" * 70)

        print(
            f"Chunk ID   : {row['id']}"
        )

        print(
            f"Document ID: {row['document_id']}"
        )

        print(
            f"Chapter    : {row['chapter']}"
        )

        print(
            f"Section    : {row['section']}"
        )

        print(
            f"Article    : {row['article']}"
        )

        print(
            f"Clause     : {row['clause']}"
        )

        print(
            f"Point      : {row['point']}"
        )

        print(
            f"Part       : {row['part']}"
        )

        print(
            f"Length     : {row['length']}"
        )

        print()

        print("TEXT:")

        print(
            row["text"]
        )


# ============================================================
# SAVE
# ============================================================

def save_chunks(
    chunks_df: pd.DataFrame,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(chunks_df):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print("VIETNAMESE LEGAL RAG")
    print("LEGAL STRUCTURE-AWARE CHUNKING")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    print()

    print("Loading dataset...")

    df = load_dataset()

    # ========================================================
    # 2. SELECT TEST DATA
    # ========================================================

    if TEST_MODE:

        source_df = df.head(
            TEST_DOCUMENTS
        ).copy()

    else:

        source_df = df.copy()

    # ========================================================
    # 3. PROCESS
    # ========================================================

    chunks_df, failed_documents = (
        process_dataset(
            df
        )
    )

    # ========================================================
    # 4. VALIDATION
    # ========================================================

    validate_chunks(
        chunks_df=chunks_df,
        source_df=source_df,
    )

    # ========================================================
    # 5. STRUCTURE STATISTICS
    # ========================================================

    print_structure_statistics(
        chunks_df
    )

    # ========================================================
    # 6. SAMPLE
    # ========================================================

    print_sample_chunks(
        chunks_df,
        n=SHOW_SAMPLES,
    )

    # ========================================================
    # 7. SAVE
    # ========================================================

    save_chunks(
        chunks_df
    )

    # ========================================================
    # 8. FINAL
    # ========================================================

    print()

    print("=" * 70)
    print("CHUNKING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    if failed_documents:

        print()

        print(
            f"WARNING: "
            f"{len(failed_documents)} "
            f"documents did not produce chunks."
        )

    else:

        print()

        print(
            "All processed documents "
            "produced chunks."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()