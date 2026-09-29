from pathlib import Path
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# preprocessing.py ghi procedures_clean.csv vào <thư mục chứa script>/processed
# (vd D:\legal-rag\src\processed). Thử lần lượt các vị trí, lấy file mới nhất.
INPUT_CANDIDATES = [
    Path(__file__).resolve().parent / "processed" / "procedures_clean.csv",
    PROJECT_ROOT / "data" / "processed" / "procedures_clean.csv",
]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_chunks.parquet"
)

# Maximum characters for one chunk (đã tính cả phần header tên + số hiệu)
MAX_CHUNK_LENGTH = 1200

# Minimum useful length for a chunk
MIN_CHUNK_LENGTH = 20

# Không cho phần thân nhỏ hơn mức này dù header dài
MIN_BODY_BUDGET = 400

# Độ dài tối đa của tên thủ tục trong header mỗi chunk
MAX_NAME_IN_HEADER = 250

# Độ dài tối đa của dòng tiêu đề "* trường hợp" khi nhắc lại ở chunk tiếp theo
MAX_HEADING_REPEAT = 150

# Giá trị của cột receiving_address coi như "không có địa điểm nộp"
NO_ADDRESS_RE = re.compile(
    r"^(không có thông tin|không có|không|n/?a|[-–—.]+)\.?$", re.IGNORECASE
)


def find_input_file() -> Path:
    existing = [p for p in INPUT_CANDIDATES if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "\nKhông tìm thấy procedures_clean.csv ở:\n"
            + "\n".join(f"  {p}" for p in INPUT_CANDIDATES)
            + "\nHãy chạy preprocessing.py trước."
        )
    return max(existing, key=lambda p: p.stat().st_mtime)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text) -> str:
    """
    Normalize text while preserving meaningful line breaks.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# SPLITTING
# ============================================================

def split_sentences(text: str):
    """Fallback: split long paragraph into sentence-like parts."""

    text = clean_text(text)

    if not text:
        return []

    sentences = []

    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        for part in re.split(
            r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỵ0-9Đ])", paragraph
        ):
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def split_long_text(text: str, max_length=MAX_CHUNK_LENGTH):
    """
    Split long text: line -> sentence -> character fallback.

    Checklist có cấu trúc dòng ("* trường hợp" / "1. giấy tờ") nên tách theo
    DÒNG, không cắt giữa một giấy tờ. Khi một chunk tiếp theo bắt đầu giữa
    một "trường hợp", dòng tiêu đề "* trường hợp" được nhắc lại ở đầu chunk
    đó, để các giấy tờ không bị mất ngữ cảnh (vd "xe biển số nước ngoài").
    """

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    has_headings = any(ln.startswith("* ") for ln in lines)

    # Chừa chỗ cho tiêu đề nhắc lại (tối đa MAX_HEADING_REPEAT ký tự)
    unit_limit = max_length - (MAX_HEADING_REPEAT + 20 if has_headings else 0)

    units = []
    for line in lines:
        if len(line) <= unit_limit:
            units.append(line)
        else:
            for sentence in split_sentences(line):
                if len(sentence) <= unit_limit:
                    units.append(sentence)
                else:
                    for start in range(0, len(sentence), unit_limit):
                        piece = sentence[start:start + unit_limit].strip()
                        if piece:
                            units.append(piece)

    chunks = []
    current = ""
    heading = ""

    for unit in units:
        is_heading = unit.startswith("* ")

        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_length:
            current += "\n" + unit
        else:
            chunks.append(current.strip())
            current = ""
            if has_headings and heading and not is_heading:
                h = heading
                if len(h) > MAX_HEADING_REPEAT:
                    h = h[:MAX_HEADING_REPEAT].rstrip() + "…"
                current = f"{h} (tiếp)\n{unit}"
            else:
                current = unit

        if is_heading:
            heading = unit

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# LOCATION HELPERS
# ============================================================

def extract_submission_methods(submission_text: str):
    """
    Lấy các hình thức nộp (Trực tiếp / Bưu chính / Trực tuyến) từ
    submission_text, dạng dòng "- Bưu chính / Trực tiếp: 5 ngày".
    """

    methods = []

    for line in submission_text.split("\n"):
        match = re.match(r"^-\s*(.+?):", line.strip())
        if not match:
            continue
        for part in match.group(1).split("/"):
            part = part.strip()
            if part and part not in methods:
                methods.append(part)

    return methods


def build_location_body(receiving_address, executing_agency, submission_text):
    """
    Nội dung chunk "Nơi nộp hồ sơ".

    Chỉ ~25% thủ tục có địa điểm nộp trong dữ liệu gốc. Khi thiếu, chunk vẫn
    được tạo nhưng NÓI RÕ là nguồn không có địa chỉ, để hệ thống không tự
    suy đoán; chỉ kèm cơ quan thực hiện và hình thức nộp (đều từ nguồn).
    """

    lines = []

    if receiving_address and not NO_ADDRESS_RE.match(receiving_address):
        lines.append(receiving_address)
    else:
        lines.append(
            "Dữ liệu gốc KHÔNG nêu địa điểm/địa chỉ nộp hồ sơ cụ thể cho "
            "thủ tục này (không được tự suy đoán địa chỉ)."
        )

    if executing_agency:
        lines.append(f"Cơ quan thực hiện: {executing_agency}")

    methods = extract_submission_methods(submission_text)
    if methods:
        lines.append(f"Hình thức nộp: {', '.join(methods)}")

    return "\n".join(lines)


# ============================================================
# ADD CHUNK
# ============================================================

def emit(chunks, ctx, chunk_type, title, body):
    """
    Tạo 1..n chunk cho một mục. MỖI chunk luôn bắt đầu bằng header:

        Tên thủ tục hành chính: ...
        Số hiệu văn bản: ...

    và mang cột metadata `legal_codes`, nên mọi kết quả truy xuất đều có
    số hiệu văn bản (kể cả khi 1 mục bị tách thành nhiều chunk).
    """

    body = clean_text(body)

    if not body:
        return

    # Tên thủ tục cực dài (có thủ tục >900 ký tự) sẽ ăn hết ngân sách chunk;
    # chỉ rút gọn trong TEXT header, cột metadata procedure_name vẫn đầy đủ.
    name_in_header = ctx["procedure_name"]
    if len(name_in_header) > MAX_NAME_IN_HEADER:
        name_in_header = name_in_header[:MAX_NAME_IN_HEADER].rstrip() + "…"

    header = (
        f"Tên thủ tục hành chính: {name_in_header}\n"
        f"Số hiệu văn bản: {ctx['legal_codes']}"
    )

    budget = max(MAX_CHUNK_LENGTH - len(header) - 2, MIN_BODY_BUDGET)

    content = f"{title}\n{body}" if title else body

    for piece in split_long_text(content, budget):

        text = f"{header}\n\n{piece}"

        if len(piece) < MIN_CHUNK_LENGTH:
            continue

        chunks.append(
            {
                "document_id": ctx["document_id"],
                "procedure_name": ctx["procedure_name"],
                "executing_agency": ctx["executing_agency"],
                "legal_codes": ctx["legal_codes"],
                "chunk_type": chunk_type,
                "text": text,
            }
        )


# ============================================================
# CREATE CHUNKS FOR ONE DOCUMENT
# ============================================================
#
# Cột đầu vào (procedures_clean.csv, xem preprocessing.py):
#   proc_id, name, executing_agency, requirements, checklist_text,
#   submission_text, fee_text, legal_codes, has_mcq, rag_text
#
# chunk_type (giữ tên cũ để retrieval.py dùng lại được):
#   required_documents  <- checklist_text
#   processing_time     <- submission_text (cách nộp + thời hạn)
#   fee                 <- fee_text
#   location            <- receiving_address (+ cơ quan, hình thức nộp)
#   legal_basis         <- legal_codes (số hiệu văn bản)
#   general_information <- tên, cơ quan thực hiện, điều kiện
# ============================================================

def create_chunks_for_document(row):

    ctx = {
        "document_id": clean_text(row.get("proc_id", "")),
        "procedure_name": clean_text(row.get("name", "")),
        "executing_agency": clean_text(row.get("executing_agency", "")),
        "legal_codes": clean_text(row.get("legal_codes", ""))
        or "Không có thông tin căn cứ pháp lý trong dữ liệu gốc",
    }

    requirements = clean_text(row.get("requirements", ""))
    checklist_text = clean_text(row.get("checklist_text", ""))
    submission_text = clean_text(row.get("submission_text", ""))
    fee_text = clean_text(row.get("fee_text", ""))
    receiving_address = clean_text(row.get("receiving_address", ""))

    chunks = []

    # 1. Thành phần hồ sơ
    emit(chunks, ctx, "required_documents", "Thành phần hồ sơ:", checklist_text)

    # 2. Cách nộp / thời hạn
    emit(
        chunks, ctx, "processing_time",
        "Cách nộp / thời hạn giải quyết:", submission_text,
    )

    # 3. Lệ phí
    emit(chunks, ctx, "fee", "Lệ phí:", fee_text)

    # 3b. Nơi nộp hồ sơ (luôn có chunk; thiếu địa chỉ thì ghi rõ là không có)
    emit(
        chunks, ctx, "location", "Nơi nộp hồ sơ:",
        build_location_body(
            receiving_address, ctx["executing_agency"], submission_text
        ),
    )

    # 4. Căn cứ pháp lý (số hiệu văn bản)
    emit(
        chunks, ctx, "legal_basis",
        "Căn cứ pháp lý (số hiệu văn bản):", ctx["legal_codes"],
    )

    # 5. Thông tin chung
    general = []
    if ctx["executing_agency"]:
        general.append(f"Cơ quan thực hiện: {ctx['executing_agency']}")
    if requirements:
        general.append(f"Điều kiện thực hiện:\n{requirements}")
    if not general:
        general.append("Thủ tục hành chính.")

    emit(chunks, ctx, "general_information", "", "\n\n".join(general))

    return chunks


# ============================================================
# CREATE ALL CHUNKS
# ============================================================

def create_all_chunks(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SEMANTIC CHUNKING")
    print("=" * 70)

    print()
    print(f"Documents to process: {len(df):,}")

    all_chunks = []

    for _, row in df.iterrows():
        all_chunks.extend(create_chunks_for_document(row))

    chunks_df = pd.DataFrame(all_chunks)

    if not chunks_df.empty:
        chunks_df.insert(
            0,
            "chunk_id",
            [
                f"doc_{document_id}_chunk_{index}"
                for document_id, index in zip(
                    chunks_df["document_id"],
                    chunks_df.groupby("document_id").cumcount() + 1,
                )
            ],
        )

    print()
    print(f"Total chunks created: {len(chunks_df):,}")

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

    lengths = chunks_df["text"].str.len()

    print()
    print("Chunk length statistics:")
    print(f"  Min      : {lengths.min():,}")
    print(f"  Max      : {lengths.max():,}")
    print(f"  Mean     : {lengths.mean():.1f}")
    print(f"  Median   : {lengths.median():.1f}")
    print(f"  > {MAX_CHUNK_LENGTH}: {(lengths > MAX_CHUNK_LENGTH).sum()}")

    print()
    print("Chunk type distribution:")
    print(chunks_df["chunk_type"].value_counts().to_string())

    print()
    print("Document statistics:")
    print(f"  Documents : {chunks_df['document_id'].nunique():,}")
    print(f"  Chunks    : {len(chunks_df):,}")

    duplicate_ids = chunks_df["chunk_id"].duplicated().sum()
    empty_chunks = chunks_df["text"].fillna("").str.strip().eq("").sum()
    no_code_col = chunks_df["legal_codes"].fillna("").str.strip().eq("").sum()
    no_code_txt = (~chunks_df["text"].str.contains("Số hiệu văn bản:")).sum()

    print()
    print("QUALITY CHECK")

    print(
        "[OK] Chunk IDs are unique"
        if duplicate_ids == 0
        else f"[ERROR] {duplicate_ids} duplicate chunk IDs"
    )
    print(
        "[OK] No empty chunks"
        if empty_chunks == 0
        else f"[ERROR] {empty_chunks} empty chunks"
    )
    print(
        "[OK] Mọi chunk đều có cột legal_codes"
        if no_code_col == 0
        else f"[ERROR] {no_code_col} chunk thiếu cột legal_codes"
    )
    print(
        "[OK] Mọi chunk đều có dòng 'Số hiệu văn bản:' trong text"
        if no_code_txt == 0
        else f"[ERROR] {no_code_txt} chunk thiếu 'Số hiệu văn bản:' trong text"
    )
    print(
        f"[OK] All chunks <= {MAX_CHUNK_LENGTH} characters"
        if (lengths <= MAX_CHUNK_LENGTH).all()
        else f"[WARNING] Some chunks exceed {MAX_CHUNK_LENGTH} characters"
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

    for chunk_type in chunks_df["chunk_type"].unique():

        row = chunks_df[chunks_df["chunk_type"] == chunk_type].iloc[0]

        print()
        print("-" * 70)
        print(f"Chunk ID   : {row['chunk_id']}")
        print(f"Document ID: {row['document_id']}")
        print(f"Type       : {row['chunk_type']}")
        print(f"Số hiệu VB : {row['legal_codes']}")
        print(f"Length     : {len(row['text'])}")
        print()
        print("TEXT:")
        print(row["text"])


# ============================================================
# SAVE
# ============================================================

def save_chunks(chunks_df: pd.DataFrame):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunks_df.to_parquet(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)
    print()
    print("Output file:")
    print(OUTPUT_FILE)
    print()
    print(f"Chunks: {len(chunks_df):,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("SEMANTIC CHUNKING")
    print("=" * 70)

    input_file = find_input_file()

    print()
    print(f"Input: {input_file}")
    print("Loading processed dataset...")

    # keep_default_na=False: ô rỗng/"Không" giữ nguyên là chuỗi
    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    print(f"Documents loaded: {len(df):,}")

    chunks_df = create_all_chunks(df)

    validate_chunks(chunks_df)
    show_sample_chunks(chunks_df)
    save_chunks(chunks_df)

    print()
    print("=" * 70)
    print("CHUNKING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()