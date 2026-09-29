"""
============================================================
CSDL THỦ TỤC HÀNH CHÍNH (Phase 1 export) - PREPROCESSING
============================================================

Đọc file .xlsx gồm 7 sheet dữ liệu liên kết qua proc_id (Thủ tục, MCQ,
Checklist, Phí, Tệp đính kèm, Căn cứ pháp lý, Cách nộp) + 2 sheet mô tả
(README, Độ phủ) và xuất ra:

  1. procedures_clean.csv  - 1 dòng / 1 thủ tục (proc_id là khoá), có sẵn
     cột rag_text gộp đầy đủ checklist + phí + cách nộp + căn cứ pháp lý,
     sẵn sàng để embedding / nạp vào Qdrant.
  2. mcq_clean.csv         - bảng câu hỏi làm rõ (MCQ), theo proc_id, để
     tích hợp vào retrieval.py ở bước sau (hỏi lại người dùng khi mơ hồ).

QUY TẮC QUAN TRỌNG (rút ra từ sheet README/Độ phủ + kiểm chứng thực tế
trên toàn bộ 1407 thủ tục, không chỉ đọc ghi chú):

- proc_id LUÔN đọc là str. Giá trị gốc dạng "1.000080" là TEXT (số thứ tự
  có số 0 đứng đầu/cuối có ý nghĩa), không phải số thực - pandas mặc định
  sẽ tự suy thành float và cắt mất số 0 cuối nếu không ép kiểu ngay từ đầu.

- TÊN TRÙNG KHÔNG CÓ NGHĨA LÀ THỦ TỤC TRÙNG. Nhiều tỉnh/cơ quan ban hành
  thủ tục cùng tên riêng biệt (proc_id khác nhau). TUYỆT ĐỐI không gộp
  theo tên - chỉ gộp thông tin của CÙNG một proc_id (đã kiểm tra: tên ở
  sheet con và sheet "Thủ tục" khớp 100% theo proc_id, không có drift).

- status_fees == "absent_confirmed" nghĩa là "cổng không công bố mức
  phí", KHÔNG được suy diễn thành "miễn phí". Tương tự cho status_files.
  status_online BỊ TRÙNG HOÀN TOÀN với has_online_submission (CÓ ↔
  present, KHÔNG ↔ absent_confirmed trên toàn bộ 1407 dòng) nên đã bỏ
  khỏi output, chỉ giữ has_online_submission (dễ đọc hơn).

- decision_date là ngày cổng công bố thủ tục, KHÔNG phải ngày văn bản
  luật có hiệu lực. Ngày luật thật nằm trong tên văn bản ở sheet "Căn cứ
  pháp lý" (dạng text tự do, không phải cột ngày riêng). Vẫn giữ lại
  decision_date trong output (đổi tên rõ ràng) vì hữu ích để biết độ mới
  của lần cào dữ liệu, KHÔNG dùng để suy ra "luật có hiệu lực từ ngày".

- Các cột đếm ở sheet "Thủ tục" (n_checklist, n_files, n_legal) thực ra
  KHỚP 100% với số dòng thật trong sheet con tương ứng (đã đối chiếu
  toàn bộ 1407 proc_id). Nhưng n_fees và n_cases thì KHÔNG đáng tin:
    * n_fees lệch ở đúng 484 thủ tục - đó là các thủ tục
      status_fees=absent_confirmed, nơi sheet "Phí" vẫn có 1 dòng
      placeholder ("(cổng KHÔNG công bố phí)") mà cột đếm ở sheet chính
      không tính vào.
    * n_cases lệch ở 1168/1407 thủ tục so với số "trường hợp" thật trong
      Checklist - không có quy luật, không dùng được.
  => Để an toàn và nhất quán, KHÔNG xuất bất kỳ cột đếm gốc nào
  (n_checklist/n_fees/n_files/n_legal/n_cases) ra procedures_clean.csv;
  luôn tự đếm lại bằng cách JOIN trực tiếp vào sheet con sau khi đã loại
  trùng lặp (xem n_*_actual bên dưới).

- LƯU Ý MỚI, quan trọng nhất khi dùng cho RAG: cột "số tiền" trong sheet
  "Phí" = 0.0 KHÔNG có nghĩa là miễn phí. Trong 2305 dòng loại "FEE" có
  số tiền = 0, chỉ 3 dòng thực sự ghi rõ "miễn phí" trong mô tả; phần
  còn lại là do cổng không nhập số cụ thể (mức phí "do HĐND tỉnh quyết
  định", "theo quy định hiện hành", v.v.). Bản gốc của script này render
  thẳng "0 đồng" cho mọi trường hợp này -> SAI, dễ khiến người dùng nghĩ
  thủ tục miễn phí. Đã sửa: chỉ hiển thị số tiền cụ thể khi số tiền > 0;
  khi = 0/rỗng thì giữ nguyên mô tả gốc, KHÔNG tự gắn nhãn "Miễn phí"
  (đã thử dùng từ khoá nhưng phát hiện nhiều câu là "miễn phí có điều
  kiện", gắn nhãn nguyên dòng sẽ lại sai theo đúng kiểu README cảnh báo).

- Đơn vị tiền tệ mặc định là "đồng", nhưng có ít nhất 1 thủ tục (lệ phí
  lãnh sự ở nước ngoài) quy đổi theo USD - đã thêm nhận diện từ khoá
  "đô la/USD" trong mô tả để không gắn nhầm "đồng" vào một khoản phí
  ngoại tệ.

- Các sheet con (Checklist/Phí/Tệp đính kèm/Căn cứ pháp lý/Cách nộp) có
  DÒNG TRÙNG LẶP HOÀN TOÀN thật sự trong dữ liệu gốc (381 dòng ở Phí,
  125 ở Tệp đính kèm, 10 ở Căn cứ pháp lý, 12 ở Cách nộp - đều trùng
  trong CÙNG proc_id, không phải trùng ngẫu nhiên giữa các thủ tục khác
  nhau). Bản gốc không loại trùng -> rag_text bị lặp lại nội dung y hệt
  nhiều lần. Đã thêm bước drop_duplicates() cho tất cả sheet con trước
  khi gộp.

- 2 cột hoàn toàn vô nghĩa (hằng số 100% trên 1407 dòng, không mang
  thông tin): "province" (luôn là "(toàn quốc)") và "status_meta" (luôn
  là "present"). Đã loại khỏi output.

- "department_promulgate" và "issuing_agency" giống hệt nhau 100% trên
  toàn bộ dữ liệu -> chỉ giữ issuing_agency (đi kèm decision_number tạo
  thành cặp thông tin văn bản chuẩn), bỏ department_promulgate.

- Cột "bắt buộc" trong Checklist toàn giá trị "—" (không ai điền) - đã
  không đưa cột này vào bất kỳ đâu trong output, đúng như cảnh báo gốc.

- Tệp đính kèm KHÔNG có URL công khai (phải gọi API nội bộ với file_id
  để tải) - không tạo link tải trực tiếp, chỉ liệt kê tên tệp + hạng
  mục.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATE_RAW_FILES = [
    PROJECT_ROOT / "data" / "raw" / "CSDL_THU_TUC_HANH_CHINH.xlsx",
    Path(__file__).resolve().parent / "CSDL_THU_TUC_HANH_CHINH.xlsx",
]

RAW_FILE = next((p for p in _CANDIDATE_RAW_FILES if p.exists()), _CANDIDATE_RAW_FILES[0])

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

PROCEDURES_OUT = PROCESSED_DIR / "procedures_clean.csv"
MCQ_OUT = PROCESSED_DIR / "mcq_clean.csv"


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(value: Any) -> str:
    """Normalize whitespace, treat NaN/None as empty string."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip()

    if text.lower() in ("nan", "none", "nat"):
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def is_blank(value: Any) -> bool:
    return clean_text(value) == ""


_FEE_USD_RE = re.compile(r"đô la|usd", re.IGNORECASE)
_FEE_PLACEHOLDER_RE = re.compile(r"không công bố phí", re.IGNORECASE)
_CONSULAR_DOMAIN_RE = re.compile(r"lãnh sự", re.IGNORECASE)

_LEADING_BULLET_RE = re.compile(r"^[\*\-\+•]\s*")

_LEADING_ZERO_NUM_RE = re.compile(r"\b0(\d)\b")


def _fee_desc_group_key(desc: str) -> str:
    """
    Chuẩn hoá NHẸ mô tả phí chỉ để SO KHỚP khi gộp các dòng phí giống
    nhau về nội dung nhưng lệch nhau do lỗi gõ/định dạng ở nguồn - ví dụ
    đã kiểm chứng thực tế trong dữ liệu: cùng 1 mức phí ghi cho "Bưu
    chính" bị thừa 1 dấu ")" cuối câu so với dòng ghi cho "Trực tiếp", hoặc
    "3 tháng" (Bưu chính) vs "03 tháng" (Trực tiếp). Đây KHÔNG phải khác
    biệt nội dung thật, chỉ là cách gõ khác nhau ở nguồn.

    Hàm này chỉ dùng để tạo khoá gộp - text hiển thị ra rag_text vẫn lấy
    nguyên bản mô tả gốc (không sửa nội dung thật của người dùng thấy).
    """

    s = desc.strip()

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    while s.endswith(")") and s.count("(") < s.count(")"):
        s = s[:-1].rstrip()

    s = _LEADING_ZERO_NUM_RE.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    return s

# Các nhãn CẤU TRÚC hồ sơ (không phải tình huống/kịch bản thật) mà nguồn dữ
# liệu đôi khi nhét nhầm vào cột "trường hợp" - đã kiểm chứng: 19/286 thủ
# tục có trục hỏi MCQ "Trường hợp" mà TOÀN BỘ lựa chọn đều là các nhãn này
# (vd: "Giấy tờ phải nộp:", "Giấy tờ phải xuất trình:", "Lưu ý:"). Đây
# không phải các lựa chọn loại trừ nhau để hỏi người dân "bạn thuộc trường
# hợp nào?" - mà là các MỤC của cùng một bộ hồ sơ, cần làm tất cả chứ
# không phải chọn 1. Nếu không lọc, hệ thống hỏi-làm-rõ sẽ đặt câu hỏi vô
# nghĩa cho người dùng.
_STRUCTURAL_CASE_LABELS = {
    "giấy tờ phải nộp",
    "giấy tờ phải xuất trình",
    "lưu ý",
    "thành phần hồ sơ",
    "ghi chú",
}


def strip_bullet(text: str) -> str:
    """Bỏ 1 dấu đầu dòng (*, -, +, •) nếu cột trường hợp/lựa chọn gốc đã tự
    có sẵn, để không bị lặp thành "* * ..." khi code tự thêm dấu đầu dòng
    của riêng nó."""

    return _LEADING_BULLET_RE.sub("", text).strip()


def is_structural_case_label(text: str) -> bool:
    """True nếu case/lựa chọn thực chất là một MỤC cấu trúc hồ sơ (không
    phải một tình huống/kịch bản thật cần người dùng chọn 1 trong nhiều)."""

    normalized = strip_bullet(text).rstrip(":： ").strip().lower()
    return normalized in _STRUCTURAL_CASE_LABELS

_FEE_KIND_LABELS = {
    "FEE": "Phí",
    "SERVICE_FEE": "Lệ phí",
    "PRICE_LEVEL": "Khung giá",
}


def format_fee_amount(amount: Any, desc: str) -> str | None:
    """
    Chỉ trả về một con số phí cụ thể khi thực sự có số > 0 trong dữ liệu.

    QUAN TRỌNG: cột "số tiền" = 0 hoặc rỗng trong sheet "Phí" thường KHÔNG
    có nghĩa là miễn phí - phần lớn là do cổng không nhập số cụ thể (đã
    kiểm chứng: 2305 dòng loại FEE có số tiền=0, chỉ 3 dòng thật sự ghi
    rõ "miễn phí" trong mô tả). Vì vậy hàm này không bao giờ tự suy ra
    "0 đồng" khi không có số > 0.

    Cố tình KHÔNG tự nhận diện từ khoá "miễn phí" trong mô tả để gắn
    nhãn "Miễn phí" riêng: đã thử và thấy nhiều dòng chứa cụm này thực
    ra là miễn phí CÓ ĐIỀU KIỆN (vd: "miễn lệ phí cho người thuộc hộ
    nghèo, người khuyết tật..." trong khi mức phí chung vẫn do HĐND tỉnh
    quyết định) - tự gắn nhãn "Miễn phí" cho cả dòng sẽ lại tạo ra đúng
    loại kết luận sai mà README cảnh báo. An toàn hơn là để nguyên mô tả
    gốc tự nói rõ ngữ cảnh, không tổng hợp thành một nhãn ngắn.
    """

    has_number = (
        amount is not None
        and not (isinstance(amount, float) and pd.isna(amount))
    )

    if has_number:
        try:
            amount_f = float(amount)
        except (TypeError, ValueError):
            amount_f = None

        if amount_f is not None and amount_f > 0:
            currency = "USD" if _FEE_USD_RE.search(desc) else "đồng"
            return f"{amount_f:,.0f} {currency}".replace(",", ".")

    return None


# ============================================================
# 3. LOAD RAW SHEETS
# ============================================================

# Các sheet có 1 dòng cảnh báo/ghi chú xen giữa tiêu đề gốc và header thật
# (header=1 để bỏ qua dòng cảnh báo đó); "Thủ tục" và "Cách nộp" có header
# ngay ở dòng đầu (header=0).
_SHEET_SPECS = {
    "procedures": ("Thủ tục", 0),
    "mcq": ("MCQ", 1),
    "checklist": ("Checklist", 1),
    "fees": ("Phí", 1),
    "attachments": ("Tệp đính kèm", 1),
    "legal_basis": ("Căn cứ pháp lý", 1),
    "submission": ("Cách nộp", 0),
}

# Các cột hằng số / trùng lặp hoàn toàn ở sheet "Thủ tục", không mang
# thêm thông tin gì so với cột còn lại -> loại thẳng khi load, không để
# lọt vào bất kỳ bước xử lý nào phía sau.
_DROP_CONSTANT_OR_REDUNDANT_COLS = {
    "province",              # hằng số "(toàn quốc)" trên 100% dòng
    "status_meta",           # hằng số "present" trên 100% dòng
    "status_online",         # trùng 100% với has_online_submission
    "department_promulgate", # trùng 100% với issuing_agency
    # Các cột đếm gốc - không đáng tin (n_fees, n_cases) hoặc dư thừa vì
    # sẽ được tính lại chính xác hơn từ chính sheet con (n_checklist,
    # n_files, n_legal).
    "n_checklist",
    "n_fees",
    "n_files",
    "n_legal",
    "n_cases",
}


def load_sheets() -> dict[str, pd.DataFrame]:
    """
    Load mọi sheet với đúng dòng header, ép proc_id về str, loại các cột
    hằng số/trùng lặp ở sheet "Thủ tục", và loại dòng TRÙNG LẶP HOÀN TOÀN
    trong các sheet con (đã kiểm chứng đây là trùng lặp thật trong dữ
    liệu gốc, cùng proc_id, không phải trùng ngẫu nhiên giữa các thủ tục
    khác nhau).
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu gốc: {RAW_FILE}"
        )

    xls = pd.ExcelFile(RAW_FILE)

    sheets: dict[str, pd.DataFrame] = {}
    for key, (sheet_name, header_row) in _SHEET_SPECS.items():
        sheets[key] = xls.parse(sheet_name, header=header_row, dtype={"proc_id": str})

    for name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        if "proc_id" in df.columns:
            df["proc_id"] = df["proc_id"].astype(str).str.strip()

    # Sheet chính: bỏ cột hằng số/trùng lặp.
    sheets["procedures"] = sheets["procedures"].drop(
        columns=[c for c in _DROP_CONSTANT_OR_REDUNDANT_COLS if c in sheets["procedures"].columns]
    )

    # Sheet con: loại dòng trùng lặp hoàn toàn trước khi gộp/nối.
    for key in ("mcq", "checklist", "fees", "attachments", "legal_basis", "submission"):
        before = len(sheets[key])
        sheets[key] = sheets[key].drop_duplicates(ignore_index=True)
        removed = before - len(sheets[key])
        if removed:
            print(f"  [dedupe] {key}: bỏ {removed} dòng trùng lặp hoàn toàn ({before} -> {len(sheets[key])})")

    return sheets


# ============================================================
# 4. GROUP CHILD SHEETS BY proc_id
# ============================================================

def group_by_proc_id(
    df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for _, row in df.iterrows():
        proc_id = str(row.get("proc_id", "")).strip()

        if not proc_id or proc_id.lower() == "nan":
            continue

        grouped[proc_id].append(row.to_dict())

    return grouped


# ============================================================
# 5. SECTION BUILDERS
# ============================================================

def build_checklist_section(rows: list[dict[str, Any]]) -> str:
    """
    Group checklist items by "trường hợp" (case) when a procedure has more
    than one case (e.g. "Đối với xe đăng ký biển số Việt Nam" vs "... biển
    số nước ngoài"); otherwise a flat numbered list.

    Deliberately does NOT try to auto-split items where "độ dài" > 300 -
    inspection showed these are long descriptive sentences, not several
    documents concatenated with a reliable delimiter. Splitting blindly
    risks corrupting real content. Flagged as a caveat for the checkbox-UI
    team instead (see main()'s DATA QUALITY NOTES).

    Cột "bắt buộc" cố tình KHÔNG được dùng ở đây - toàn bộ 7656 dòng gốc
    đều có giá trị "—" (không ai điền), đưa vào chỉ tạo nhiễu.

    LƯU Ý: cột "trường hợp" ở một số thủ tục đã tự có sẵn dấu "*" ở đầu
    (vd: "* Giấy tờ phải nộp:") - nếu không bỏ dấu này trước khi tự thêm
    "* " cho case, kết quả sẽ bị lặp thành "* * Giấy tờ phải nộp:".
    """

    if not rows:
        return ""

    by_case: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        case = strip_bullet(clean_text(row.get("trường hợp")))
        item = clean_text(row.get("giấy tờ cần nộp"))

        if not item:
            continue

        # Số lượng bản chính / bản sao. Bỏ qua khi cả hai đều 0/trống
        # (thường là dòng tiêu đề/ghi chú).
        qty = []
        for label in ("bản chính", "bản sao"):
            v = clean_text(row.get(label))
            try:
                n = int(float(v)) if v else 0
            except ValueError:
                n = 0
            if n > 0:
                qty.append(f"{label}: {n}")
        if qty:
            item = f"{item} ({'; '.join(qty)})"

        by_case[case].append(item)

    lines: list[str] = []

    # Single, unnamed case -> flat list.
    if len(by_case) == 1 and "" in by_case:
        for i, item in enumerate(by_case[""], start=1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)

    for case, items in by_case.items():
        if case:
            lines.append(f"* {case}")
        for i, item in enumerate(items, start=1):
            lines.append(f"  {i}. {item}")

    return "\n".join(lines)


def build_fee_section(
    status_fees: str,
    rows: list[dict[str, Any]],
    domain: str = "",
) -> str:
    """
    Respect the 3-state semantics documented in the source README:
    status_fees == "absent_confirmed" must NEVER be rendered as "miễn
    phí" - it only means the portal did not publish a fee amount.

    Với status_fees == "present", mỗi dòng phí chỉ hiển thị số tiền cụ
    thể khi thật sự có số > 0 (xem format_fee_amount); nếu không, dòng
    vẫn được giữ lại với phần mô tả gốc (thường mới là thông tin thật:
    "mức phí do HĐND tỉnh quyết định", v.v.) thay vì bịa ra "0 đồng".
    Dòng nào vừa không có số vừa không có mô tả thì bị loại vì không
    mang thông tin gì.

    CẢNH BÁO ĐƠN VỊ TIỀN TỆ: cột "số tiền" gốc không có cột đơn vị riêng.
    Với các thủ tục thuộc lĩnh vực lãnh sự (vd: gia hạn tạm trú/thị thực
    cho người nước ngoài), các mức phí kiểu "5", "10", "25"... rất có
    thể là USD theo biểu phí lãnh sự thông thường, không phải VNĐ, nhưng
    mô tả không luôn ghi rõ "đô la". Không đủ căn cứ để tự khẳng định
    đơn vị, nên chỉ CẢNH BÁO cho nhóm lãnh sự thay vì mặc định "đồng"
    một cách im lặng như bản gốc.
    """

    status = clean_text(status_fees).lower()

    if status == "absent_confirmed":
        return (
            "Cổng dịch vụ công không công bố mức phí cho thủ tục này "
            "(không đồng nghĩa với miễn phí - người dân nên liên hệ cơ "
            "quan thực hiện để xác nhận trước khi nộp hồ sơ)."
        )

    if status != "present" or not rows:
        return "Chưa có thông tin về phí (chưa cào / cào lỗi)."

    # BƯỚC 1: dọn từng dòng, bỏ placeholder / dòng rỗng.
    items: list[tuple[str, str, str, str]] = []  # (kind, amount_text, desc, method)

    # Thủ tục lãnh sự: nguồn không nêu đơn vị tiền tệ -> chỉ giữ con số,
    # KHÔNG gắn chữ "đồng". (Muốn bỏ "đồng" cho MỌI thủ tục: đặt
    # is_consular = True.)
    is_consular = bool(_CONSULAR_DOMAIN_RE.search(domain))
    unit_removed = False

    for row in rows:
        amount = row.get("số tiền")
        desc = clean_text(row.get("mô tả mức phí"))
        method = clean_text(row.get("áp dụng cho cách nộp"))
        kind_raw = clean_text(row.get("loại"))

        # Phòng vệ: nhãn placeholder "(cổng KHÔNG công bố phí)" đáng lẽ
        # chỉ xuất hiện ở các thủ tục status_fees=absent_confirmed (đã
        # return ở nhánh trên); nếu lỡ có lẫn vào đây thì đây không phải
        # nội dung phí thật, bỏ qua để không lặp lại cảnh báo vô nghĩa.
        if _FEE_PLACEHOLDER_RE.search(kind_raw):
            continue

        amount_text = format_fee_amount(amount, desc) or ""
        if is_consular and amount_text.endswith(" đồng"):
            amount_text = amount_text[: -len(" đồng")]
            unit_removed = True

        if not amount_text and not desc:
            continue

        kind = _FEE_KIND_LABELS.get(kind_raw, kind_raw)
        # Tránh lặp kiểu "Lệ phí - Phí : theo quy định..." khi chính mô
        # tả đã tự mở đầu bằng đúng từ nhãn loại phí (dữ liệu gốc có
        # trường hợp "mô tả mức phí" tự chứa sẵn "Phí :" ở đầu).
        if desc.lower().startswith(kind.lower()):
            kind = ""

        items.append((kind, amount_text, desc, method))

    if not items:
        return (
            "Cổng có công bố phí nhưng chưa trích xuất được chi tiết mức "
            "phí - cần kiểm tra lại nguồn."
        )

    # BƯỚC 2: chỉ hiện nhãn loại (Phí/Lệ phí/Khung giá) trên từng dòng
    # khi CHÍNH thủ tục này có lẫn nhiều loại khác nhau (hiếm - 28/1407
    # thủ tục theo khảo sát thực tế); còn lại 1 thủ tục chỉ có 1 loại
    # phí duy nhất thì nhãn đó không cần lặp lại ở mọi dòng.
    kinds_present = {k for k, _, _, _ in items if k}
    show_kind = len(kinds_present) > 1

    # BƯỚC 3: gộp các dòng phí có CÙNG (loại, số tiền, mô tả) nhưng khác
    # hình thức áp dụng (Trực tiếp/Trực tuyến/Bưu chính) thành 1 dòng -
    # đây KHÔNG phải trùng lặp do lỗi xử lý: sheet "Phí" gốc thực sự có
    # nhiều dòng (mỗi dòng 1 hình thức nộp) share chung mức phí vì mức
    # phí không đổi theo hình thức nộp. Khoá gộp dùng bản mô tả đã chuẩn
    # hoá nhẹ (_fee_desc_group_key) để không bị lệch nhau chỉ vì lỗi gõ
    # ở nguồn (thừa dấu ")", "3" vs "03"...), nhưng text hiển thị vẫn lấy
    # nguyên bản mô tả gốc của dòng đầu tiên gặp trong nhóm.
    groups: dict[tuple[str, str, str], list] = {}
    order: list[tuple[str, str, str]] = []

    for kind, amount_text, desc, method in items:
        group_key = (kind, amount_text, _fee_desc_group_key(desc))
        if group_key not in groups:
            groups[group_key] = [desc, []]
            order.append(group_key)

        if method and method not in groups[group_key][1]:
            groups[group_key][1].append(method)

    # BƯỚC 4: chỉ nhắc "(áp dụng: ...)" trên từng dòng khi hình thức nộp
    # THỰC SỰ khác nhau giữa các mức phí trong CÙNG thủ tục (khảo sát
    # thực tế: chỉ 46/1407 thủ tục rơi vào trường hợp này). Còn lại, khi
    # mọi mức phí đều áp dụng chung 1 bộ hình thức nộp như nhau (đã có ở
    # mục "Cách nộp / thời hạn giải quyết" phía trên), nhắc lại trên mỗi
    # dòng phí chỉ gây dài dòng không cần thiết nên được bỏ hẳn.
    method_tuples = {tuple(m) for _, m in groups.values() if m}
    show_methods_per_line = len(method_tuples) > 1

    lines: list[str] = []

    for group_key in order:
        kind, amount_text, _ = group_key
        desc, methods = groups[group_key]

        # Bỏ dấu ")" thừa ở cuối do lỗi gõ ở nguồn (vd "nhập cảnh 1 lần)").
        while desc.endswith(")") and desc.count("(") < desc.count(")"):
            desc = desc[:-1].rstrip()

        if desc and amount_text:
            label = f"{desc}: {amount_text}"
        else:
            label = desc or amount_text

        if show_kind and kind:
            label = f"{kind} - {label}"

        if show_methods_per_line and methods:
            label = f"{label} (áp dụng: {' / '.join(methods)})"

        if label:
            lines.append(f"- {label}")

    if not lines:
        return (
            "Cổng có công bố phí nhưng chưa trích xuất được chi tiết mức "
            "phí - cần kiểm tra lại nguồn."
        )

    if unit_removed:
        lines.insert(
            0,
            "[Lưu ý: đây là thủ tục lãnh sự - nguồn KHÔNG nêu đơn vị tiền "
            "tệ nên các mức phí bên dưới chỉ ghi số; biểu phí lãnh sự "
            "thường tính bằng USD, cần xác minh lại trước khi công bố "
            "cho người dùng.]",
        )

    return "\n".join(lines)


def build_submission_section(rows: list[dict[str, Any]]) -> str:
    """
    Chỉ giữ 2 trường: "cách nộp" và "thời hạn". Mỗi hình thức nộp gom các
    thời hạn của nó (vd Trực tiếp: 1 ngày làm việc, 2 ngày làm việc); các
    hình thức có cùng danh sách thời hạn được gộp thành 1 dòng.
    """

    if not rows:
        return ""

    per_method: dict[str, list[str]] = {}
    for row in rows:
        method = clean_text(row.get("cách nộp"))
        deadline = clean_text(row.get("thời hạn"))
        if not method and not deadline:
            continue
        lst = per_method.setdefault(method, [])
        if deadline and deadline not in lst:
            lst.append(deadline)

    groups: dict[tuple[str, ...], list[str]] = {}
    for method, deadlines in per_method.items():
        groups.setdefault(tuple(deadlines), []).append(method)

    lines: list[str] = []
    for deadlines, methods in groups.items():
        name = " / ".join(m for m in methods if m)
        dl = ", ".join(deadlines)
        if name and dl:
            lines.append(f"- {name}: {dl}")
        elif name or dl:
            lines.append(f"- {name or dl}")

    return "\n".join(lines)


def build_attachments_section(rows: list[dict[str, Any]]) -> str:
    """
    Files have no public URL (must be fetched via an internal API using
    file_id) - list names/categories only, never fabricate a download
    link.
    """

    if not rows:
        return ""

    lines = []
    seen: set[str] = set()

    for row in rows:
        filename = clean_text(row.get("tên tệp"))
        category = clean_text(row.get("thuộc mục hồ sơ"))

        if not filename:
            continue

        piece = filename

        if category:
            piece += f" (thuộc: {category})"

        # Cùng 1 tệp có thể được liệt kê cho nhiều mục hồ sơ khác nhau
        # (hợp lệ) nhưng nếu piece giống hệt (cùng tệp + cùng mục) thì
        # không cần lặp lại lần nữa trong rag_text.
        if piece in seen:
            continue
        seen.add(piece)

        lines.append(f"- {piece}")

    return "\n".join(lines)


_LEGAL_CODE_RE = re.compile(
    r"\d+(?:\.\d+)?/(?:\d{4}/)?[A-ZĐ][A-ZĐ0-9]*(?:-[A-ZĐ0-9]+)*"
)

NO_LEGAL_CODE_TEXT = "Không có thông tin căn cứ pháp lý trong dữ liệu gốc"


def normalize_legal_code(code: str, name: str = "") -> str:
    """
    Trả về SỐ HIỆU văn bản sạch (vd "25/2025/NĐ-CP") từ cột "số hiệu văn
    bản". Khảo sát thực tế: 915/5660 dòng nguồn không phải số hiệu sạch
    mà lẫn tiền tố loại văn bản ("Nghị định số 25/2025/NĐ-CP"), hậu tố
    ngày ("101/2024/NĐ-CP ngày 29/7/2024"), khoảng trắng thừa
    ("105/2025/NĐ- CP") hoặc chỉ là "06". Quy tắc:
      1. Tìm mẫu số hiệu trong chính ô "số hiệu văn bản".
      2. Không thấy thì thử tìm trong "tên văn bản".
      3. Vẫn không thấy thì giữ NGUYÊN giá trị gốc (không bịa, không bỏ).
    """

    raw = clean_text(code)
    fixed = re.sub(r"\s*-\s*", "-", raw)

    for text in (fixed, re.sub(r"\s*-\s*", "-", clean_text(name))):
        m = _LEGAL_CODE_RE.search(text)
        if m:
            return m.group(0)

    return raw


def collect_legal_codes(rows: list[dict[str, Any]]) -> list[str]:
    """Danh sách số hiệu văn bản của 1 thủ tục, đã chuẩn hoá, bỏ trùng, giữ thứ tự."""

    seen: list[str] = []
    for row in rows:
        code = normalize_legal_code(
            row.get("số hiệu văn bản"), row.get("tên văn bản")
        )
        if code and code not in seen:
            seen.append(code)
    return seen


def build_legal_basis_section(rows: list[dict[str, Any]]) -> str:
    """
    This is where the REAL "law effective from" date lives - buried in
    free-text văn bản names (e.g. "... ngày 08 tháng 5 năm 2026 ..."),
    since decision_date in the main sheet is only the portal's publish
    date, not the legal document's date.

    Mỗi dòng luôn mở đầu bằng SỐ HIỆU VĂN BẢN đã chuẩn hoá.
    """

    if not rows:
        return ""

    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        name = clean_text(row.get("tên văn bản"))
        code = normalize_legal_code(row.get("số hiệu văn bản"), name)

        if (code, name) in seen:
            continue
        seen.add((code, name))

        # Nếu tên văn bản chỉ lặp lại số hiệu thì không in 2 lần.
        if name and name.lower() == clean_text(row.get("số hiệu văn bản")).lower():
            name = ""

        piece = " - ".join(p for p in [code, name] if p)

        if piece:
            lines.append(f"- {piece}")

    return "\n".join(lines)


def build_mcq_summary(rows: list[dict[str, Any]]) -> str:
    """
    Short summary noting a procedure has disambiguating cases, so a
    reader of procedures_clean.csv alone (without querying mcq_clean.csv)
    still knows clarifying questions exist. Full question/option detail
    lives in mcq_clean.csv for retrieval.py to use interactively.

    QUAN TRỌNG: loại bỏ các trục hỏi "giả" - đã kiểm chứng có 19/286 thủ
    tục mà trục "Trường hợp" thực ra KHÔNG phải các tình huống loại trừ
    nhau, mà là các MỤC cấu trúc hồ sơ bị gán nhầm vào cột "trường hợp"
    ở nguồn (vd: lựa chọn 1="Giấy tờ phải nộp:", lựa chọn 2="Giấy tờ phải
    xuất trình:", lựa chọn 3="Lưu ý:" - đây là 3 phần của CÙNG một bộ hồ
    sơ, người dân cần làm cả 3, không phải chọn 1 trong 3). Một trục chỉ
    được coi là "cần hỏi làm rõ" nếu có ít nhất 1 lựa chọn không phải là
    nhãn cấu trúc thuần tuý.
    """

    if not rows:
        return ""

    rows_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        axis = clean_text(r.get("trục hỏi"))
        # Bỏ qua dòng không có lựa chọn thật (rỗng/NaN) khi xét xem trục
        # này có phải toàn nhãn cấu trúc hay không - một dòng rỗng không
        # phải bằng chứng cho thấy đây là 1 tình huống thật.
        if axis and clean_text(r.get("lựa chọn")):
            rows_by_axis[axis].append(r)

    axes = sorted(
        axis
        for axis, axis_rows in rows_by_axis.items()
        if not all(
            is_structural_case_label(clean_text(r.get("lựa chọn")))
            for r in axis_rows
        )
    )

    if not axes:
        return ""

    return (
        "Thủ tục này có nhiều trường hợp khác nhau, cần làm rõ trước khi "
        "trả lời chi tiết. Trục cần hỏi: " + ", ".join(axes)
    )


# ============================================================
# 6. BUILD ONE MERGED DOCUMENT PER proc_id
# ============================================================

def build_rag_text(fields: dict[str, str]) -> str:
    """
    Chỉ gồm các trường được yêu cầu:
      Thủ tục   : name, requirements (Điều kiện), executing_agency
      Checklist : trường hợp, giấy tờ cần nộp, bản chính / bản sao
      Cách nộp  : cách nộp, thời hạn
      Phí       : số tiền, mô tả mức phí
      Căn cứ PL : số hiệu văn bản (LUÔN có, đặt ngay đầu để mọi chunk
                  truy xuất ra đều kèm số hiệu)
    Không có: trình tự thực hiện, kết quả, biểu mẫu, tên văn bản...
    """

    parts = [f"Tên thủ tục hành chính: {fields['name']}"]

    parts.append(f"Số hiệu văn bản: {fields['legal_codes']}")

    if fields["executing_agency"]:
        parts.append(f"Cơ quan thực hiện: {fields['executing_agency']}")

    if fields["requirements"]:
        parts.append(f"\nĐiều kiện thực hiện:\n{fields['requirements']}")

    if fields["checklist_text"]:
        parts.append(f"\nThành phần hồ sơ:\n{fields['checklist_text']}")

    if fields["submission_text"]:
        parts.append(f"\nCách nộp / thời hạn:\n{fields['submission_text']}")

    if fields["fee_text"]:
        parts.append(f"\nLệ phí:\n{fields['fee_text']}")

    return "\n".join(parts).strip()


def process(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:

    checklist_by_id = group_by_proc_id(sheets["checklist"])
    fees_by_id = group_by_proc_id(sheets["fees"])
    submission_by_id = group_by_proc_id(sheets["submission"])
    attachments_by_id = group_by_proc_id(sheets["attachments"])
    legal_by_id = group_by_proc_id(sheets["legal_basis"])
    mcq_by_id = group_by_proc_id(sheets["mcq"])

    out_rows = []

    for _, row in sheets["procedures"].iterrows():

        proc_id = str(row["proc_id"]).strip()

        fields = {
            "proc_id": proc_id,
            "name": clean_text(row.get("name")),
            "domain": clean_text(row.get("domain")),
            "executing_agency": clean_text(row.get("executing_agency")),
            "agency_levels": clean_text(row.get("agency_levels")),
            "subject_types": clean_text(row.get("subject_types")),
            "receiving_address": clean_text(row.get("receiving_address")),
            "processing_time_text": clean_text(
                row.get("processing_time_text")
            ),
            "portal_url": clean_text(row.get("portal_url")),
            "online_url": clean_text(row.get("online_url")),
            "has_online_submission": clean_text(
                row.get("has_online_submission")
            ),
            "requirements": clean_text(row.get("requirements")),
            "results": clean_text(row.get("results")),
            "description": clean_text(row.get("description")),
            "decision_number": clean_text(row.get("decision_number")),
            "issuing_agency": clean_text(row.get("issuing_agency")),
            # Ngày cổng công bố thủ tục - KHÔNG phải ngày văn bản luật có
            # hiệu lực (ngày luật thật nằm trong legal_basis_text).
            "decision_date_portal": clean_text(row.get("decision_date")),
            "source_updated_at": clean_text(row.get("source_updated_at")),
            "content_hash": clean_text(row.get("content_hash")),
            "status_fees": clean_text(row.get("status_fees")),
            "status_files": clean_text(row.get("status_files")),
        }

        fields["checklist_text"] = build_checklist_section(
            checklist_by_id.get(proc_id, [])
        )
        fields["fee_text"] = build_fee_section(
            fields["status_fees"], fees_by_id.get(proc_id, []), fields["domain"]
        )
        fields["submission_text"] = build_submission_section(
            submission_by_id.get(proc_id, [])
        )
        fields["attachments_text"] = build_attachments_section(
            attachments_by_id.get(proc_id, [])
        )
        fields["legal_basis_text"] = build_legal_basis_section(
            legal_by_id.get(proc_id, [])
        )
        fields["legal_codes"] = "; ".join(
            collect_legal_codes(legal_by_id.get(proc_id, []))
        ) or NO_LEGAL_CODE_TEXT
        fields["mcq_summary"] = build_mcq_summary(
            mcq_by_id.get(proc_id, [])
        )
        fields["has_mcq"] = bool(mcq_by_id.get(proc_id))
        fields["n_checklist_actual"] = len(checklist_by_id.get(proc_id, []))
        fields["n_fees_actual"] = len(fees_by_id.get(proc_id, []))
        fields["n_attachments_actual"] = len(
            attachments_by_id.get(proc_id, [])
        )
        fields["n_legal_actual"] = len(legal_by_id.get(proc_id, []))

        fields["rag_text"] = build_rag_text(fields)

        out_rows.append(fields)

    return pd.DataFrame(out_rows)


def build_mcq_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    QUAN TRỌNG: loại bỏ các nhóm (proc_id, trục hỏi) mà TOÀN BỘ lựa chọn
    chỉ là nhãn cấu trúc hồ sơ (xem is_structural_case_label) - đây không
    phải câu hỏi làm rõ thật, giữ lại sẽ khiến retrieval.py hỏi người
    dùng một câu vô nghĩa kiểu "bạn thuộc trường hợp 'Giấy tờ phải nộp'
    hay 'Giấy tờ phải xuất trình'?".
    """

    mcq = sheets["mcq"].copy()

    for col in ["proc_id", "tên thủ tục", "trục hỏi", "#", "lựa chọn", "số giấy tờ"]:
        if col not in mcq.columns:
            mcq[col] = ""

    mcq["tên thủ tục"] = mcq["tên thủ tục"].map(clean_text)
    mcq["trục hỏi"] = mcq["trục hỏi"].map(clean_text)
    mcq["lựa chọn"] = mcq["lựa chọn"].map(clean_text)

    mcq = mcq[mcq["proc_id"].astype(str).str.strip() != ""]
    mcq = mcq[mcq["lựa chọn"] != ""]

    is_structural = mcq["lựa chọn"].map(is_structural_case_label)
    bogus_group_all_structural = (
        is_structural.groupby([mcq["proc_id"], mcq["trục hỏi"]]).transform("all")
    )
    n_removed_rows = int(bogus_group_all_structural.sum())
    if n_removed_rows:
        print(
            f"  [mcq] bỏ {n_removed_rows} dòng thuộc các trục hỏi 'giả' "
            f"(toàn bộ lựa chọn chỉ là nhãn cấu trúc hồ sơ, không phải "
            f"tình huống thật cần hỏi)"
        )
    mcq = mcq[~bogus_group_all_structural]

    return mcq[["proc_id", "tên thủ tục", "trục hỏi", "#", "lựa chọn", "số giấy tờ"]]


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CSDL THỦ TỤC HÀNH CHÍNH - PREPROCESSING")
    print("=" * 70)
    print(f"Input file : {RAW_FILE}")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy: {RAW_FILE}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Đang nạp & loại trùng lặp các sheet con...")
    sheets = load_sheets()

    print()
    print(f"Thủ tục     : {len(sheets['procedures'])} dòng")
    print(f"MCQ         : {len(sheets['mcq'])} dòng")
    print(f"Checklist   : {len(sheets['checklist'])} dòng")
    print(f"Phí         : {len(sheets['fees'])} dòng")
    print(f"Tệp đính kèm: {len(sheets['attachments'])} dòng")
    print(f"Căn cứ PL   : {len(sheets['legal_basis'])} dòng")
    print(f"Cách nộp    : {len(sheets['submission'])} dòng")

    procedures_df = process(sheets)
    mcq_df = build_mcq_table(sheets)
    codes_by_id = dict(zip(procedures_df["proc_id"], procedures_df["legal_codes"]))
    mcq_df["legal_codes"] = (
        mcq_df["proc_id"].astype(str).str.strip().map(codes_by_id).fillna(NO_LEGAL_CODE_TEXT)
    )

    OUT_COLS = [
        "proc_id", "name", "executing_agency", "requirements",
        "checklist_text", "submission_text", "fee_text",
        "receiving_address",
        "legal_codes", "has_mcq", "rag_text",
    ]
    procedures_df[OUT_COLS].to_csv(
        PROCEDURES_OUT, index=False, encoding="utf-8-sig"
    )
    mcq_df.to_csv(MCQ_OUT, index=False, encoding="utf-8-sig")

    print()
    print(f"Rows out (procedures): {len(procedures_df)}")
    print(f"Rows out (mcq)       : {len(mcq_df)}")
    print(f"Cột output (procedures): {len(procedures_df.columns)}")
    print(f"Saved: {PROCEDURES_OUT}")
    print(f"Saved: {MCQ_OUT}")

    n_no_fee_data = (
        procedures_df["status_fees"].str.lower() == "absent_confirmed"
    ).sum()
    n_multi_agency_names = (
        procedures_df.groupby("name")["proc_id"].nunique() > 1
    ).sum()
    n_fee_present_but_unclear = (
        procedures_df["fee_text"].str.startswith("Cổng có công bố phí nhưng")
    ).sum()

    print()
    print("=" * 70)
    print("DATA QUALITY NOTES")
    print("=" * 70)
    print(
        f"- {n_no_fee_data} thủ tục có status_fees=absent_confirmed "
        f"(không công bố phí - ĐÃ được ghi rõ trong rag_text, không suy "
        f"diễn thành miễn phí)."
    )
    print(
        f"- {n_fee_present_but_unclear} thủ tục có status_fees=present "
        f"nhưng KHÔNG có dòng phí nào chứa số tiền>0 hay mô tả hữu ích "
        f"(sau khi loại bỏ placeholder) - rag_text ghi rõ 'cần kiểm tra "
        f"lại nguồn' thay vì bịa ra một mức phí."
    )
    print(
        f"- {n_multi_agency_names} tên thủ tục bị trùng giữa NHIỀU "
        f"proc_id khác nhau (do nhiều tỉnh/cơ quan cùng ban hành thủ tục "
        f"trùng tên) - đã giữ TÁCH RIÊNG theo proc_id, không gộp."
    )
    print(
        "- Checklist có 'độ dài' > 300 ký tự chưa được tự động tách nhỏ "
        "(không tìm được delimiter đáng tin cậy) - đội UI cần tự xử lý "
        "khi render checkbox, theo đúng cảnh báo trong sheet README gốc."
    )


if __name__ == "__main__":
    main()