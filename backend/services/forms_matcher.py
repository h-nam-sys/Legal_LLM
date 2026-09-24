import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase, strip punctuation, collapse whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

FORMS_CATALOG = [
    {
        "id": "tinh_trang_hon_nhan",
        "keywords": ["tình trạng hôn nhân", "xác nhận hôn nhân", "giấy độc thân"],
        "filename": "xac_nhan_tinh_trang_hon_nhan.docx",
        "display_name": "Tờ khai cấp Giấy xác nhận tình trạng hôn nhân"
    },
    {
        "id": "ket_hon",
        "keywords": ["kết hôn"],
        "filename": "dang_ky_ket_hon.docx",
        "display_name": "Tờ khai đăng ký kết hôn"
    },
    {
        "id": "khai_sinh",
        "keywords": ["khai sinh", "giấy khai sinh"],
        "filename": "dang_ky_khai_sinh.docx",
        "display_name": "Tờ khai đăng ký khai sinh"
    },
    {
        "id": "khai_tu",
        "keywords": ["khai tử", "chứng tử", "giấy báo tử"],
        "filename": "dang_ky_khai_tu.docx",
        "display_name": "Tờ khai đăng ký khai tử"
    },
    {
        "id": "nhan_cha_me_con",
        "keywords": ["nhận cha", "mẹ con", "nhận con"],
        "filename": "to_khai_nhan_cha_me_con.docx",
        "display_name": "Tờ khai nhận cha, mẹ, con"
    },
    {
        "id": "ho_kinh_doanh",
        "keywords": ["hộ kinh doanh", "thành lập hộ kinh doanh"],
        "filename": "thanh_lap_ho_kinh_doanh.docx",
        "display_name": "Giấy đề nghị đăng ký hộ kinh doanh"
    },
    {
        "id": "giay_phep_xay_dung",
        "keywords": ["giấy phép xây dựng", "cấp phép xây dựng"],
        "filename": "cap_phep_xay_dung.docx",
        "display_name": "Đơn đề nghị cấp giấy phép xây dựng"
    },
    {
        "id": "cap_so_nha",
        "keywords": ["cấp số nhà", "đổi số nhà"],
        "filename": "cap_so_nha.docx",
        "display_name": "Đơn đề nghị cấp số nhà"
    },
    {
        "id": "dat_dai_lan_dau",
        "keywords": ["đất đai", "tài sản gắn liền với đất", "sổ đỏ lần đầu"],
        "filename": "dang_ky_dat_dai_lan_dau.docx",
        "display_name": "Đơn đăng ký đất đai, tài sản gắn liền với đất"
    }
]

def find_matching_form(procedure_title: str, query: str = "") -> dict | None:
    """Matches form based on procedure title first, falling back to user query."""
    normalized_title = normalize_text(procedure_title)
    normalized_query = normalize_text(query)

    # Pass 1: Match against the RAG detected title
    for form in FORMS_CATALOG:
        for kw in form["keywords"]:
            if normalize_text(kw) in normalized_title:
                return form

    # Pass 2: Fallback to matching against user's actual prompt
    for form in FORMS_CATALOG:
        for kw in form["keywords"]:
            if normalize_text(kw) in normalized_query:
                return form

    return None
