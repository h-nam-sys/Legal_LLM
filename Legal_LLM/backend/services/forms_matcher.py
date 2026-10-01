import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase, strip punctuation, collapse whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

FORMS_CATALOG = [
    {
        "id": "thu_tuc_xac_nhan_tinh_trang_ho",
        "keywords": ["thủ tục xác nhận tình trạng hôn nhân"],
        "filename": "xac_nhan_tinh_trang_hon_nhan.doc",
        "display_name": "Thủ tục xác nhận tình trạng hôn nhân"
    },
    {
        "id": "thu_tuc_ang_ky_khai_sinh",
        "keywords": ["thủ tục đăng ký khai sinh"],
        "filename": "dang_ky_khai_sinh.doc",
        "display_name": "Thủ tục đăng ký khai sinh"
    },
    {
        "id": "thu_tuc_ang_ky_khai_tu",
        "keywords": ["thủ tục đăng ký khai tử"],
        "filename": "dang_ky_khai_tu.doc",
        "display_name": "Thủ tục đăng ký khai tử"
    },
    {
        "id": "thu_tuc_ang_ky_ket_hon",
        "keywords": ["thủ tục đăng ký kết hôn"],
        "filename": "dang_ky_ket_hon.docx",
        "display_name": "Thủ tục đăng ký kết hôn"
    },
    {
        "id": "thu_tuc_nhan_cha_me_con",
        "keywords": ["thủ tục nhận cha mẹ con"],
        "filename": "nhan_cha_me_con.doc",
        "display_name": "Thủ tục nhận cha mẹ con"
    },
    {
        "id": "ang_ky_ket_hon_co_yeu_to_nuoc",
        "keywords": ["đăng ký kết hôn có yếu tố nước ngoài"],
        "filename": "dang_ky_ket_hon_co_yeu_to_nuoc_ngoai.doc",
        "display_name": "Đăng ký kết hôn có yếu tố nước ngoài"
    },
    {
        "id": "noi_quy_lao_ong",
        "keywords": ["nội quy lao động"],
        "filename": "noi_quy_lao_dong.doc",
        "display_name": "Nội quy lao động"
    },
    {
        "id": "thuc_hien_ieu_chinh_thoi_hu",
        "keywords": ["thực hiện, điều chỉnh, thôi hưởng trợ cấp hưu trí xã hội"],
        "filename": "thoi_huong_tro_cap_huu_tri_xa_hoi.docx",
        "display_name": "Thực hiện, điều chỉnh, thôi hưởng trợ cấp hưu trí xã hội"
    },
    {
        "id": "xac_inh_xac_inh_lai_muc_o",
        "keywords": ["xác định, xác định lại mức độ khuyết tật và cấp giấy xác nhận khuyết tật"],
        "filename": "xac_dinh_khuyet_tat.docx",
        "display_name": "Xác định, xác định lại mức độ khuyết tật và cấp Giấy xác nhận khuyết tật"
    },
    {
        "id": "thu_tuc_ho_tro_chi_phi_khuyen",
        "keywords": ["thủ tục hỗ trợ chi phí khuyến khích hoả táng đối với người dân có hộ khẩu thành phố hồ chí minh"],
        "filename": "ho_tro_chi_phi_khuyen_khich_hoa_tang.doc",
        "display_name": "Thủ tục hỗ trợ chi phí khuyến khích hoả táng đối với người dân có hộ khẩu Thành phố Hồ Chí Minh"
    },
    {
        "id": "giai_quyet_che_o_tro_cap_tho",
        "keywords": ["giải quyết chế độ trợ cấp thờ cúng liệt sĩ."],
        "filename": "giai_quyet_che_do_tho_cung_liet_si.doc",
        "display_name": "Giải quyết chế độ trợ cấp thờ cúng liệt sĩ."
    },
    {
        "id": "cap_oi_bang_to_quoc_ghi_cong",
        "keywords": ["cấp đổi bằng “tổ quốc ghi công”"],
        "filename": "cap_doi_bang_To_quoc_ghi_cong.docx",
        "display_name": "Cấp đổi Bằng “Tổ quốc ghi công”"
    },
    {
        "id": "cap_giay_xac_nhan_than_nhan_cu",
        "keywords": ["cấp giấy xác nhận thân nhân của người có công"],
        "filename": "xac_nhan_than_nhan_nguoi_co_cong.docx",
        "display_name": "Cấp giấy xác nhận thân nhân của người có công"
    },
    {
        "id": "huong_tro_cap_khi_nguoi_co_con",
        "keywords": ["hưởng trợ cấp khi người có công đang hưởng trợ cấp ưu đãi từ trần"],
        "filename": "huong_tro_cap_uu_dai_tu_tran.doc",
        "display_name": "Hưởng trợ cấp khi người có công đang hưởng trợ cấp ưu đãi từ trần"
    },
    {
        "id": "ho_tro_chi_phi_khuyen_khich_ho",
        "keywords": ["hỗ trợ chi phí khuyến khích hỏa táng"],
        "filename": "ho_tro_chi_phi_khuyen_khich_hoa_tang.doc",
        "display_name": "Hỗ trợ chi phí khuyến khích hỏa táng"
    },
    {
        "id": "ho_tro_chi_phi_mai_tang_cho_o",
        "keywords": ["hỗ trợ chi phí mai táng cho đối tượng bảo trợ xã hội"],
        "filename": "ho_tro_chi_phi_mai_tang.docx",
        "display_name": "Hỗ trợ chi phí mai táng cho đối tượng bảo trợ xã hội"
    },
    {
        "id": "thuc_hien_ieu_chinh_thoi_hu_xa_hoi",
        "keywords": ["thực hiện, điều chỉnh, thôi hưởng trợ cấp xã hội hàng tháng, hỗ trợ kinh phí chăm sóc, nuôi dưỡng hàng tháng"],
        "filename": "thoi_huong_tro_cap_huu_tri_xa_hoi.docx",
        "display_name": "Thực hiện, điều chỉnh, thôi hưởng trợ cấp xã hội hàng tháng, hỗ trợ kinh phí chăm sóc, nuôi dưỡng hàng tháng"
    },
    {
        "id": "ho_tro_chi_phi_mai_tang_oi_vo",
        "keywords": ["hỗ trợ chi phí mai táng đối với đối tượng hưởng trợ cấp hưu trí xã hội"],
        "filename": "ho_tro_chi_phi_mai_tang.docx",
        "display_name": "Hỗ trợ chi phí mai táng đối với đối tượng hưởng trợ cấp hưu trí xã hội"
    },
    {
        "id": "giai_quyet_che_o_ho_tro_e_th",
        "keywords": ["giải quyết chế độ hỗ trợ để theo học đến trình độ đại học tại các cơ sở giáo dục thuộc hệ thống giáo dục quốc dân"],
        "filename": "giai_quyet_che_do_uu_dai_trong_giao_duc.doc",
        "display_name": "Giải quyết chế độ hỗ trợ để theo học đến trình độ đại học tại các cơ sở giáo dục thuộc hệ thống giáo dục quốc dân"
    },
    {
        "id": "giai_quyet_che_o_nguoi_hoat",
        "keywords": ["giải quyết chế độ người hoạt động kháng chiến giải phóng dân tộc, bảo vệ tổ quốc và làm nghĩa vụ quốc tế"],
        "filename": "giai_quyet_che_do_nguoi_hoat_dong_khang_chien.doc",
        "display_name": "Giải quyết chế độ người hoạt động kháng chiến giải phóng dân tộc, bảo vệ tổ quốc và làm nghĩa vụ quốc tế"
    },
    {
        "id": "ho_so_cap_giay_phep_xay_dung",
        "keywords": ["hồ sơ cấp giấy phép xây dựng"],
        "filename": "cap_giay_phep_xay_dung.doc",
        "display_name": "HỒ SƠ CẤP GIẤY PHÉP XÂY DỰNG"
    },
    {
        "id": "thong_tin_quy_hoach",
        "keywords": ["thông tin quy hoạch"],
        "filename": "cung_cap_thong_tin_quy_hoach.doc",
        "display_name": "THÔNG TIN QUY HOẠCH"
    },
    {
        "id": "cap_giay_phep_xay_dung",
        "keywords": ["cấp giấy phép xây dựng"],
        "filename": "cap_giay_phep_xay_dung.doc",
        "display_name": "CẤP GIẤY PHÉP XÂY DỰNG"
    },
    {
        "id": "vay_von_ho_tro_tao_viec_lam_d",
        "keywords": ["vay vốn hỗ trợ tạo việc làm, duy trì và mở rộng việc làm từ quỹ quốc gia về việc làm đối với người lao động"],
        "filename": "vay_von_ho_tro_viec_lam.doc",
        "display_name": "Vay vốn hỗ trợ tạo việc làm, duy trì và mở rộng việc làm từ Quỹ quốc gia về việc làm đối với người lao động"
    },
    {
        "id": "chuyen_truong_oi_voi_hoc_sinh",
        "keywords": ["chuyển trường đối với học sinh thcs"],
        "filename": "chuyen_truong_THCS.docx",
        "display_name": "Chuyển trường đối với học sinh THCS"
    },
    {
        "id": "thong_bao_khoi_cong",
        "keywords": ["thông báo khởi công"],
        "filename": "thong_bao_khoi_cong.docx",
        "display_name": "THÔNG BÁO KHỞI CÔNG"
    },
    {
        "id": "ang_ky_at_ai_tai_san_gan_l",
        "keywords": ["đăng ký đất đai, tài sản gắn liền với đất lần đầu"],
        "filename": "dang_ky_dat_dai_va_tai_san_gan_lien_voi_dat.docx",
        "display_name": "ĐĂNG KÝ ĐẤT ĐAI, TÀI SẢN GẮN LIỀN VỚI ĐẤT LẦN ĐẦU"
    },
    {
        "id": "vay_von_ho_tro_tao_viec_lam_dn",
        "keywords": ["vay vốn hỗ trợ tạo việc làm, duy trì và mở rộng việc làm từ quỹ quốc gia về việc làm đối với cơ sở sản xuất, kinh doanh."],
        "filename": "vay_von_ho_tro_viec_lam.doc",
        "display_name": "Vay vốn hỗ trợ tạo việc làm, duy trì và mở rộng việc làm từ Quỹ quốc gia về việc làm đối với cơ sở sản xuất, kinh doanh."
    },
    {
        "id": "ho_so_cap_so_nha",
        "keywords": ["hồ sơ cấp số nhà"],
        "filename": "xac_nhan_so_nha.docx",
        "display_name": "HỒ SƠ CẤP SỐ NHÀ"
    },
    {
        "id": "xac_nhan_vi_tri_nha__at",
        "keywords": ["xác nhận vị trí nhà - đất"],
        "filename": "xac_nhan_vi_tri_nha_dat.doc",
        "display_name": "XÁC NHẬN VỊ TRÍ NHÀ - ĐẤT"
    },
    {
        "id": "xac_nhan_tinh_trang_nha_o",
        "keywords": ["xác nhận tình trạng nhà ở"],
        "filename": "xac_nhan_tinh_trang_nha_o.docx",
        "display_name": "XÁC NHẬN TÌNH TRẠNG NHÀ Ở"
    },
    {
        "id": "thu_tuc_ang_ky_thanh_lap_ho_k",
        "keywords": ["thủ tục đăng ký thành lập hộ kinh doanh"],
        "filename": "dang_ky_ho_kinh_doanh.docx",
        "display_name": "Thủ tục Đăng ký thành lập hộ kinh doanh"
    },
    {
        "id": "ang_ky_thay_oi_noi_dung_ang",
        "keywords": ["đăng ký thay đổi nội dung đăng ký hộ kinh doanh"],
        "filename": "dang_ky_thay_doi_noi_dung_dang_ky_ho_kinh_doanh.doc",
        "display_name": "Đăng ký thay đổi nội dung đăng ký hộ kinh doanh"
    },
    {
        "id": "tam_ngung_kinh_doanhtiep_tuc",
        "keywords": ["tạm ngừng kinh doanh/tiếp tục kinh doanh trước thời hạn đã thông báo của hộ kinh doanh"],
        "filename": "thong_bao_tam_dung_ho_kinh_doanh.doc",
        "display_name": "Tạm ngừng kinh doanh/tiếp tục kinh doanh trước thời hạn đã thông báo của hộ kinh doanh"
    },
    {
        "id": "cham_dut_hoat_ong_ho_kinh_doa",
        "keywords": ["chấm dứt hoạt động hộ kinh doanh"],
        "filename": "cham_dut_hoat_dong_ho_kinh_doanh.docx",
        "display_name": "Chấm dứt hoạt động hộ kinh doanh"
    },
    {
        "id": "cap_lai_giay_chung_nhan_ang_k",
        "keywords": ["cấp lại giấy chứng nhận đăng ký hộ kinh doanh"],
        "filename": "cap_lai_giay_chung_nhan_dang_ky_ho_kinh_doanh.doc",
        "display_name": "Cấp lại Giấy chứng nhận đăng ký hộ kinh doanh"
    },
    {
        "id": "thu_tuc_xet_cap_hoc_bong_chin",
        "keywords": ["thủ tục xét, cấp học bổng chính sách"],
        "filename": "cap_hoc_bong_chinh_sach.docx",
        "display_name": "Thủ tục xét, cấp học bổng chính sách"
    },
    {
        "id": "chung_thuc_chu_ky__chung_thuc",
        "keywords": ["chứng thực chữ ký - chứng thực hợp đồng giao dịch/di chúc"],
        "filename": "chung_thuc_chu_ky.docx",
        "display_name": "chứng thực chữ ký - chứng thực Hợp đồng giao dịch/Di chúc"
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
