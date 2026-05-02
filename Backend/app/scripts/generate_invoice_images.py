"""
generate_invoice_images.py - Tạo ảnh hóa đơn VAT Việt Nam để test OCR
======================================================================
Sử dụng Pillow để render hóa đơn thật với font hệ thống (Times New Roman).
Mỗi hóa đơn có cấu trúc chuẩn: header, thông tin bên bán/mua, bảng hàng hóa,
thuế GTGT, tổng cộng, chữ ký.
"""
from __future__ import annotations

import os
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Fonts ──────────────────────────────────────────────────────
FONT_DIR = Path("C:/Windows/Fonts")
FONT_REGULAR = str(FONT_DIR / "times.ttf")
FONT_BOLD = str(FONT_DIR / "timesbd.ttf")
FONT_ITALIC = str(FONT_DIR / "timesi.ttf")
FONT_ARIAL = str(FONT_DIR / "arial.ttf")
FONT_ARIAL_BOLD = str(FONT_DIR / "arialbd.ttf")


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ── Sample data pools ─────────────────────────────────────────
SELLERS = [
    ("CÔNG TY TNHH THƯƠNG MẠI ABC", "0101234567", "Số 12, Phố Trần Hưng Đạo, Hoàn Kiếm, Hà Nội"),
    ("CÔNG TY CP ĐẦU TƯ VÀ XÂY DỰNG MINH PHÁT", "0312345678", "45 Nguyễn Huệ, Quận 1, TP.HCM"),
    ("CÔNG TY TNHH SẢN XUẤT THIÊN LONG", "3600123456", "KCN Biên Hòa 2, Đồng Nai"),
    ("CÔNG TY CP DỊCH VỤ TƯ VẤN THUẾ VIỆT", "0108765432", "Tầng 10, Tòa Keangnam, Hà Nội"),
    ("CÔNG TY TNHH LOGISTICS TOÀN CẦU", "0314567890", "Lô C3, KCN Tân Bình, TP.HCM"),
]

BUYERS = [
    ("CÔNG TY CP CÔNG NGHỆ THÔNG TIN SMART", "0109876543", "88 Láng Hạ, Đống Đa, Hà Nội"),
    ("CÔNG TY TNHH THỰC PHẨM SẠCH XANH", "0317654321", "12 Lê Lợi, Quận 3, TP.HCM"),
    ("DOANH NGHIỆP TƯ NHÂN VẬN TẢI HOÀNG ANH", "2001234567", "156 Trần Phú, TP Đà Nẵng"),
    ("CÔNG TY CP BẤT ĐỘNG SẢN PHÚC LỘC", "0111223344", "27 Trần Duy Hưng, Cầu Giấy, Hà Nội"),
    ("HTX NÔNG NGHIỆP ĐỒNG XANH", "1400567890", "Xã Tân Phú, Huyện Củ Chi, TP.HCM"),
]

ITEMS_POOL = [
    [("Dịch vụ tư vấn quản lý doanh nghiệp", 1, 50_000_000),
     ("Phí đào tạo nhân sự nội bộ", 2, 15_000_000)],
    [("Xi măng Nghi Sơn PCB40 (tấn)", 50, 1_850_000),
     ("Thép Hòa Phát D10 (tấn)", 20, 14_500_000),
     ("Cát xây dựng (m³)", 100, 250_000)],
    [("Laptop Dell Latitude 5540", 10, 22_000_000),
     ("Màn hình Dell 27\" UltraSharp", 10, 8_500_000),
     ("Chuột không dây Logitech MX", 20, 1_200_000)],
    [("Gạo ST25 (kg)", 5000, 28_000),
     ("Dầu ăn Neptune 5L (chai)", 200, 135_000),
     ("Đường Biên Hòa (kg)", 1000, 22_000)],
    [("Vận chuyển hàng hóa tuyến HN-HCM", 5, 8_500_000),
     ("Dịch vụ kho bãi tháng 03/2026", 1, 25_000_000)],
    [("Dịch vụ marketing online tổng thể", 1, 120_000_000),
     ("Thiết kế bộ nhận diện thương hiệu", 1, 45_000_000)],
]


def _random_invoice_number() -> str:
    series = random.choice(["AA", "AB", "AC", "TT", "KH"])
    year = random.choice(["23", "24", "25", "26"])
    seq = random.randint(1, 9999999)
    return f"{series}/{year}E-{seq:07d}"


def _random_date() -> date:
    base = date(2025, 1, 1)
    return base + timedelta(days=random.randint(0, 500))


# ── Drawing helpers ───────────────────────────────────────────

def _draw_centered(draw: ImageDraw.ImageDraw, y: int, text: str,
                   font: ImageFont.FreeTypeFont, fill: str,
                   width: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (width - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)
    return bbox[3] - bbox[1]


def _draw_line(draw: ImageDraw.ImageDraw, y: int, x1: int, x2: int,
               fill: str = "#000", width: int = 1):
    draw.line([(x1, y), (x2, y)], fill=fill, width=width)


def _format_money(amount: float) -> str:
    return f"{int(amount):,}".replace(",", ".")


# ── Main generator ────────────────────────────────────────────

def generate_invoice(
    output_path: str | Path,
    *,
    seller: tuple | None = None,
    buyer: tuple | None = None,
    items: list | None = None,
    vat_rate: int = 10,
    add_stamp: bool = False,
    add_noise: bool = False,
) -> dict[str, Any]:
    """
    Generate a realistic Vietnamese VAT invoice image.

    Returns dict with invoice metadata for verification.
    """
    W, H = 1200, 1700
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title = _font(FONT_BOLD, 28)
    f_subtitle = _font(FONT_BOLD, 22)
    f_header = _font(FONT_BOLD, 18)
    f_body = _font(FONT_REGULAR, 17)
    f_body_bold = _font(FONT_BOLD, 17)
    f_small = _font(FONT_REGULAR, 14)
    f_small_bold = _font(FONT_BOLD, 14)
    f_red = _font(FONT_BOLD, 16)

    seller = seller or random.choice(SELLERS)
    buyer = buyer or random.choice(BUYERS)
    items = items or random.choice(ITEMS_POOL)
    inv_num = _random_invoice_number()
    inv_date = _random_date()

    margin_l, margin_r = 60, W - 60
    y = 40

    # ── Header ──
    _draw_centered(draw, y, "HÓA ĐƠN GIÁ TRỊ GIA TĂNG", f_title, "#CC0000", W)
    y += 38
    _draw_centered(draw, y, "(VAT INVOICE)", f_small, "#CC0000", W)
    y += 28

    # Form & serial
    draw.text((margin_l, y), f"Mẫu số (Form): 01GTKT0/001", font=f_small, fill="#CC0000")
    draw.text((W - 400, y), f"Ký hiệu (Serial): {inv_num.split('-')[0] if '-' in inv_num else inv_num[:7]}", font=f_small, fill="#CC0000")
    y += 22
    draw.text((margin_l, y), f"Số (No.): {inv_num}", font=f_small_bold, fill="#000")
    y += 22

    date_str = f"Ngày {inv_date.day} tháng {inv_date.month} năm {inv_date.year}"
    _draw_centered(draw, y, date_str, f_small, "#000", W)
    y += 35

    _draw_line(draw, y, margin_l, margin_r, "#000", 2)
    y += 15

    # ── Seller info ──
    draw.text((margin_l, y), "Đơn vị bán hàng (Seller):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), seller[0], font=f_body, fill="#000")
    y += 26
    draw.text((margin_l, y), "Mã số thuế (Tax code):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), seller[1], font=f_body, fill="#000")
    y += 26
    draw.text((margin_l, y), "Địa chỉ (Address):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), seller[2], font=f_body, fill="#000")
    y += 35

    # ── Buyer info ──
    draw.text((margin_l, y), "Đơn vị mua hàng (Buyer):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), buyer[0], font=f_body, fill="#000")
    y += 26
    draw.text((margin_l, y), "Mã số thuế (Tax code):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), buyer[1], font=f_body, fill="#000")
    y += 26
    draw.text((margin_l, y), "Địa chỉ (Address):", font=f_body_bold, fill="#000")
    draw.text((margin_l + 280, y), buyer[2], font=f_body, fill="#000")
    y += 40

    # ── Items table ──
    col_stt = margin_l
    col_name = margin_l + 50
    col_qty = 620
    col_unit = 740
    col_price = 870
    col_total = 1020
    row_h = 32

    # Header row
    _draw_line(draw, y, margin_l, margin_r, "#000", 2)
    y += 5
    draw.text((col_stt, y), "STT", font=f_header, fill="#000")
    draw.text((col_name, y), "Tên hàng hóa, dịch vụ", font=f_header, fill="#000")
    draw.text((col_qty, y), "Số lượng", font=f_header, fill="#000")
    draw.text((col_unit, y), "ĐVT", font=f_header, fill="#000")
    draw.text((col_price, y), "Đơn giá", font=f_header, fill="#000")
    draw.text((col_total, y), "Thành tiền", font=f_header, fill="#000")
    y += row_h
    _draw_line(draw, y, margin_l, margin_r, "#000", 1)
    y += 5

    subtotal = 0.0
    for idx, (name, qty, price) in enumerate(items, 1):
        line_total = qty * price
        subtotal += line_total
        draw.text((col_stt, y), str(idx), font=f_body, fill="#000")
        # Truncate long names
        display_name = name[:40]
        draw.text((col_name, y), display_name, font=f_body, fill="#000")
        draw.text((col_qty, y), f"{qty:,}".replace(",", "."), font=f_body, fill="#000")
        draw.text((col_unit, y), "Bộ" if qty == 1 else "Cái", font=f_body, fill="#000")
        draw.text((col_price, y), _format_money(price), font=f_body, fill="#000")
        draw.text((col_total, y), _format_money(line_total), font=f_body, fill="#000")
        y += row_h

    _draw_line(draw, y, margin_l, margin_r, "#000", 1)
    y += 10

    # ── Totals ──
    vat_amount = subtotal * vat_rate / 100
    grand_total = subtotal + vat_amount

    draw.text((col_price - 200, y), "Cộng tiền hàng (Subtotal):", font=f_body_bold, fill="#000")
    draw.text((col_total, y), _format_money(subtotal), font=f_body_bold, fill="#000")
    y += 28
    draw.text((col_price - 200, y), f"Thuế suất GTGT (VAT rate): {vat_rate}%", font=f_body, fill="#000")
    y += 28
    draw.text((col_price - 200, y), "Tiền thuế GTGT (VAT amount):", font=f_body_bold, fill="#000")
    draw.text((col_total, y), _format_money(vat_amount), font=f_body_bold, fill="#000")
    y += 28

    _draw_line(draw, y, margin_l, margin_r, "#CC0000", 2)
    y += 8
    draw.text((col_price - 200, y), "Tổng cộng tiền thanh toán:", font=f_subtitle, fill="#CC0000")
    draw.text((col_total, y), _format_money(grand_total), font=f_subtitle, fill="#CC0000")
    y += 40

    # ── Amount in words ──
    draw.text((margin_l, y), f"Số tiền viết bằng chữ: {_amount_words(grand_total)}", font=f_body, fill="#000")
    y += 50

    # ── Signatures ──
    sig_y = y + 10
    draw.text((margin_l + 30, sig_y), "Người mua hàng", font=f_body_bold, fill="#000")
    draw.text((margin_l + 30, sig_y + 22), "(Buyer)", font=f_small, fill="#666")
    draw.text((W // 2 - 40, sig_y), "Người bán hàng", font=f_body_bold, fill="#000")
    draw.text((W // 2 - 40, sig_y + 22), "(Seller)", font=f_small, fill="#666")
    draw.text((W - 280, sig_y), "Thủ trưởng đơn vị", font=f_body_bold, fill="#000")
    draw.text((W - 280, sig_y + 22), "(Director)", font=f_small, fill="#666")

    # ── Red stamp (optional) ──
    if add_stamp:
        _draw_stamp(img, draw, W - 300, sig_y + 50)

    # ── Add scan noise (optional) ──
    if add_noise:
        img = _add_scan_noise(img)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), quality=95)

    return {
        "invoice_number": inv_num,
        "invoice_date": inv_date.isoformat(),
        "seller_name": seller[0],
        "seller_tax_code": seller[1],
        "buyer_name": buyer[0],
        "buyer_tax_code": buyer[1],
        "subtotal": subtotal,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "grand_total": grand_total,
        "items": [{"name": n, "qty": q, "price": p} for n, q, p in items],
    }


def _draw_stamp(img: Image.Image, draw: ImageDraw.ImageDraw, cx: int, cy: int):
    """Draw a red circular stamp."""
    r = 55
    for i in range(3):
        draw.ellipse(
            [(cx - r + i, cy - r + i), (cx + r - i, cy + r - i)],
            outline="#CC0000", width=2
        )
    f_stamp = _font(FONT_BOLD, 11)
    _draw_centered_at(draw, cx, cy - 15, "CÔNG TY", f_stamp, "#CC0000")
    _draw_centered_at(draw, cx, cy, "★", _font(FONT_BOLD, 20), "#CC0000")
    _draw_centered_at(draw, cx, cy + 18, "ĐÃ KÝ", f_stamp, "#CC0000")


def _draw_centered_at(draw, cx, cy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=fill)


def _add_scan_noise(img: Image.Image) -> Image.Image:
    """Simulate scan artifacts: slight blur, noise, rotation."""
    import numpy as np
    # Slight rotation
    angle = random.uniform(-1.5, 1.5)
    img = img.rotate(angle, fillcolor="#FFFFFF", expand=False)
    # Add gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def _amount_words(amount: float) -> str:
    """Simple Vietnamese number-to-words for common invoice totals."""
    billions = int(amount) // 1_000_000_000
    millions = (int(amount) % 1_000_000_000) // 1_000_000
    thousands = (int(amount) % 1_000_000) // 1_000
    remainder = int(amount) % 1_000

    parts = []
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    if billions > 0:
        parts.append(f"{digits[billions] if billions < 10 else str(billions)} tỷ")
    if millions > 0:
        parts.append(f"{millions} triệu")
    if thousands > 0:
        parts.append(f"{thousands} nghìn")
    if remainder > 0:
        parts.append(str(remainder))

    text = " ".join(parts) if parts else "không"
    return text.capitalize() + " đồng chẵn"


def generate_batch(
    output_dir: str | Path,
    count: int = 6,
) -> list[dict]:
    """Generate a batch of invoice images with varied styles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    styles = [
        {"add_stamp": False, "add_noise": False},  # Clean digital
        {"add_stamp": True, "add_noise": False},   # With stamp
        {"add_stamp": False, "add_noise": True},    # Scanned look
        {"add_stamp": True, "add_noise": True},     # Scanned + stamp
    ]

    for i in range(count):
        style = styles[i % len(styles)]
        fname = f"invoice_{i+1:03d}.png"
        path = output_dir / fname
        meta = generate_invoice(path, **style)
        meta["filename"] = fname
        meta["style"] = style
        results.append(meta)
        print(f"  [+] {fname} | {meta['seller_tax_code']} -> {meta['buyer_tax_code']} | {_format_money(meta['grand_total'])} VND")

    # Save metadata JSON
    import json
    meta_path = output_dir / "invoice_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Metadata saved: {meta_path}")

    return results


if __name__ == "__main__":
    out = Path(r"e:\TaxInspector\Frontend\assets\test_invoices")
    print("Generating VAT invoice images...")
    generate_batch(out, count=8)
    print("Done!")
