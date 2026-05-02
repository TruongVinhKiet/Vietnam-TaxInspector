from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


ROOT = Path(__file__).resolve().parents[2] / "data" / "mock_multimodal"
INDUSTRIES = [
    "Thuong mai",
    "Dich vu tai chinh",
    "Xay dung",
    "Logistics",
    "Cong nghe thong tin",
]


def _tax_code(i: int) -> str:
    return f"03{i:08d}"[-10:]


def generate_risk_csv(path: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "tax_code",
        "company_name",
        "industry",
        "year",
        "revenue",
        "cost_of_goods",
        "operating_expenses",
        "total_expenses",
        "net_profit",
        "vat_input",
        "vat_output",
        "industry_avg_profit_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(rows):
            revenue = rng.randint(2_000_000_000, 80_000_000_000)
            suspicious = i % 7 == 0
            expense_ratio = rng.uniform(0.55, 0.82) if not suspicious else rng.uniform(0.92, 1.12)
            total_expenses = int(revenue * expense_ratio)
            vat_output = int(revenue * 0.1)
            vat_input = int(total_expenses * (0.08 if not suspicious else 0.14))
            writer.writerow({
                "tax_code": _tax_code(i + 1),
                "company_name": f"CTY TNHH Mock Risk {i + 1}",
                "industry": rng.choice(INDUSTRIES),
                "year": 2025,
                "revenue": revenue,
                "cost_of_goods": int(total_expenses * 0.62),
                "operating_expenses": int(total_expenses * 0.38),
                "total_expenses": total_expenses,
                "net_profit": revenue - total_expenses,
                "vat_input": vat_input,
                "vat_output": vat_output,
                "industry_avg_profit_margin": round(rng.uniform(0.06, 0.14), 3),
            })


def generate_vat_csv(path: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed + 31)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "invoice_number",
        "seller_tax_code",
        "buyer_tax_code",
        "seller_name",
        "buyer_name",
        "seller_industry",
        "buyer_industry",
        "amount",
        "vat_rate",
        "date",
        "goods_category",
        "payment_status",
        "is_adjustment",
        "quantity",
        "unit_price",
        "item_description",
    ]
    base_day = date(2025, 1, 1)
    companies = [_tax_code(i + 500) for i in range(max(6, rows // 3))]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(rows):
            if i < 6:
                seller = companies[i % 3]
                buyer = companies[(i + 1) % 3]
                amount = 19_500_000 + rng.randint(-400_000, 400_000)
                goods = "Dich vu tu van"
            else:
                seller, buyer = rng.sample(companies, 2)
                amount = rng.randint(8_000_000, 2_800_000_000)
                goods = rng.choice(["Linh kien", "Van tai", "Dich vu", "Vat lieu"])
            vat_rate = 10 if i % 11 else 0
            invoice_day = base_day + timedelta(days=rng.randint(0, 90))
            writer.writerow({
                "invoice_number": f"VATMOCK-{seed}-{i + 1:05d}",
                "seller_tax_code": seller,
                "buyer_tax_code": buyer,
                "seller_name": f"CTY Mock Seller {seller[-4:]}",
                "buyer_name": f"CTY Mock Buyer {buyer[-4:]}",
                "seller_industry": rng.choice(INDUSTRIES),
                "buyer_industry": rng.choice(INDUSTRIES),
                "amount": amount,
                "vat_rate": vat_rate,
                "date": invoice_day.isoformat(),
                "goods_category": goods,
                "payment_status": rng.choice(["paid", "unknown", "overdue"]),
                "is_adjustment": "true" if i % 17 == 0 else "false",
                "quantity": rng.randint(1, 20),
                "unit_price": amount,
                "item_description": goods,
            })


def generate_invoice_images(directory: Path, count: int, seed: int) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to generate invoice images.")
    rng = random.Random(seed + 73)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        small = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    for i in range(count):
        seller = _tax_code(i + 800)
        buyer = _tax_code(i + 900)
        amount = rng.randint(12_000_000, 780_000_000)
        vat = int(amount * 0.1)
        total = amount + vat
        inv_no = f"IMGMOCK-{seed}-{i + 1:04d}"
        img = Image.new("RGB", (1200, 760), "white")
        draw = ImageDraw.Draw(img)
        y = 42
        lines = [
            "HOA DON GIA TRI GIA TANG",
            f"So hoa don: {inv_no}",
            f"Ngay hoa don: {date.today().isoformat()}",
            f"Nguoi ban: CTY Mock OCR {i + 1}",
            f"MST nguoi ban: {seller}",
            f"Nguoi mua: CTY Buyer OCR {i + 1}",
            f"MST nguoi mua: {buyer}",
            "Hang hoa: Dich vu kiem thu he thong",
            f"Tien hang: {amount}",
            "Thue suat VAT: 10%",
            f"Tien VAT: {vat}",
            f"Tong cong thanh toan: {total}",
        ]
        for idx, line in enumerate(lines):
            draw.text((70, y), line, fill="black", font=font if idx == 0 else small)
            y += 55
        draw.rectangle((55, 25, 1145, 720), outline="black", width=3)
        draw.line((55, 120, 1145, 120), fill="black", width=2)
        img.save(directory / f"{inv_no}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock multimodal tax data.")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()

    generate_risk_csv(args.out / "risk_scoring_mock.csv", args.rows, args.seed)
    generate_vat_csv(args.out / "vat_graph_mock.csv", args.rows, args.seed)
    generate_invoice_images(args.out / "invoice_images", args.images, args.seed)
    print(f"Generated multimodal mock data under {args.out}")


if __name__ == "__main__":
    main()
