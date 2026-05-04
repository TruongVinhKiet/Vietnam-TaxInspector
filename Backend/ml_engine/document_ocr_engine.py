"""
document_ocr_engine.py – OCR Engine cho Hóa Đơn / PDF Thuế
=============================================================
Pipeline OCR chuyên biệt cho tài liệu thuế Việt Nam:
    hóa đơn điện tử, hóa đơn giấy scan, tờ khai thuế PDF.

Capabilities:
    1. PDF/Image extraction pipeline (multi-page support)
    2. Vietnamese-optimized OCR (PaddleOCR primary, Tesseract fallback)
    3. Invoice field extraction: mã số thuế, số tiền, ngày, hàng hóa
    4. Table structure detection cho báo cáo tài chính

Architecture:
    - PaddleOCR → text detection + recognition (Vietnamese)
    - Regex + heuristic → field extraction (MST, amount, date)
    - Tabular parser → bảng kê, phụ lục hóa đơn

Design:
    - CPU-first: PaddleOCR chạy tốt trên CPU
    - Lazy loading: chỉ load model khi cần
    - Thread-safe OCR processing
    - Graceful degradation khi thiếu dependency
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class OCRResult:
    """Kết quả OCR cho một page/image."""
    page_index: int
    raw_text: str
    confidence: float
    boxes: list[dict[str, Any]] = field(default_factory=list)
    language: str = "vi"
    processing_time_ms: float = 0.0


@dataclass
class InvoiceFields:
    """Các trường thông tin trích xuất từ hóa đơn."""
    invoice_number: str = ""
    invoice_date: str = ""
    seller_tax_code: str = ""
    seller_name: str = ""
    buyer_tax_code: str = ""
    buyer_name: str = ""
    total_amount: float = 0.0
    vat_amount: float = 0.0
    vat_rate: float = 0.0
    grand_total: float = 0.0
    currency: str = "VND"
    line_items: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    extraction_method: str = "regex"


@dataclass
class TableStructure:
    """Cấu trúc bảng trích xuất từ tài liệu."""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    page_index: int = 0
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=list)
    extraction_method: str = "heuristic"


@dataclass
class DocumentResult:
    """Kết quả xử lý toàn bộ tài liệu."""
    file_path: str = ""
    file_type: str = ""
    num_pages: int = 0
    ocr_results: list[OCRResult] = field(default_factory=list)
    invoice_fields: InvoiceFields | None = None
    tables: list[TableStructure] = field(default_factory=list)
    full_text: str = ""
    total_processing_ms: float = 0.0
    status: str = "success"
    errors: list[str] = field(default_factory=list)
    table_extraction_method: str = "none"


# ════════════════════════════════════════════════════════════════
#  1. OCR Backend (PaddleOCR + Tesseract fallback)
# ════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """
    OpenCV-based image preprocessing for OCR optimization.

    Techniques:
        - Red stamp / seal removal (filter red channel)
        - Adaptive binarization (Otsu + adaptive threshold)
        - Deskew (correct slight rotation)
        - Noise reduction (morphological ops)
    """

    @staticmethod
    def preprocess(image, *, remove_stamps: bool = True, binarize: bool = False) -> Any:
        """Full preprocessing pipeline. Returns numpy array."""
        import numpy as np
        try:
            import cv2
        except ImportError:
            return image if isinstance(image, np.ndarray) else np.array(image)

        img = image if isinstance(image, np.ndarray) else np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1. Remove red stamps/seals
        if remove_stamps:
            img = ImagePreprocessor._remove_red_channel(img)

        # 2. Deskew slight rotations
        img = ImagePreprocessor._deskew(img)

        # 3. Optional binarization for noisy scans
        if binarize:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10
            )
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def _remove_red_channel(img):
        """Remove red stamps by filtering out high-red, low-blue/green pixels."""
        import numpy as np
        try:
            import cv2
        except ImportError:
            return img
        img = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Red hue ranges (0-10 and 160-180)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        # Replace red regions with white
        img[red_mask > 0] = [255, 255, 255]
        return img

    @staticmethod
    def _deskew(img, max_angle: float = 5.0):
        """Correct slight rotation using Hough line detection."""
        import numpy as np
        try:
            import cv2
        except ImportError:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is None or len(lines) < 3:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < max_angle:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


class OCRBackend:
    """
    Vietnamese-optimized OCR backend.

    Priority: PaddleOCR → EasyOCR → Tesseract → regex_only
    EasyOCR supports Vietnamese out of the box and runs on CPU.
    """

    def __init__(self):
        self._ocr_engine = None
        self._backend: str = "none"
        self._lock = threading.Lock()
        self._loaded = False
        self._preprocessor = ImagePreprocessor()

    def load(self) -> str:
        """Lazy load OCR engine. Returns backend name."""
        if self._loaded:
            return self._backend

        with self._lock:
            if self._loaded:
                return self._backend

            # Try PaddleOCR first
            backend = self._try_paddle()
            if not backend:
                backend = self._try_easyocr()
            if not backend:
                backend = self._try_tesseract()
            if not backend:
                self._backend = "regex_only"
                logger.warning(
                    "[OCR] Không có OCR engine. Chỉ hỗ trợ text-based PDF."
                )
            self._loaded = True

        return self._backend

    def _try_paddle(self) -> str | None:
        """Thử load PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="vi",
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.3,
                rec_batch_num=6,
            )
            self._backend = "paddleocr"
            logger.info("[OCR] PaddleOCR loaded (Vietnamese mode)")
            return "paddleocr"
        except ImportError:
            logger.debug("[OCR] PaddleOCR not installed")
            return None

    def _try_easyocr(self) -> str | None:
        """Thử load EasyOCR (hỗ trợ Vietnamese)."""
        try:
            import easyocr  # type: ignore
            self._ocr_engine = easyocr.Reader(
                ["vi", "en"],
                gpu=False,
                verbose=False,
            )
            self._backend = "easyocr"
            logger.info("[OCR] EasyOCR loaded (vi+en, CPU mode)")
            return "easyocr"
        except ImportError:
            logger.debug("[OCR] EasyOCR not installed")
            return None
        except Exception as exc:
            logger.warning("[OCR] EasyOCR init failed: %s", exc)
            return None

    def _try_tesseract(self) -> str | None:
        """Thử load Tesseract."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._backend = "tesseract"
            logger.info("[OCR] Tesseract loaded")
            return "tesseract"
        except Exception:
            logger.debug("[OCR] Tesseract not available")
            return None

    def recognize(self, image) -> OCRResult:
        """
        Nhận dạng text từ image (numpy array hoặc PIL Image).
        Applies preprocessing before OCR.

        Returns:
            OCRResult với raw_text và bounding boxes.
        """
        import numpy as np

        t0 = time.perf_counter()

        # Preprocess image (red stamp removal, deskew)
        processed = self._preprocessor.preprocess(image, remove_stamps=True)

        if self._backend == "paddleocr" and self._ocr_engine is not None:
            return self._recognize_paddle(processed, t0)
        elif self._backend == "easyocr" and self._ocr_engine is not None:
            return self._recognize_easyocr(processed, t0)
        elif self._backend == "tesseract":
            return self._recognize_tesseract(processed, t0)
        else:
            return OCRResult(
                page_index=0, raw_text="", confidence=0.0,
                processing_time_ms=0.0,
            )

    def _recognize_paddle(self, image, t0: float) -> OCRResult:
        """OCR bằng PaddleOCR."""
        import numpy as np

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        result = self._ocr_engine.ocr(image, cls=True)
        lines = []
        boxes = []
        total_conf = 0.0

        if result and result[0]:
            for line_info in result[0]:
                box_coords = line_info[0]
                text = line_info[1][0]
                conf = line_info[1][1]
                lines.append(text)
                boxes.append({
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox": box_coords,
                })
                total_conf += conf

        avg_conf = total_conf / max(1, len(lines))
        elapsed = (time.perf_counter() - t0) * 1000

        return OCRResult(
            page_index=0,
            raw_text="\n".join(lines),
            confidence=round(avg_conf, 4),
            boxes=boxes,
            processing_time_ms=round(elapsed, 1),
        )

    def _recognize_easyocr(self, image, t0: float) -> OCRResult:
        """OCR bằng EasyOCR (Vietnamese + English)."""
        import numpy as np

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        results = self._ocr_engine.readtext(image)
        lines = []
        boxes = []
        total_conf = 0.0

        for (bbox_coords, text, conf) in results:
            text = text.strip()
            if not text:
                continue
            lines.append(text)
            boxes.append({
                "text": text,
                "confidence": round(float(conf), 4),
                "bbox": [list(map(float, pt)) for pt in bbox_coords],
            })
            total_conf += float(conf)

        avg_conf = total_conf / max(1, len(lines))
        elapsed = (time.perf_counter() - t0) * 1000

        return OCRResult(
            page_index=0,
            raw_text="\n".join(lines),
            confidence=round(avg_conf, 4),
            boxes=boxes,
            processing_time_ms=round(elapsed, 1),
        )

    def _recognize_tesseract(self, image, t0: float) -> OCRResult:
        """OCR bằng Tesseract."""
        import pytesseract

        text = pytesseract.image_to_string(image, lang="vie")
        data = pytesseract.image_to_data(image, lang="vie", output_type=pytesseract.Output.DICT)

        confidences = [int(c) for c in data.get("conf", []) if int(c) > 0]
        avg_conf = sum(confidences) / max(1, len(confidences)) / 100.0
        elapsed = (time.perf_counter() - t0) * 1000

        return OCRResult(
            page_index=0,
            raw_text=text.strip(),
            confidence=round(avg_conf, 4),
            processing_time_ms=round(elapsed, 1),
        )


# ════════════════════════════════════════════════════════════════
#  2. PDF / Image Extractor
# ════════════════════════════════════════════════════════════════

class DocumentExtractor:
    """
    Trích xuất pages từ PDF và ảnh.

    Hỗ trợ:
        - PDF → render từng page thành image
        - Image formats: JPG, PNG, TIFF, BMP
        - Multi-page TIFF
    """

    def extract_pages(self, file_path: str | Path) -> list:
        """
        Trích xuất danh sách images từ file.

        Returns:
            List of numpy arrays (mỗi array = 1 page).
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            return self._extract_image(file_path)
        else:
            logger.warning("[Extractor] Unsupported format: %s", suffix)
            return []

    def extract_text_pdf(self, file_path: str | Path) -> str:
        """Trích xuất text trực tiếp từ PDF (không cần OCR)."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            logger.debug("[Extractor] pdfplumber not installed")
        except Exception as exc:
            logger.warning("[Extractor] PDF text extraction failed: %s", exc)
        return ""

    def _extract_pdf(self, file_path: Path) -> list:
        """Render PDF pages thành images."""
        # Try pdf2image (poppler)
        try:
            from pdf2image import convert_from_path
            import numpy as np
            images = convert_from_path(str(file_path), dpi=200)
            return [np.array(img) for img in images]
        except ImportError:
            logger.debug("[Extractor] pdf2image not installed")
        except Exception as exc:
            logger.warning("[Extractor] PDF render failed: %s", exc)

        # Try fitz (PyMuPDF)
        try:
            import fitz
            import numpy as np
            doc = fitz.open(str(file_path))
            pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.h, pix.w, pix.n
                )
                pages.append(img)
            doc.close()
            return pages
        except ImportError:
            logger.debug("[Extractor] PyMuPDF not installed")
        except Exception as exc:
            logger.warning("[Extractor] PyMuPDF render failed: %s", exc)

        return []

    def _extract_image(self, file_path: Path) -> list:
        """Load image file."""
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(str(file_path)).convert("RGB")
            return [np.array(img)]
        except ImportError:
            logger.debug("[Extractor] PIL not installed")
        except Exception as exc:
            logger.warning("[Extractor] Image load failed: %s", exc)
        return []


# ════════════════════════════════════════════════════════════════
#  3. Invoice Field Extractor (Regex + Heuristic)
# ════════════════════════════════════════════════════════════════

class InvoiceFieldExtractor:
    """
    Trích xuất các trường thông tin từ OCR text của hóa đơn.

    Patterns:
        - MST (Mã số thuế): 10 hoặc 13 chữ số
        - Số hóa đơn: pattern AA/YYE-NNNNNNN
        - Số tiền: dạng 1.000.000 hoặc 1,000,000
        - Ngày: dd/mm/yyyy hoặc dd-mm-yyyy
    """

    # Regex patterns cho hóa đơn Việt Nam
    TAX_CODE_PATTERN = re.compile(r'\b(\d{10}(?:-\d{3})?)\b')
    INVOICE_NUM_PATTERN = re.compile(
        r'(?:So|Số|No\.?)\s*[:.]?\s*([A-Z0-9]{2,}/\d{2,}[A-Z]?[-–]\d{4,})',
        re.IGNORECASE
    )
    DATE_PATTERN = re.compile(
        r'(?:Ngay|Ngày|Date)\s*[:.]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
        re.IGNORECASE
    )
    AMOUNT_PATTERN = re.compile(
        r'(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?)\s*(?:đ|VND|dong)?',
        re.IGNORECASE
    )
    VAT_RATE_PATTERN = re.compile(
        r'(?:GTGT|VAT|Thue suat|Thuế suất)\s*[:.]?\s*(\d{1,2})\s*%',
        re.IGNORECASE
    )

    def extract(self, text: str) -> InvoiceFields:
        """
        Trích xuất các trường hóa đơn từ OCR text.

        Args:
            text: Raw OCR text

        Returns:
            InvoiceFields với các trường đã trích xuất.
        """
        fields = InvoiceFields(raw_text=text, extraction_method="regex")
        confidence_parts: list[float] = []

        # Số hóa đơn
        inv_match = self.INVOICE_NUM_PATTERN.search(text)
        if inv_match:
            fields.invoice_number = inv_match.group(1).strip()
            confidence_parts.append(0.9)

        # Ngày hóa đơn
        date_match = self.DATE_PATTERN.search(text)
        if date_match:
            fields.invoice_date = date_match.group(1).strip()
            confidence_parts.append(0.85)

        # MST — tìm tất cả, gán seller = đầu tiên, buyer = thứ hai
        tax_codes = self.TAX_CODE_PATTERN.findall(text)
        tax_codes = list(dict.fromkeys(tax_codes))  # Deduplicate
        if len(tax_codes) >= 1:
            fields.seller_tax_code = tax_codes[0]
            confidence_parts.append(0.9)
        if len(tax_codes) >= 2:
            fields.buyer_tax_code = tax_codes[1]

        # Tên đơn vị (heuristic: dòng sau "Đơn vị bán" / "Tên đơn vị")
        fields.seller_name = self._extract_entity_name(
            text, ["Don vi ban", "Đơn vị bán", "Nguoi ban", "Người bán"]
        )
        fields.buyer_name = self._extract_entity_name(
            text, ["Don vi mua", "Đơn vị mua", "Nguoi mua", "Người mua"]
        )

        # Số tiền — tìm tổng lớn nhất
        amounts = self._extract_amounts(text)
        if amounts:
            fields.grand_total = max(amounts)
            if len(amounts) >= 2:
                sorted_amounts = sorted(amounts, reverse=True)
                fields.grand_total = sorted_amounts[0]
                fields.vat_amount = sorted_amounts[1] if sorted_amounts[1] < sorted_amounts[0] * 0.15 else 0.0
                fields.total_amount = fields.grand_total - fields.vat_amount
            confidence_parts.append(0.75)

        # Thuế suất
        vat_match = self.VAT_RATE_PATTERN.search(text)
        if vat_match:
            fields.vat_rate = float(vat_match.group(1))
            confidence_parts.append(0.9)

        # Overall confidence
        fields.confidence = round(
            sum(confidence_parts) / max(1, len(confidence_parts)), 4
        ) if confidence_parts else 0.0

        return fields

    def _extract_entity_name(self, text: str, keywords: list[str]) -> str:
        """Trích xuất tên đơn vị dựa trên keyword (hỗ trợ tên nằm ở dòng tiếp theo)."""
        lines = text.splitlines()
        for kw in keywords:
            for i, line in enumerate(lines):
                if kw.lower() in line.lower():
                    # Thử lấy phần text phía sau keyword trên cùng 1 dòng
                    pattern = re.compile(rf'{re.escape(kw)}[^:]*[:.]?\s*(.+)$', re.IGNORECASE)
                    match = pattern.search(line)
                    name = ""
                    if match:
                        name = match.group(1).strip()
                    
                    # Nếu tên quá ngắn hoặc rỗng, lấy nguyên dòng tiếp theo
                    if len(name) < 5 and i + 1 < len(lines):
                        name = lines[i+1].strip()
                        
                    # Clean up
                    name = re.sub(r'\s+', ' ', name)
                    if len(name) > 5:
                        return name[:200]
        return ""

    def _extract_amounts(self, text: str) -> list[float]:
        """Trích xuất tất cả số tiền từ text."""
        amounts = []
        for match in self.AMOUNT_PATTERN.finditer(text):
            raw = match.group(1)
            # Chuẩn hóa: bỏ dấu chấm ngăn nghìn, giữ dấu phẩy thập phân
            cleaned = raw.replace(".", "").replace(",", ".")
            try:
                val = float(cleaned)
                if val > 0:
                    amounts.append(val)
            except ValueError:
                continue
        return amounts


# ════════════════════════════════════════════════════════════════
#  4. Table Detector
# ════════════════════════════════════════════════════════════════

class TableDetector:
    """
    Phát hiện và trích xuất bảng từ tài liệu.

    Sử dụng pdfplumber cho PDF text-based,
    heuristic alignment cho OCR-based tables.
    """

    def extract_tables_from_pdf(self, file_path: str | Path) -> list[TableStructure]:
        """Trích xuất bảng từ PDF bằng pdfplumber."""
        tables = []
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables() or []
                    for raw_table in page_tables:
                        if not raw_table or len(raw_table) < 2:
                            continue
                        headers = [str(c or "").strip() for c in raw_table[0]]
                        rows = [
                            [str(c or "").strip() for c in row]
                            for row in raw_table[1:]
                        ]
                        tables.append(TableStructure(
                            headers=headers, rows=rows,
                            page_index=page_idx, confidence=0.85,
                        ))
        except ImportError:
            logger.debug("[TableDetector] pdfplumber not installed")
        except Exception as exc:
            logger.warning("[TableDetector] Table extraction failed: %s", exc)
        return tables

    def extract_tables_from_ocr(
        self, ocr_boxes: list[dict[str, Any]]
    ) -> list[TableStructure]:
        """
        Heuristic table detection từ OCR bounding boxes.

        Nhóm các box theo dòng (y-coordinate), sau đó theo cột (x).
        """
        if not ocr_boxes:
            return []

        # Nhóm theo dòng (y tolerance = 15px)
        lines: list[list[dict]] = []
        sorted_boxes = sorted(
            ocr_boxes,
            key=lambda b: b.get("bbox", [[0, 0]])[0][1] if b.get("bbox") else 0
        )

        current_line: list[dict] = []
        current_y = 0.0
        y_tolerance = 15.0

        for box in sorted_boxes:
            bbox = box.get("bbox", [[0, 0]])
            y = bbox[0][1] if bbox else 0
            if current_line and abs(y - current_y) > y_tolerance:
                lines.append(current_line)
                current_line = []
            current_line.append(box)
            current_y = y

        if current_line:
            lines.append(current_line)

        # Xác định dòng có nhiều cột nhất → header candidate
        if len(lines) < 2:
            return []

        max_cols = max(len(line) for line in lines)
        if max_cols < 2:
            return []

        # Lọc dòng có đủ cột (>= 50% max)
        table_lines = [
            line for line in lines if len(line) >= max_cols * 0.5
        ]

        if len(table_lines) < 2:
            return []

        headers = [b.get("text", "") for b in sorted(
            table_lines[0],
            key=lambda b: b.get("bbox", [[0, 0]])[0][0] if b.get("bbox") else 0
        )]
        rows = []
        for line in table_lines[1:]:
            row = [b.get("text", "") for b in sorted(
                line,
                key=lambda b: b.get("bbox", [[0, 0]])[0][0] if b.get("bbox") else 0
            )]
            rows.append(row)

        return [TableStructure(
            headers=headers, rows=rows,
            page_index=0, confidence=0.6,
        )]


# ════════════════════════════════════════════════════════════════
#  4b. Table Transformer Pipeline (AI-powered)
# ════════════════════════════════════════════════════════════════

TATR_MODEL_DIR = MODEL_DIR / "table_transformer"


class TableTransformerPipeline:
    """
    AI-powered table detection & structure recognition using
    Microsoft Table Transformer (DETR-based, ~110M params each).

    Two-stage pipeline:
        Stage 1 – Detection:  Locate table bounding boxes on full page.
        Stage 2 – Structure:  Identify rows, columns, headers in cropped table.

    Models (auto-downloaded from HuggingFace on first use):
        - microsoft/table-transformer-detection       (~220 MB)
        - microsoft/table-transformer-structure-recognition (~220 MB)

    Designed for CPU inference on i7 + 12 GB RAM.
    Lazy-loaded to avoid startup penalty.
    Falls back gracefully when `transformers` / `timm` are missing.
    """

    DETECTION_MODEL = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"
    DETECTION_THRESHOLD = 0.7
    STRUCTURE_THRESHOLD = 0.6

    # Label maps from model config
    STRUCTURE_LABELS = {
        0: "table",
        1: "table column",
        2: "table row",
        3: "table column header",
        4: "table projected row header",
        5: "table spanning cell",
    }

    def __init__(self):
        self._det_model = None
        self._det_processor = None
        self._str_model = None
        self._str_processor = None
        self._lock = threading.Lock()
        self._available: bool | None = None  # None = not checked yet

    # ── public API ──────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Check whether required deps exist (without loading models)."""
        if self._available is not None:
            return self._available
        try:
            import transformers  # noqa: F401
            import timm  # noqa: F401
            import torch  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.info(
                "[TableTransformer] transformers/timm not installed; "
                "AI table extraction disabled."
            )
        return self._available

    def detect_and_recognize(
        self,
        page_images: list,
        ocr_results: list[OCRResult],
    ) -> list[TableStructure]:
        """
        End-to-end: detect tables on each page, recognize structure,
        then map OCR text into cells.

        Args:
            page_images:  list of numpy arrays (one per page).
            ocr_results:  corresponding OCR results with bounding boxes.

        Returns:
            List of TableStructure with headers/rows populated.
        """
        if not self.available:
            return []

        self._ensure_loaded()
        if self._det_model is None:
            return []

        tables: list[TableStructure] = []
        for page_idx, page_img in enumerate(page_images):
            page_ocr = (
                ocr_results[page_idx] if page_idx < len(ocr_results) else None
            )
            page_tables = self._process_page(page_img, page_idx, page_ocr)
            tables.extend(page_tables)

        return tables

    # ── lazy loading ────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._det_model is not None:
            return
        with self._lock:
            if self._det_model is not None:
                return
            try:
                self._load_models()
            except Exception as exc:
                logger.warning(
                    "[TableTransformer] Model load failed: %s", exc
                )
                self._available = False

    def _load_models(self) -> None:
        import torch  # noqa: F811
        from transformers import (  # type: ignore
            DetrImageProcessor,
            TableTransformerForObjectDetection,
        )

        cache_dir = str(TATR_MODEL_DIR)

        logger.info("[TableTransformer] Loading detection model …")
        self._det_processor = DetrImageProcessor.from_pretrained(
            self.DETECTION_MODEL, cache_dir=cache_dir
        )
        self._det_model = TableTransformerForObjectDetection.from_pretrained(
            self.DETECTION_MODEL,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        self._det_model.eval()

        logger.info("[TableTransformer] Loading structure model …")
        self._str_processor = DetrImageProcessor.from_pretrained(
            self.STRUCTURE_MODEL, cache_dir=cache_dir
        )
        self._str_model = TableTransformerForObjectDetection.from_pretrained(
            self.STRUCTURE_MODEL,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        self._str_model.eval()

        logger.info("[TableTransformer] ✓ Both models loaded (CPU).")

    # ── per-page processing ─────────────────────────────────────

    def _process_page(
        self,
        page_img,
        page_idx: int,
        page_ocr: OCRResult | None,
    ) -> list[TableStructure]:
        """Detect tables on one page, then recognize structure for each."""
        import torch
        from PIL import Image
        import numpy as np

        if not isinstance(page_img, np.ndarray):
            page_img = np.array(page_img)

        pil_img = Image.fromarray(page_img).convert("RGB")

        # Stage 1 – Detection
        detected = self._detect_tables(pil_img)
        if not detected:
            return []

        ocr_boxes = page_ocr.boxes if page_ocr else []
        results: list[TableStructure] = []

        for det in detected:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            score = det["score"]

            # Crop table region (with 10px padding)
            x1 = max(0, int(bbox[0]) - 10)
            y1 = max(0, int(bbox[1]) - 10)
            x2 = min(pil_img.width, int(bbox[2]) + 10)
            y2 = min(pil_img.height, int(bbox[3]) + 10)
            cropped = pil_img.crop((x1, y1, x2, y2))

            # Stage 2 – Structure recognition
            structure = self._recognize_structure(cropped)

            # Stage 3 – Map OCR text into cell grid
            table = self._build_grid(
                structure, ocr_boxes,
                table_offset=(x1, y1),
                page_idx=page_idx,
                detection_score=score,
            )
            if table is not None:
                results.append(table)

        return results

    def _detect_tables(self, pil_image) -> list[dict]:
        """Stage 1: Find table bounding boxes on a full page."""
        import torch

        inputs = self._det_processor(
            images=pil_image, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._det_model(**inputs)

        target_sizes = [list(reversed(pil_image.size))]  # [h, w]
        raw = self._det_processor.post_process_object_detection(
            outputs,
            threshold=self.DETECTION_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(
            raw["scores"], raw["labels"], raw["boxes"]
        ):
            detections.append({
                "bbox": [round(float(c), 1) for c in box.tolist()],
                "score": round(float(score), 4),
                "label": self._det_model.config.id2label.get(
                    label.item(), "table"
                ),
            })

        logger.debug(
            "[TableTransformer] Detected %d table(s) on page.",
            len(detections),
        )
        return detections

    def _recognize_structure(self, cropped_pil) -> dict:
        """
        Stage 2: Identify rows, columns, headers inside a cropped table.

        Returns dict with keys:
            rows:    list of [y1, y2] pairs
            columns: list of [x1, x2] pairs
            headers: list of [y1, y2] pairs (column header rows)
        """
        import torch

        inputs = self._str_processor(
            images=cropped_pil, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._str_model(**inputs)

        target_sizes = [list(reversed(cropped_pil.size))]
        raw = self._str_processor.post_process_object_detection(
            outputs,
            threshold=self.STRUCTURE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        rows = []
        columns = []
        headers = []

        for score, label, box in zip(
            raw["scores"], raw["labels"], raw["boxes"]
        ):
            coords = [round(float(c), 1) for c in box.tolist()]
            label_name = self._str_model.config.id2label.get(
                label.item(), "unknown"
            )

            if "row" in label_name and "header" not in label_name:
                rows.append(coords)  # [x1, y1, x2, y2]
            elif "column" in label_name and "header" not in label_name:
                columns.append(coords)
            elif "header" in label_name:
                headers.append(coords)

        # Sort rows top-to-bottom, columns left-to-right
        rows.sort(key=lambda b: b[1])
        columns.sort(key=lambda b: b[0])

        return {"rows": rows, "columns": columns, "headers": headers}

    def _build_grid(
        self,
        structure: dict,
        ocr_boxes: list[dict],
        table_offset: tuple[int, int],
        page_idx: int,
        detection_score: float,
    ) -> TableStructure | None:
        """
        Stage 3: Combine row/column grid with OCR text to build
        a structured table.

        For each cell (row_i × col_j intersection), find OCR boxes
        whose center falls inside the cell, concatenate their text.
        """
        rows = structure.get("rows", [])
        cols = structure.get("columns", [])
        header_boxes = structure.get("headers", [])

        if not rows or not cols:
            return None

        ox, oy = table_offset

        def _cell_text(row_bbox, col_bbox) -> str:
            """Find OCR texts whose center falls within the cell."""
            # Cell = intersection of row and column bounding boxes
            cell_x1 = col_bbox[0] + ox
            cell_y1 = row_bbox[1] + oy
            cell_x2 = col_bbox[2] + ox
            cell_y2 = row_bbox[3] + oy

            texts = []
            for box in ocr_boxes:
                bbox = box.get("bbox", [])
                if not bbox:
                    continue
                # OCR bbox can be [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # or [x1,y1,x2,y2]. Handle both.
                if isinstance(bbox[0], (list, tuple)):
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                else:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2

                if cell_x1 <= cx <= cell_x2 and cell_y1 <= cy <= cell_y2:
                    texts.append(box.get("text", ""))

            return " ".join(texts).strip()

        # Determine which rows are headers
        header_y_centers = set()
        for hb in header_boxes:
            header_y_centers.add(round((hb[1] + hb[3]) / 2, 0))

        header_row_texts: list[str] = []
        data_rows: list[list[str]] = []

        for row_bbox in rows:
            row_center_y = round((row_bbox[1] + row_bbox[3]) / 2, 0)
            is_header = any(
                abs(row_center_y - hy) < 20 for hy in header_y_centers
            )

            row_cells = [_cell_text(row_bbox, col) for col in cols]

            if is_header and not header_row_texts:
                header_row_texts = row_cells
            else:
                data_rows.append(row_cells)

        # If no explicit header detected, use first row
        if not header_row_texts and data_rows:
            header_row_texts = data_rows.pop(0)

        # Need at least 1 header + 1 data row
        if not header_row_texts or not data_rows:
            return None

        return TableStructure(
            headers=header_row_texts,
            rows=data_rows,
            page_index=page_idx,
            confidence=round(detection_score, 4),
            bbox=[],
            extraction_method="table_transformer",
        )


# ════════════════════════════════════════════════════════════════
#  5. Main Document OCR Engine
# ════════════════════════════════════════════════════════════════

class DocumentOCREngine:
    """
    Pipeline OCR end-to-end cho tài liệu thuế.

    Usage:
        engine = DocumentOCREngine()
        result = engine.process("invoice_scan.pdf")
        print(result.invoice_fields)
        print(result.tables)
    """

    def __init__(self):
        self._ocr = OCRBackend()
        self._extractor = DocumentExtractor()
        self._field_extractor = InvoiceFieldExtractor()
        self._table_detector = TableDetector()
        self._table_transformer = TableTransformerPipeline()
        self._lock = threading.Lock()

    def process(
        self,
        file_path: str | Path,
        extract_fields: bool = True,
        extract_tables: bool = True,
    ) -> DocumentResult:
        """
        Xử lý toàn bộ tài liệu.

        Args:
            file_path: Đường dẫn file PDF/image.
            extract_fields: Trích xuất invoice fields.
            extract_tables: Trích xuất tables.

        Returns:
            DocumentResult đầy đủ.
        """
        file_path = Path(file_path)
        t0 = time.perf_counter()
        result = DocumentResult(
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
        )

        if not file_path.exists():
            result.status = "error"
            result.errors.append(f"File không tồn tại: {file_path}")
            return result

        # Load OCR backend
        self._ocr.load()

        # Thử text extraction trước (nhanh hơn OCR)
        direct_text = ""
        if file_path.suffix.lower() == ".pdf":
            direct_text = self._extractor.extract_text_pdf(file_path)

        pages_for_ai: list = []  # page images for AI table detection

        if direct_text and len(direct_text.strip()) > 50:
            # PDF có text layer → không cần OCR
            result.full_text = direct_text
            result.num_pages = direct_text.count("\n\n") + 1
            result.ocr_results.append(OCRResult(
                page_index=0, raw_text=direct_text,
                confidence=0.95, processing_time_ms=0.0,
            ))
            # Still extract page images for AI table detection
            if self._table_transformer.available:
                pages_for_ai = self._extractor.extract_pages(file_path)
        else:
            # Cần OCR
            pages = self._extractor.extract_pages(file_path)
            result.num_pages = len(pages)
            pages_for_ai = pages  # Keep reference for AI table detection

            all_text_parts = []
            for idx, page_img in enumerate(pages):
                ocr_result = self._ocr.recognize(page_img)
                ocr_result.page_index = idx
                result.ocr_results.append(ocr_result)
                all_text_parts.append(ocr_result.raw_text)

            result.full_text = "\n\n".join(all_text_parts)

        # Extract invoice fields
        if extract_fields and result.full_text:
            result.invoice_fields = self._field_extractor.extract(result.full_text)

        # Extract tables — Fallback chain:
        #   1. Table Transformer AI (highest quality)
        #   2. pdfplumber (text-based PDF only)
        #   3. Heuristic OCR box alignment (lowest quality)
        if extract_tables:
            # Tier 1: AI-powered table detection + structure recognition
            if (
                self._table_transformer.available
                and pages_for_ai
            ):
                try:
                    ai_tables = self._table_transformer.detect_and_recognize(
                        pages_for_ai, result.ocr_results
                    )
                    if ai_tables:
                        result.tables = ai_tables
                        result.table_extraction_method = "table_transformer"
                        logger.info(
                            "[OCR] AI Table Transformer extracted %d table(s).",
                            len(ai_tables),
                        )
                except Exception as exc:
                    logger.warning(
                        "[OCR] Table Transformer failed, falling back: %s", exc
                    )

            # Tier 2: pdfplumber (text-based PDF)
            if not result.tables and file_path.suffix.lower() == ".pdf":
                result.tables = self._table_detector.extract_tables_from_pdf(file_path)
                if result.tables:
                    result.table_extraction_method = "pdfplumber"

            # Tier 3: Heuristic alignment from OCR boxes
            if not result.tables and result.ocr_results:
                all_boxes = []
                for ocr_r in result.ocr_results:
                    all_boxes.extend(ocr_r.boxes)
                if all_boxes:
                    result.tables = self._table_detector.extract_tables_from_ocr(all_boxes)
                    if result.tables:
                        result.table_extraction_method = "heuristic"

        result.total_processing_ms = round(
            (time.perf_counter() - t0) * 1000, 1
        )

        logger.info(
            "[OCR] Processed %s: %d pages, %d chars, %d tables, %.0fms",
            file_path.name, result.num_pages, len(result.full_text),
            len(result.tables), result.total_processing_ms,
        )

        return result

    def process_bytes(
        self,
        data: bytes,
        filename: str = "upload.pdf",
        **kwargs,
    ) -> DocumentResult:
        """
        Xử lý từ bytes (upload qua API).

        Lưu tạm file rồi gọi process().
        """
        import tempfile
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=str(MODEL_DIR.parent)
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            return self.process(tmp_path, **kwargs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ════════════════════════════════════════════════════════════════
#  Singleton
# ════════════════════════════════════════════════════════════════

_ocr_engine: DocumentOCREngine | None = None


def get_ocr_engine() -> DocumentOCREngine:
    """Singleton cho DocumentOCREngine."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = DocumentOCREngine()
    return _ocr_engine
