"""
test_table_transformer.py – Unit tests for TableTransformerPipeline
====================================================================
Tests cover:
    1. Graceful degradation when transformers/timm not installed
    2. Data structure integrity (TableStructure, DocumentResult)
    3. Pipeline integration with mock models
    4. Cell text mapping logic (_build_grid)
    5. Fallback chain in DocumentOCREngine
"""

import sys
import types
import pytest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.document_ocr_engine import (
    TableStructure,
    DocumentResult,
    OCRResult,
    TableDetector,
    DocumentOCREngine,
)


# ════════════════════════════════════════════════════════════════
#  1. Data Structure Tests
# ════════════════════════════════════════════════════════════════

class TestDataStructures:
    """Verify new fields on TableStructure and DocumentResult."""

    def test_table_structure_has_extraction_method(self):
        ts = TableStructure(
            headers=["A", "B"],
            rows=[["1", "2"]],
            confidence=0.9,
            extraction_method="table_transformer",
        )
        assert ts.extraction_method == "table_transformer"
        assert ts.bbox == []

    def test_table_structure_default_method_is_heuristic(self):
        ts = TableStructure()
        assert ts.extraction_method == "heuristic"

    def test_document_result_has_table_extraction_method(self):
        dr = DocumentResult()
        assert dr.table_extraction_method == "none"

    def test_table_structure_serializable(self):
        ts = TableStructure(
            headers=["Col1", "Col2"],
            rows=[["a", "b"], ["c", "d"]],
            confidence=0.85,
            extraction_method="pdfplumber",
        )
        d = asdict(ts)
        assert d["extraction_method"] == "pdfplumber"
        assert len(d["rows"]) == 2

    def test_document_result_serializable(self):
        dr = DocumentResult(
            table_extraction_method="table_transformer",
            tables=[
                TableStructure(
                    headers=["X"], rows=[["1"]],
                    extraction_method="table_transformer",
                )
            ],
        )
        d = asdict(dr)
        assert d["table_extraction_method"] == "table_transformer"
        assert len(d["tables"]) == 1
        assert d["tables"][0]["extraction_method"] == "table_transformer"


# ════════════════════════════════════════════════════════════════
#  2. Graceful Degradation Tests
# ════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Pipeline must not crash when transformers/timm missing."""

    def test_pipeline_available_false_when_no_transformers(self):
        """Simulate missing transformers package."""
        from ml_engine.document_ocr_engine import TableTransformerPipeline

        pipeline = TableTransformerPipeline()
        # Force re-check
        pipeline._available = None

        with patch.dict(sys.modules, {"transformers": None}):
            # Reset cached availability
            pipeline._available = None
            try:
                import transformers
                # If transformers is actually installed, skip this test
                pytest.skip("transformers is installed; cannot test missing import")
            except ImportError:
                result = pipeline.available
                assert result is False

    def test_detect_and_recognize_returns_empty_when_unavailable(self):
        from ml_engine.document_ocr_engine import TableTransformerPipeline

        pipeline = TableTransformerPipeline()
        pipeline._available = False

        result = pipeline.detect_and_recognize([], [])
        assert result == []

    def test_detect_and_recognize_returns_empty_when_model_not_loaded(self):
        from ml_engine.document_ocr_engine import TableTransformerPipeline

        pipeline = TableTransformerPipeline()
        pipeline._available = True
        pipeline._det_model = None  # Model failed to load

        # _ensure_loaded would try to load, but model stays None
        with patch.object(pipeline, "_ensure_loaded"):
            result = pipeline.detect_and_recognize(
                [MagicMock()], [OCRResult(page_index=0, raw_text="", confidence=0.0)]
            )
            assert result == []


# ════════════════════════════════════════════════════════════════
#  3. Cell Text Mapping Tests (_build_grid logic)
# ════════════════════════════════════════════════════════════════

class TestBuildGrid:
    """Test the cell-text mapping logic in isolation."""

    def _get_pipeline(self):
        from ml_engine.document_ocr_engine import TableTransformerPipeline
        pipeline = TableTransformerPipeline()
        pipeline._available = True
        return pipeline

    def test_build_grid_with_valid_structure(self):
        pipeline = self._get_pipeline()

        structure = {
            "rows": [
                [0, 0, 400, 30],    # Header row
                [0, 30, 400, 60],   # Data row 1
                [0, 60, 400, 90],   # Data row 2
            ],
            "columns": [
                [0, 0, 200, 90],    # Col 1
                [200, 0, 400, 90],  # Col 2
            ],
            "headers": [
                [0, 0, 400, 30],    # Header bbox
            ],
        }

        ocr_boxes = [
            {"text": "Name", "bbox": [[50, 10], [150, 10], [150, 25], [50, 25]]},
            {"text": "Value", "bbox": [[250, 10], [350, 10], [350, 25], [250, 25]]},
            {"text": "Item A", "bbox": [[50, 40], [150, 40], [150, 55], [50, 55]]},
            {"text": "100", "bbox": [[250, 40], [350, 40], [350, 55], [250, 55]]},
            {"text": "Item B", "bbox": [[50, 70], [150, 70], [150, 85], [50, 85]]},
            {"text": "200", "bbox": [[250, 70], [350, 70], [350, 85], [250, 85]]},
        ]

        result = pipeline._build_grid(
            structure, ocr_boxes,
            table_offset=(0, 0),
            page_idx=0,
            detection_score=0.95,
        )

        assert result is not None
        assert result.headers == ["Name", "Value"]
        assert len(result.rows) == 2
        assert result.rows[0] == ["Item A", "100"]
        assert result.rows[1] == ["Item B", "200"]
        assert result.extraction_method == "table_transformer"
        assert result.confidence == 0.95

    def test_build_grid_returns_none_when_no_rows(self):
        pipeline = self._get_pipeline()
        result = pipeline._build_grid(
            {"rows": [], "columns": [[0, 0, 100, 100]], "headers": []},
            [], table_offset=(0, 0), page_idx=0, detection_score=0.8,
        )
        assert result is None

    def test_build_grid_returns_none_when_no_columns(self):
        pipeline = self._get_pipeline()
        result = pipeline._build_grid(
            {"rows": [[0, 0, 100, 30]], "columns": [], "headers": []},
            [], table_offset=(0, 0), page_idx=0, detection_score=0.8,
        )
        assert result is None

    def test_build_grid_uses_first_row_as_header_when_no_explicit_header(self):
        pipeline = self._get_pipeline()

        structure = {
            "rows": [
                [0, 0, 200, 30],
                [0, 30, 200, 60],
            ],
            "columns": [
                [0, 0, 100, 60],
                [100, 0, 200, 60],
            ],
            "headers": [],  # No explicit headers
        }

        ocr_boxes = [
            {"text": "H1", "bbox": [[50, 15], [90, 15], [90, 25], [50, 25]]},
            {"text": "H2", "bbox": [[150, 15], [190, 15], [190, 25], [150, 25]]},
            {"text": "D1", "bbox": [[50, 45], [90, 45], [90, 55], [50, 55]]},
            {"text": "D2", "bbox": [[150, 45], [190, 45], [190, 55], [150, 55]]},
        ]

        result = pipeline._build_grid(
            structure, ocr_boxes,
            table_offset=(0, 0), page_idx=0, detection_score=0.88,
        )

        assert result is not None
        assert result.headers == ["H1", "H2"]
        assert result.rows == [["D1", "D2"]]

    def test_build_grid_handles_table_offset(self):
        """When table is not at (0,0), OCR coords must be offset-adjusted."""
        pipeline = self._get_pipeline()

        structure = {
            "rows": [
                [0, 0, 200, 30],
                [0, 30, 200, 60],
            ],
            "columns": [
                [0, 0, 100, 60],
                [100, 0, 200, 60],
            ],
            "headers": [[0, 0, 200, 30]],
        }

        # OCR boxes are in page coordinates (offset by 100, 200)
        ocr_boxes = [
            {"text": "Col1", "bbox": [[150, 215], [190, 215], [190, 225], [150, 225]]},
            {"text": "Col2", "bbox": [[250, 215], [290, 215], [290, 225], [250, 225]]},
            {"text": "Val1", "bbox": [[150, 245], [190, 245], [190, 255], [150, 255]]},
            {"text": "Val2", "bbox": [[250, 245], [290, 245], [290, 255], [250, 255]]},
        ]

        result = pipeline._build_grid(
            structure, ocr_boxes,
            table_offset=(100, 200),  # Table starts at (100, 200) on page
            page_idx=0, detection_score=0.9,
        )

        assert result is not None
        assert result.headers == ["Col1", "Col2"]
        assert result.rows == [["Val1", "Val2"]]


# ════════════════════════════════════════════════════════════════
#  4. Fallback Chain Tests
# ════════════════════════════════════════════════════════════════

class TestFallbackChain:
    """DocumentOCREngine uses the correct fallback chain."""

    def test_engine_has_table_transformer(self):
        engine = DocumentOCREngine()
        assert hasattr(engine, "_table_transformer")

    def test_table_detector_still_exists(self):
        """Legacy TableDetector is preserved as fallback."""
        engine = DocumentOCREngine()
        assert isinstance(engine._table_detector, TableDetector)

    def test_document_result_tracks_extraction_method(self):
        dr = DocumentResult(table_extraction_method="table_transformer")
        assert dr.table_extraction_method == "table_transformer"


# ════════════════════════════════════════════════════════════════
#  5. Heuristic TableDetector (existing, verify unchanged)
# ════════════════════════════════════════════════════════════════

class TestHeuristicTableDetector:
    """Ensure the existing heuristic detector still works."""

    def test_empty_boxes_returns_empty(self):
        td = TableDetector()
        assert td.extract_tables_from_ocr([]) == []

    def test_single_line_returns_empty(self):
        td = TableDetector()
        boxes = [
            {"text": "Hello", "bbox": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        ]
        assert td.extract_tables_from_ocr(boxes) == []

    def test_multi_column_lines_returns_table(self):
        td = TableDetector()
        boxes = [
            # Row 1 (y=10)
            {"text": "H1", "bbox": [[0, 10]]},
            {"text": "H2", "bbox": [[100, 10]]},
            {"text": "H3", "bbox": [[200, 10]]},
            # Row 2 (y=40)
            {"text": "A", "bbox": [[0, 40]]},
            {"text": "B", "bbox": [[100, 40]]},
            {"text": "C", "bbox": [[200, 40]]},
        ]
        tables = td.extract_tables_from_ocr(boxes)
        assert len(tables) >= 1
        assert tables[0].headers == ["H1", "H2", "H3"]
        assert tables[0].extraction_method == "heuristic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
