import os
import json
from pathlib import Path
from ml_engine.document_ocr_engine import DocumentOCREngine

import logging

def test_ocr():
    logging.basicConfig(level=logging.INFO)
    engine = DocumentOCREngine()
    
    pdf_path = Path("test_invoice_complex.pdf")
    if not pdf_path.exists():
        print(f"File {pdf_path} not found!")
        return
        
    print(f"Reading {pdf_path}...")
    content = pdf_path.read_bytes()
    
    print("Processing with DocumentOCREngine...")
    result = engine.process_bytes(content, filename=pdf_path.name)
    
    print("\n--- EXTRACTION RESULTS ---")
    print(f"Extraction Method: {result.table_extraction_method}")
    print(f"Processing Time: {result.total_processing_ms}ms")
    
    print("\nEntities (Invoice Fields):")
    if result.invoice_fields:
        for k, v in result.invoice_fields.__dict__.items():
            print(f"  - {k}: {v}")
        
    print(f"\nTables ({len(result.tables)}):")
    for i, table in enumerate(result.tables):
        print(f"\nTable {i+1} (Method: {table.extraction_method}):")
        if table.headers:
            print(" | ".join(table.headers))
            print("-" * 50)
        for row in table.rows:
            print(" | ".join([str(c) if c is not None else "" for c in row]))

if __name__ == "__main__":
    test_ocr()
