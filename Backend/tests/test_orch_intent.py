import sys
import asyncio
from pathlib import Path
sys.path.append('e:/TaxInspector/Backend')

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

def test_intent():
    from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator
    orch = TaxAgentOrchestrator()
    msg1 = "Công ty tôi kinh doanh dịch vụ ăn uống, nhà hàng. Xin hỏi theo Nghị định 72/2024/NĐ-CP, tôi có được áp dụng thuế GTGT 8% không? Thời hạn áp dụng đến khi nào?"
    
    db = SessionLocal()
    try:
        events = list(orch.process_streaming(db, "test", msg1, model_mode="legal"))
        for ev in events:
            if ev.get("event") == "thinking" and ev["data"].get("step") == "intent_done":
                print(f"Orchestrator Intent Event: {ev}")
            if ev.get("event") == "done":
                print(f"Final Intent: {ev['data'].get('intent')}")
    finally:
        db.close()

if __name__ == "__main__":
    test_intent()
