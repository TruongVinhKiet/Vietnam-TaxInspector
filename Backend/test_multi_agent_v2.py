import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from ml_engine.tax_agent_orchestrator import get_orchestrator

def main():
    db = SessionLocal()
    orchestrator = get_orchestrator()
    session_id = "test-session-001"
    
    queries = [
        "Kiểm tra rủi ro hoàn thuế của công ty mã số thuế 7910000338",
        "Có bao nhiêu công ty có rủi ro nợ thuế cao?",
        "Xin chào, tôi cần tra cứu thông tin",
        "Phân tích mạng lưới sở hữu chéo của doanh nghiệp 7910000338"
    ]
    
    print("🚀 Bắt đầu test Multi-Agent Orchestrator...")
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query}")
        t0 = time.time()
        
        try:
            # We will use streaming to see events and the final result
            events = list(orchestrator.process_streaming(
                db=db,
                session_id=session_id,
                message=query,
                model_mode="full"
            ))
            
            latency = time.time() - t0
            print(f"✅ Hoàn thành trong {latency:.2f}s")
            
            for ev in events:
                if ev["event"] == "done":
                    data = ev["data"]
                    print(f"  - Intent: {data.get('intent')} ({data.get('intent_confidence', data.get('confidence'))})")
                    print(f"  - Tools used: {data['tools_used']}")
                    print(f"  - Answer length: {len(data['answer'])}")
                    print(f"  - Compliance: Abstained={data['abstained']}")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()

if __name__ == "__main__":
    main()
