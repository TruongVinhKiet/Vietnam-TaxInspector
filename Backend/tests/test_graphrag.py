import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
from ml_engine.tax_agent_tools import _tool_knowledge_search
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def get_db():
    engine = create_engine('postgresql://postgres:Kiet2004@localhost:5432/TaxInspector')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def run_test():
    db = get_db()
    try:
        print("--- Thử nghiệm truy vấn 1: Vi phạm trốn thuế ---")
        res1 = _tool_knowledge_search(db, query="Hành vi trốn thuế bị xử phạt như thế nào? Có quy định nào mới không?", session_id="test-session")
        print("Kết quả (1):", res1.get('status'))
        
        if 'graph_context' in res1:
            kg = res1['graph_context']
            print(f"  GraphRAG Info: {kg.get('total_entities')} nodes, {kg.get('total_relations')} relations.")
            print(f"  Traversal Path: {kg.get('traversal_path')}")
            
        print("\n--- Thử nghiệm truy vấn 2: Khấu trừ thuế GTGT ---")
        res2 = _tool_knowledge_search(db, query="Điều kiện khấu trừ thuế GTGT đầu vào là gì? Hóa đơn từ 20 triệu thì sao?", session_id="test-session")
        print("Kết quả (2):", res2.get('status'))
        if 'graph_context' in res2:
            kg = res2['graph_context']
            print(f"  GraphRAG Info: {kg.get('total_entities')} nodes, {kg.get('total_relations')} relations.")
            print(f"  Traversal Path: {kg.get('traversal_path')}")
            print(f"  Subgraph size: {len(kg.get('subgraph', {}).get('nodes', []))} nodes, {len(kg.get('subgraph', {}).get('edges', []))} edges")

        print("\nTest hoàn tất.")
    finally:
        db.close()

if __name__ == "__main__":
    run_test()

