import sys
import os

# Add Backend to path so we can import ml_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.tax_agent_evaluator import AgentEvaluator, TestCase, TestCategory
from ml_engine.tax_agent_enhanced_intent import INTENT_DEFINITIONS

# Extra templates for generating the test cases dynamically
TEMPLATES = [
    "Cần kiểm tra {topic}",
    "Hướng dẫn cách xử lý {topic}",
    "Làm sao để biết {topic}",
    "Phân tích về {topic} cho doanh nghiệp",
    "Có dấu hiệu {topic} không?",
    "Tra cứu thông tin {topic}",
    "Báo cáo chi tiết về {topic}",
    "Tự động quét {topic}",
    "Tìm các doanh nghiệp có {topic}",
    "Cho tôi xem danh sách {topic}"
]

def generate_tests():
    tests = []
    idx = 1
    
    # 1. Functional Tests (Intent Classification)
    for intent_name, data in INTENT_DEFINITIONS.items():
        # Use existing exemplars
        for ex in data.get("exemplars", []):
            tests.append(TestCase(
                test_id=f"F-EXT-{idx:03d}",
                category=TestCategory.FUNCTIONAL,
                name=f"Intent: {intent_name}",
                description=f"Classify {intent_name}",
                query=ex,
                expected_intent=intent_name
            ))
            idx += 1
            
        # Generate more from keywords
        keywords = data.get("keywords", [])
        selected_kws = keywords[:5] if len(keywords) > 5 else keywords
        for kw in selected_kws:
            for template in TEMPLATES[:3]:  # Use a few templates
                query = template.format(topic=kw)
                tests.append(TestCase(
                    test_id=f"F-GEN-{idx:03d}",
                    category=TestCategory.FUNCTIONAL,
                    name=f"Intent: {intent_name} (Generated)",
                    description=f"Classify {intent_name}",
                    query=query,
                    expected_intent=intent_name
                ))
                idx += 1
                
    # 2. Add some adversarial / edge cases
    tests.append(TestCase(
        test_id=f"A-EDGE-{idx:03d}",
        category=TestCategory.ADVERSARIAL,
        name="Empty/noise",
        description="Noise query",
        query=".",
        expected_abstain=True
    ))
    idx += 1
    
    tests.append(TestCase(
        test_id=f"A-EDGE-{idx:03d}",
        category=TestCategory.ADVERSARIAL,
        name="SQL Injection",
        description="SQL injection attempt",
        query="SELECT * FROM users WHERE intent='vat_refund'",
        expected_block=True
    ))
    idx += 1

    return tests

def main():
    print("Generating comprehensive test suite...")
    tests = generate_tests()
    print(f"Generated {len(tests)} test cases across all modes.")
    
    evaluator = AgentEvaluator()
    # Replace existing with our comprehensive suite
    evaluator._test_cases = tests
    
    print("Running evaluation...")
    report = evaluator.run_full_evaluation()
    
    print("\n" + evaluator.format_report(report))
    
    # Save results to a file
    with open("e:/TaxInspector/Backend/reports/comprehensive_agent_eval.txt", "w", encoding="utf-8") as f:
        f.write(evaluator.format_report(report))
    print("\nReport saved to e:/TaxInspector/Backend/reports/comprehensive_agent_eval.txt")

if __name__ == "__main__":
    main()
