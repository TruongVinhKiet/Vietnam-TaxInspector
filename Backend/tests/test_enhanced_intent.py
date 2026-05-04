import sys
import logging
from pathlib import Path
sys.path.append('e:/TaxInspector/Backend')

logging.basicConfig(level=logging.INFO)

from ml_engine.tax_agent_enhanced_intent import get_intent_classifier
classifier = get_intent_classifier()
print(f"Tier: {classifier.tier}")

msg1 = "Công ty tôi kinh doanh dịch vụ ăn uống, nhà hàng. Xin hỏi theo Nghị định 72/2024/NĐ-CP, tôi có được áp dụng thuế GTGT 8% không? Thời hạn áp dụng đến khi nào?"
msg2 = "Doanh nghiệp FDI của chúng tôi đang mở rộng nhà máy ở KCN Long Hậu, Long An. Vốn đầu tư thêm 50 tỷ. Xin hỏi chúng tôi có được hưởng ưu đãi thuế TNDN cho phần thu nhập tăng thêm không?"

res1 = classifier.classify(msg1)
print(f"msg1: {res1.primary_intent} ({res1.primary_confidence}) source={res1.classification_source}")
print(res1.all_scores)

res2 = classifier.classify(msg2)
print(f"msg2: {res2.primary_intent} ({res2.primary_confidence}) source={res2.classification_source}")
print(res2.all_scores)
