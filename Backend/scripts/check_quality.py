import json, re
from collections import Counter

data = [json.loads(l) for l in open('e:/TaxInspector/Backend/data/agent_ultimate_dataset.jsonl','r',encoding='utf-8')]
legal_deep = [d for d in data if len(d['messages']) == 5]
simple = [d for d in data if len(d['messages']) == 3]

print("=== DATASET QUALITY REPORT ===")
print(f"Total records: {len(data):,}")
print(f"Simple tool routing: {len(simple):,}")
print(f"Legal deep answer (multi-turn): {len(legal_deep):,}")
print()

unique_q = set(d['messages'][1]['content'] for d in legal_deep)
print(f"Unique legal questions: {len(unique_q)}")

unique_topics = set()
for d in legal_deep:
    tc = d['messages'][2]['content']
    m = re.search(r'"query":\s*"([^"]+)"', tc)
    if m:
        unique_topics.add(m.group(1))
print(f"Unique legal topics (tool_query): {len(unique_topics)}")

lens = [len(d['messages'][4]['content']) for d in legal_deep]
print(f"Avg answer length: {sum(lens)/len(lens):.0f} chars")
print(f"Min answer length: {min(lens)} chars")
print(f"Max answer length: {max(lens)} chars")
print()

tools = []
for d in simple:
    tc = d['messages'][2]['content']
    m = re.search(r'"name":\s*"([^"]+)"', tc)
    if m:
        tools.append(m.group(1))
tool_counts = Counter(tools)
print("=== TOOL DISTRIBUTION ===")
for tool, cnt in tool_counts.most_common():
    print(f"  {tool}: {cnt}")
print()

print("=== SAMPLE LEGAL DEEP ===")
sample = legal_deep[0]
print(f"Q: {sample['messages'][1]['content'][:120]}")
print(f"A: {sample['messages'][4]['content'][:300]}...")
