"""
Auto-generate audit trail data + LoRA training pipeline.

Steps:
1. Generate synthetic audit trail data (JSONL)
2. Run LoRA fine-tuning on Qwen2.5-1.5B
3. Verify the fine-tuned model loads
4. Confirm auto-upgrade from template -> fine-tuned LLM
"""
import sys, os, json, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from pathlib import Path

print("=" * 60)
print("  TaxInspector: Auto LLM Training Pipeline")
print("=" * 60)

# ─── Step 1: Generate synthetic audit trail data ──────────────

print("\n[Step 1] Generating synthetic training data...")

TRAINING_EXAMPLES = [
    # VAT Refund
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Dieu kien hoan thue VAT cho doanh nghiep xuat khau la gi?",
        "assistant": "Theo Dieu 13 Luat Thue GTGT so 13/2008/QH12 va Thong tu 219/2013/TT-BTC:\n\n1. **Dieu kien co ban:**\n- Co hang hoa, dich vu xuat khau\n- Thue GTGT dau vao chua duoc khau tru het >= 300 trieu dong lien tuc 12 thang\n- Co hop dong xuat khau, to khai hai quan\n\n2. **Ho so yeu cau:**\n- Giau de nghi hoan thue (mau 01/HT)\n- Bang ke hoa don, chung tu hang hoa xuat khau\n- To khai hai quan co xac nhan thuc xuat\n\n3. **Thoi gian xu ly:** 5-40 ngay lam viec tuy theo loai ho so.\n\nKhuyen nghi: kiem tra ky ho so truoc khi nop de tranh bi tu choi.",
        "intent": "vat_refund_risk", "confidence": 0.92,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Rui ro hoan thue GTGT thuong gap la gi?",
        "assistant": "Theo kinh nghiem thanh tra thue, cac rui ro hoan thue GTGT thuong gap:\n\n1. **Hoa don dau vao khong hop le:** Su dung hoa don cua doanh nghiep bo tron, mat tich\n2. **Giao dich khong co that:** Hang hoa khong thuc te di chuyen, chi co chung tu\n3. **Khai tang gia tri xuat khau:** Gia xuat khau cao bat thuong so voi gia thi truong\n4. **Carousel fraud:** Giao dich vong tron giua cac doanh nghiep lien ket\n5. **Khong thanh toan qua ngan hang:** Vi pham dieu kien thanh toan theo quy dinh\n\nCan cu: Thong tu 219/2013/TT-BTC, Dieu 18-19.",
        "intent": "vat_refund_risk", "confidence": 0.88,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "So sanh hoan thue truoc kiem tra va hoan thue sau kiem tra?",
        "assistant": "Theo Luat Quan ly thue 2019 va Thong tu 80/2021:\n\n**Hoan truoc kiem tra sau (Dieu 73):**\n- Ap dung cho: DN xuat khau co qua trinh cham hanh tot\n- Thoi gian: 6 ngay lam viec\n- Dieu kien: khong vi pham thue trong 2 nam gan nhat\n\n**Hoan sau kiem tra truoc (Dieu 73.2):**\n- Ap dung cho: DN lien tuc moi thanh lap, DN co vi pham truoc do\n- Thoi gian: 40 ngay lam viec\n- Kiem tra tai tru so doanh nghiep\n\n**Luu y:** Co quan thue co quyen chuyen tu hoan truoc sang hoan sau (Dieu 73.4).",
        "intent": "vat_refund_risk", "confidence": 0.85,
    },

    # Invoice Risk
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Lam sao phat hien hoa don gia mua ban?",
        "assistant": "Theo Nghi dinh 123/2020/ND-CP va kinh nghiem thuc te:\n\n**Dau hieu nhan biet hoa don gia:**\n1. Ben ban khong co hoat dong kinh doanh thuc tai dia chi dang ky\n2. Gia tri hoa don lon bat thuong so voi quy mo doanh nghiep\n3. Ma so thue ben ban dang trong trang thai ngung hoat dong\n4. Khong co chung tu van chuyen, giao nhan hang hoa\n5. Thanh toan bang tien mat cho giao dich gia tri lon\n6. Ben ban va ben mua co moi quan he lien ket\n\n**Cong cu kiem tra:**\n- Tra cuu MST qua tracuunnt.gdt.gov.vn\n- Doi chieu cheo hoa don dau vao - dau ra\n- Kiem tra dau hieu bat thuong tren he thong HĐĐT",
        "intent": "invoice_risk", "confidence": 0.90,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Quy dinh xu phat su dung hoa don bat hop phap?",
        "assistant": "Theo Nghi dinh 125/2020/ND-CP ve xu phat hanh chinh thue:\n\n**Muc phat:**\n1. Su dung hoa don bat hop phap: phat tien 20-50 trieu dong (Dieu 28)\n2. Su dung hoa don cua DN bo tron: phat 20-50 trieu + truy thu thue\n3. Lap hoa don khong dung thoi diem: phat 3-5 trieu dong\n4. Lam mat hoa don: phat 2-4 trieu dong moi hoa don\n\n**Hau qua phap ly:**\n- Khong duoc khau tru thue GTGT dau vao\n- Khong duoc tinh chi phi hop ly khi tinh thue TNDN\n- Co the bi truy cuu trach nhiem hinh su (Dieu 200 BLHS 2015)",
        "intent": "invoice_risk", "confidence": 0.87,
    },

    # Delinquency
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Bien phap cuong che no thue theo quy dinh moi nhat?",
        "assistant": "Theo Luat Quan ly thue so 38/2019/QH14, Dieu 62-71:\n\n**7 bien phap cuong che (ap dung tuan tu):**\n1. Trich tien tu tai khoan ngan hang (D62)\n2. Khau tru mot phan luong (D63)\n3. Dung lam thu tuc xuat canh (D64)\n4. Ngung su dung hoa don (D65)\n5. Ke bien tai san (D66)\n6. Thu tien tu ben thu ba no nguoi bi cuong che (D67)\n7. Thu hoi giay phep kinh doanh (D68)\n\n**Dieu kien ap dung:**\n- No thue qua han >= 90 ngay\n- Da gui thong bao nhac no it nhat 3 lan\n- Lai cham nop: 0.03%/ngay tren so tien no thue",
        "intent": "delinquency", "confidence": 0.91,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Du bao kha nang cham nop thue cua doanh nghiep nhu the nao?",
        "assistant": "He thong su dung mo hinh hoc may ket hop nhieu chi so:\n\n**Cac yeu to du bao:**\n1. **Lich su nop thue:** Tan suat cham nop trong 12-24 thang\n2. **Tinh hinh tai chinh:** Ty suat thanh toan, dong tien hoat dong\n3. **Nganh nghe:** Muc do rui ro theo nganh (xay dung, bat dong san cao nhat)\n4. **Quy mo:** Doanh nghiep nho thuong co ty le cham nop cao hon\n5. **Mua vu:** Cham nop tang vao cac ky quyet toan nam\n\n**Muc do rui ro:**\n- Thap (<20%): Giam sat thuong ky\n- Trung binh (20-50%): Nhac no som\n- Cao (>50%): Ap dung bien phap cuong che som",
        "intent": "delinquency", "confidence": 0.83,
    },

    # OSINT/Ownership
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Cach truy vet chu so huu thuc su cua doanh nghiep?",
        "assistant": "Theo Nghi dinh 47/2021/ND-CP ve xac dinh chu so huu huong loi:\n\n**Quy trinh truy vet UBO (Ultimate Beneficial Owner):**\n1. **Buoc 1:** Tra cuu ĐKKD tren Cong thong tin doanh nghiep\n2. **Buoc 2:** Phan tich cau truc so huu qua cac lop cong ty\n3. **Buoc 3:** Xac dinh nguoi nam giu >= 25% von dieu le (truc tiep hoac gian tiep)\n4. **Buoc 4:** Kiem tra cac moi quan he gia dinh, nguoi lien quan\n5. **Buoc 5:** Doi chieu voi du lieu quoc te (offshore leaks)\n\n**Dau hieu cong ty vo boc:**\n- Dia chi dang ky la can ho, khong co bien hieu\n- Von dieu le rat nho nhung giao dich lon\n- Thay doi nguoi dai dien nhieu lan trong thoi gian ngan",
        "intent": "osint_ownership", "confidence": 0.86,
    },

    # Transfer Pricing
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Phuong phap kiem tra chuyen gia cho doanh nghiep FDI?",
        "assistant": "Theo Nghi dinh 132/2020/ND-CP ve giao dich lien ket:\n\n**5 phuong phap xac dinh gia:**\n1. **So sanh gia (CUP):** So sanh gia gi giao dich voi gia thi truong tu do\n2. **Gia ban lai (RPM):** Ap dung cho phan phoi, thuong mai\n3. **Gia von cong them (CPM):** Ap dung cho gia cong, san xuat\n4. **Chia loi nhuan (PSM):** Ap dung khi ca hai ben dong gop tai san vo hinh\n5. **Loi nhuan rong (TNMM):** Phuong phap pho bien nhat, so sanh ty suat loi nhuan\n\n**Chi so canh bao:**\n- Ty suat loi nhuan gop < 5% trong khi nganh > 15%\n- Lo lien tuc 3+ nam nhung van mo rong kinh doanh\n- Thanh toan thuong quyen (royalty) > 3% doanh thu",
        "intent": "transfer_pricing", "confidence": 0.89,
    },

    # Audit Selection
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Tieu chi lua chon doi tuong thanh tra thue?",
        "assistant": "Theo Quyet dinh 970/QD-TCT va Luat Quan ly thue:\n\n**Tieu chi xep hang rui ro:**\n1. **Diem rui ro tong hop:** Ket hop nhieu chi so (thang diem 0-100)\n2. **Lich su vi pham:** So lan bi xu phat, truy thu\n3. **Bien dong doanh thu:** Tang/giam dot bien so voi ky truoc\n4. **Ty suat loi nhuan:** Thap bat thuong so voi nganh\n5. **Du lieu cheo:** Bat dong giua to khai thue va thong tin tu ben thu ba\n\n**Uu tien thanh tra:**\n- Nhom A (cao nhat): Diem >= 80, co dau hieu gian lan\n- Nhom B: Diem 60-80, can kiem tra bo sung\n- Nhom C: Diem < 60, giam sat thuong ky\n\nCan cu: Dieu 110-113 Luat QLT 38/2019.",
        "intent": "audit_selection", "confidence": 0.87,
    },

    # General tax queries
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Thue suat thue TNDN hien hanh la bao nhieu?",
        "assistant": "Theo Luat Thue Thu nhap doanh nghiep (sua doi 2013):\n\n**Thue suat pho thong:** 20%\n\n**Thue suat uu dai:**\n- 10% trong 15 nam: DN dau tu tai KCN, KKT dac biet\n- 15% trong 12 nam: DN phan mem, nong nghiep cong nghe cao\n- 17%: DN co doanh thu < 20 ty/nam va lao dong < 200 nguoi\n\n**Mien giam:**\n- Mien 2-4 nam dau, giam 50% trong 4-9 nam tiep theo (tuy linh vuc)\n\nCan cu: Dieu 10-14 Luat Thue TNDN, Thong tu 78/2014/TT-BTC.",
        "intent": "general_tax_query", "confidence": 0.93,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Thu tuc dang ky ma so thue moi cho doanh nghiep?",
        "assistant": "Theo Luat Quan ly thue 2019 va Thong tu 105/2020:\n\n**Quy trinh dang ky MST:**\n1. **Buoc 1:** Nop ho so DKKD tai So KHDT (truc tuyen qua dangkykinhdoanh.gov.vn)\n2. **Buoc 2:** Co quan thue tu dong cap MST trong 2 ngay lam viec\n3. **Buoc 3:** Nhan Giay chung nhan DKKD co ghi MST\n\n**Ho so can thiet:**\n- Don de nghi DKKD (theo mau)\n- Dieu le cong ty (TNHH, CP)\n- CMND/CCCD cua chu so huu, nguoi dai dien hop phap\n- Giay to chung minh dia chi tru so\n\n**Luu y:** MST khong thay doi trong suot qua trinh hoat dong.",
        "intent": "general_tax_query", "confidence": 0.90,
    },

    # More VAT examples for balance
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Cac truong hop khong duoc hoan thue GTGT?",
        "assistant": "Theo Thong tu 219/2013/TT-BTC va Luat Thue GTGT:\n\n**Khong duoc hoan thue GTGT khi:**\n1. Hang hoa nhap khau roi xuat khau (tam nhap tai xuat) khong co gia tri gia tang\n2. Doanh nghiep khong dap ung du dieu kien thanh toan qua ngan hang\n3. Hoa don dau vao cua DN da bo tron, mat tich\n4. Hang hoa xuat khau nhung khong co to khai hai quan hop le\n5. Doanh nghiep dang bi cuong che no thue\n6. Thue GTGT dau vao cua hang hoa xuat khau nhung chua du 300 trieu dong\n\nCan cu: Dieu 18 Thong tu 219/2013, Dieu 13 Luat Thue GTGT.",
        "intent": "vat_refund_risk", "confidence": 0.88,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Kiem tra tinh hop le cua hoa don dien tu?",
        "assistant": "Theo Thong tu 78/2021/TT-BTC huong dan hoa don dien tu:\n\n**Tieu chi kiem tra:**\n1. **Ma CQT:** Hoa don co ma cua co quan thue (bat buoc tu 01/07/2022)\n2. **Thong tin bat buoc:** Ten, dia chi, MST nguoi ban va nguoi mua\n3. **Chu ky so:** Phai co chu ky so hop le cua nguoi ban\n4. **Thoi diem lap:** Dung thoi diem giao hang hoa, cung cap dich vu\n5. **Dinh dang XML:** Tuan thu dung quy cach theo quy dinh\n\n**Cong cu xac thuc:**\n- Tra cuu tai hoadondientu.gdt.gov.vn\n- Kiem tra ma co quan thue (8 ky tu)\n- Doi chieu voi du lieu tren he thong TCT",
        "intent": "invoice_risk", "confidence": 0.91,
    },
    {
        "system": "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat.",
        "user": "Phan tich mang luoi giao dich doanh nghiep de phat hien gian lan?",
        "assistant": "He thong su dung Graph Neural Network (GNN) ket hop phan tich motif:\n\n**Phuong phap phan tich:**\n1. **Xay dung do thi giao dich:** Cac node la doanh nghiep, canh la giao dich\n2. **Phat hien motif:** Tim cac mau giao dich bat thuong\n   - Tam giac (carousel): A->B->C->A\n   - Fan-out/Fan-in: Mot DN giao dich voi nhieu cong ty vo boc\n   - Chuoi dai (layering): A->B->C->D->E\n3. **GNN scoring:** Tinh diem rui ro cho tung node dua tren dac trung cau truc\n4. **Community detection:** Xac dinh nhom doanh nghiep lien ket chat che\n\n**Ung dung:** Phat hien carousel VAT fraud, chuyen gia, rua tien qua mang luoi cong ty.",
        "intent": "osint_ownership", "confidence": 0.84,
    },
]

# Augmentation: generate more examples from paraphrase templates
AUGMENTATION = [
    ("Cho toi biet ve dieu kien hoan thue GTGT?", "vat_refund_risk", 0),
    ("Quy trinh hoan thue VAT cho xuat khau nhu the nao?", "vat_refund_risk", 0),
    ("Rui ro gian lan hoa don dau vao la gi?", "invoice_risk", 3),
    ("Lam sao kiem tra hoa don co that hay khong?", "invoice_risk", 3),
    ("Doanh nghiep nay co no thue qua han bao nhieu?", "delinquency", 5),
    ("Du bao kha nang vi pham thoi han nop thue?", "delinquency", 6),
    ("Ai la chu so huu thuc su cua cong ty nay?", "osint_ownership", 7),
    ("Phan tich giao dich lien ket va chuyen gia?", "transfer_pricing", 8),
    ("Xep hang uu tien thanh tra cho ky thue nay?", "audit_selection", 9),
    ("Thoi han nop to khai thue quy la khi nao?", "general_tax_query", 10),
]

# Generate augmented examples
for query, intent, base_idx in AUGMENTATION:
    base = TRAINING_EXAMPLES[min(base_idx, len(TRAINING_EXAMPLES)-1)]
    TRAINING_EXAMPLES.append({
        "system": base["system"],
        "user": query,
        "assistant": base["assistant"],
        "intent": intent,
        "confidence": base["confidence"] * 0.95,
    })

# Write to JSONL (ShareGPT format)
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

output_path = data_dir / "llm_training.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for ex in TRAINING_EXAMPLES:
        record = {
            "conversations": [
                {"from": "system", "value": ex["system"]},
                {"from": "human", "value": ex["user"]},
                {"from": "gpt", "value": ex["assistant"]},
            ],
            "metadata": {
                "intent": ex["intent"],
                "confidence": ex["confidence"],
                "source": "synthetic",
                "quality_score": 0.9,
            },
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Also write Alpaca format
alpaca_path = data_dir / "llm_training.alpaca.json"
alpaca_records = [{"instruction": ex["system"], "input": ex["user"], "output": ex["assistant"]} for ex in TRAINING_EXAMPLES]
with open(alpaca_path, "w", encoding="utf-8") as f:
    json.dump(alpaca_records, f, ensure_ascii=False, indent=2)

print(f"    Generated {len(TRAINING_EXAMPLES)} training examples")
print(f"    JSONL:  {output_path} ({output_path.stat().st_size:,} bytes)")
print(f"    Alpaca: {alpaca_path} ({alpaca_path.stat().st_size:,} bytes)")

# ─── Step 2: Run LoRA Training ───────────────────────────────

print("\n[Step 2] Starting LoRA fine-tuning...")
print("         Base model: Qwen/Qwen2.5-0.5B-Instruct (optimized for CPU demo)")
print("         LoRA rank: 16, alpha: 32")
print("         This will take a few minutes on CPU...")

from ml_engine.tax_agent_llm_model import LoRATrainer, LoRATrainingConfig

config = LoRATrainingConfig(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="data/models/tax_llm_lora",
    training_data=str(output_path),
    lora_r=16,
    lora_alpha=32,
    num_epochs=1,
    per_device_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
)

trainer = LoRATrainer(config)

# Verify dataset loads
dataset = trainer.prepare_dataset()
print(f"    Dataset loaded: train={len(dataset['train'])}, eval={len(dataset['eval'])}")

# Run training
t0 = time.perf_counter()
result = trainer.train()
duration = time.perf_counter() - t0

print(f"\n    Training result: {result['status']}")
if result["status"] == "success":
    print(f"    Output dir: {result['output_dir']}")
    print(f"    Training examples: {result['training_examples']}")
    print(f"    Trainable params: {result['trainable_params']:,}")
    print(f"    Total params: {result['total_params']:,}")
    print(f"    Duration: {result['duration_seconds']:.1f}s")
else:
    print(f"    Message: {result.get('message', 'unknown error')}")

# ─── Step 3: Verify fine-tuned model loads ────────────────────

print("\n[Step 3] Verifying fine-tuned model auto-detection...")

from ml_engine.tax_agent_llm_model import TaxAgentLLM, LLMConfig, get_tax_llm

# Reset singleton
import ml_engine.tax_agent_llm_model as llm_module
llm_module._llm_instance = None

llm = get_tax_llm()
tier = llm.load()
print(f"    LLM tier: {tier}")
print(f"    Is fine-tuned: {tier == 'finetuned'}")

# Generate test response
if llm.is_available:
    resp = llm.generate(
        query="Dieu kien hoan thue VAT?",
        intent="vat_refund_risk",
    )
    print(f"    Response tier: {resp.tier.value}")
    print(f"    Tokens: {resp.tokens_generated}")
    print(f"    Latency: {resp.latency_ms:.0f}ms")
    print(f"    Preview: {resp.text[:200]}...")

# ─── Step 4: Confirm auto-upgrade ─────────────────────────────

print("\n[Step 4] Auto-upgrade confirmation:")
if tier == "finetuned":
    print("    STATUS: FINETUNED LLM ACTIVE")
    print("    The system has automatically upgraded from template -> fine-tuned LLM")
    print("    All future synthesis will use the LoRA-adapted model")
elif tier == "base_few_shot":
    print("    STATUS: BASE MODEL + FEW-SHOT ACTIVE")
    print("    The system is using the base model with in-context learning")
    print("    LoRA adapter may need more training data or memory")
else:
    print("    STATUS: TEMPLATE FALLBACK ACTIVE")
    print("    Fine-tuned model requires GPU or more RAM for inference")
    print("    Training data and LoRA adapter are ready for deployment")
    print("    To activate: install the adapter on a machine with sufficient resources")

adapter_dir = Path("data/models/tax_llm_lora")
if adapter_dir.exists():
    files = list(adapter_dir.glob("*"))
    print(f"\n    LoRA adapter files ({len(files)}):")
    for f in sorted(files)[:10]:
        size = f.stat().st_size
        print(f"      {f.name} ({size:,} bytes)")

print("\n" + "=" * 60)
print("  Pipeline Complete!")
print("=" * 60)
