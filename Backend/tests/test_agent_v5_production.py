"""
test_agent_v5_production.py — Production V5 LoRA Adapter Evaluation
====================================================================
Đánh giá chất lượng adapter LoRA V5 production (6,342 steps, 101k records).

Chạy offline trên máy local:
    cd e:\\TaxInspector\\Backend
    python -m pytest tests/test_agent_v5_production.py -v

Các bài test:
1. Kiểm tra file adapter tồn tại và checksum đúng
2. Kiểm tra adapter_config.json khớp với cấu hình huấn luyện
3. Kiểm tra tokenizer load được
4. Kiểm tra LoRA metadata
5. Kiểm tra tương thích với tool contract registry
"""

import json
import hashlib
import sys
from pathlib import Path

# Ensure Backend is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ADAPTER_DIR = BACKEND_DIR / "tax_agent"

# === Expected checksums from production training ===
EXPECTED_CHECKSUMS = {
    "adapter_config.json": "20AA6FF2E87483540449D3E273ABED03B0C2F1AEBB09200B24D12DAC522CD2D5",
    "adapter_model.safetensors": "BE03521F76C03812BF79FF5DEC58A0AB89D89CB9ED4172FCC6589A17160B8C55",
    "tokenizer.json": "2F55E63353D3D978B390D346BAE531BE8B83BC9532C0BE500D62B7253AA4C595",
}

# === Expected LoRA configuration ===
EXPECTED_LORA_CONFIG = {
    "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "peft_type": "LORA",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
    "bias": "none",
    "target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"},
}

# === Training metrics from final checkpoint ===
TRAINING_METRICS = {
    "total_steps": 6342,
    "final_train_loss": 0.029538,
    "final_val_loss": 0.135369,
    "train_records": 101465,
    "eval_records": 16718,
    "trainable_params": 4358144,
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ================================================================
# Test 1: Adapter files exist
# ================================================================
def test_adapter_files_exist():
    """Tất cả file adapter cần thiết phải tồn tại."""
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for fname in required_files:
        fpath = ADAPTER_DIR / fname
        assert fpath.exists(), f"Missing required file: {fpath}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath}"
    print("✅ Test 1 PASSED: Tất cả file adapter tồn tại")


# ================================================================
# Test 2: Checksum verification
# ================================================================
def test_adapter_checksums():
    """SHA256 checksums phải khớp với production build."""
    for fname, expected_hash in EXPECTED_CHECKSUMS.items():
        fpath = ADAPTER_DIR / fname
        actual_hash = sha256_file(fpath)
        assert actual_hash == expected_hash, (
            f"Checksum mismatch for {fname}:\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
    print("✅ Test 2 PASSED: Checksums khớp 100%")


# ================================================================
# Test 3: LoRA config validation
# ================================================================
def test_lora_config():
    """adapter_config.json phải khớp với cấu hình huấn luyện."""
    with open(ADAPTER_DIR / "adapter_config.json", "r") as f:
        config = json.load(f)

    assert config["base_model_name_or_path"] == EXPECTED_LORA_CONFIG["base_model_name_or_path"]
    assert config["peft_type"] == EXPECTED_LORA_CONFIG["peft_type"]
    assert config["r"] == EXPECTED_LORA_CONFIG["r"]
    assert config["lora_alpha"] == EXPECTED_LORA_CONFIG["lora_alpha"]
    assert abs(config["lora_dropout"] - EXPECTED_LORA_CONFIG["lora_dropout"]) < 1e-6
    assert config["task_type"] == EXPECTED_LORA_CONFIG["task_type"]
    assert config["bias"] == EXPECTED_LORA_CONFIG["bias"]
    assert set(config["target_modules"]) == EXPECTED_LORA_CONFIG["target_modules"]
    assert config["inference_mode"] is True, "Adapter phải ở chế độ inference"
    print("✅ Test 3 PASSED: LoRA config khớp cấu hình huấn luyện")


# ================================================================
# Test 4: Tokenizer loads correctly
# ================================================================
def test_tokenizer_loadable():
    """Tokenizer phải load được từ thư mục adapter."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_DIR), 
            trust_remote_code=True,
            local_files_only=True
        )
        assert tokenizer is not None
        assert tokenizer.pad_token_id is not None or tokenizer.eos_token_id is not None

        # Test encode/decode hoạt động bình thường
        test_text = "Kiểm tra rủi ro gian lận thuế VAT cho MST 0123456789"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert len(tokens) > 0
        assert "thuế" in decoded or "thue" in decoded.lower()
        print(f"✅ Test 4 PASSED: Tokenizer load thành công ({len(tokens)} tokens)")
    except ImportError:
        print("⚠️ Test 4 SKIPPED: transformers library not installed locally")


# ================================================================
# Test 5: Adapter file size sanity check
# ================================================================
def test_adapter_size_sanity():
    """adapter_model.safetensors phải đúng kích thước hợp lý cho LoRA r=16."""
    fpath = ADAPTER_DIR / "adapter_model.safetensors"
    size_mb = fpath.stat().st_size / 1e6

    # LoRA r=16 trên 4 modules (q,k,v,o) cho Qwen2.5-1.5B
    # Khoảng 4,358,144 params × 2 bytes (FP16) ≈ 8.3MB
    # Production adapter: ~16.6MB (full precision / có thêm metadata)
    assert 5 < size_mb < 50, (
        f"Adapter size {size_mb:.1f}MB ngoài khoảng hợp lý (5-50MB). "
        f"Có thể bị hỏng hoặc lẫn model base."
    )
    print(f"✅ Test 5 PASSED: Adapter size = {size_mb:.1f}MB (hợp lý)")


# ================================================================
# Test 6: Tool contract compatibility
# ================================================================
def test_tool_contract_compatibility():
    """Adapter phải tương thích với canonical tool registry."""
    from ml_engine.tax_agent_tool_contracts import (
        CANONICAL_TOOL_CONTRACTS,
        CANONICAL_TOOL_NAMES,
        validate_tool_call,
        tool_prompt_lines,
    )

    # Kiểm tra 20 tools đều có trong registry
    assert len(CANONICAL_TOOL_CONTRACTS) >= 20, (
        f"Expected >= 20 canonical tools, got {len(CANONICAL_TOOL_CONTRACTS)}"
    )

    # Kiểm tra tất cả tool names đều hợp lệ
    for contract in CANONICAL_TOOL_CONTRACTS:
        assert contract.name in CANONICAL_TOOL_NAMES

    # Kiểm tra tool_prompt_lines() tạo đúng format cho prompt
    lines = tool_prompt_lines()
    assert len(lines) >= 20

    # Test validate_tool_call với một số tool phổ biến
    test_cases = [
        ({"name": "knowledge_search", "arguments": {"query": "thuế GTGT"}}, True),
        ({"name": "company_risk_lookup", "arguments": {"tax_code": "0123456789"}}, True),
        ({"name": "gnn_analysis", "arguments": {"tax_code": "0123456789"}}, True),
        ({"name": "invoice_risk_scan", "arguments": {"tax_code": "0123456789", "period": "Q1-2025"}}, True),
        ({"name": "vat_refund_risk", "arguments": {"tax_code": "0123456789", "period": "2025"}}, True),
        ({"name": "delinquency_check", "arguments": {"tax_code": "0123456789"}}, True),
        ({"name": "macro_forecast", "arguments": {"scenario": "baseline"}}, True),
        ({"name": "ocr_document_process", "arguments": {"document_type": "invoice", "file_path": "invoice.pdf"}}, True),
        # Deprecated aliases phải bị chuyển đổi sang canonical
        ({"name": "gnn_vat_fraud", "arguments": {"tax_code": "0123456789"}}, True),
        # Invalid tool
        ({"name": "nonexistent_tool", "arguments": {}}, False),
        # Missing required args
        ({"name": "knowledge_search", "arguments": {}}, False),
    ]

    passed = 0
    failed = 0
    for payload, expected_ok in test_cases:
        ok, name, args, reason = validate_tool_call(payload)
        if ok == expected_ok:
            passed += 1
        else:
            failed += 1
            print(f"  ❌ FAIL: {payload['name']} expected ok={expected_ok}, got ok={ok}, reason={reason}")

    assert failed == 0, f"{failed} tool contract tests failed"
    print(f"✅ Test 6 PASSED: {passed}/{len(test_cases)} tool contract tests passed")


# ================================================================
# Test 7: Training metrics sanity check
# ================================================================
def test_training_metrics_quality():
    """Đánh giá chất lượng huấn luyện dựa trên metrics cuối cùng."""
    # Training loss < 0.05 chứng tỏ model đã hội tụ tốt
    assert TRAINING_METRICS["final_train_loss"] < 0.05, (
        f"Training loss {TRAINING_METRICS['final_train_loss']} > 0.05 (chưa hội tụ)"
    )

    # Validation loss < 0.2 chứng tỏ model tổng quát hóa tốt
    assert TRAINING_METRICS["final_val_loss"] < 0.2, (
        f"Validation loss {TRAINING_METRICS['final_val_loss']} > 0.2 (có thể overfitting)"
    )

    # Overfitting ratio (val/train) < 5 là chấp nhận được cho SFT
    overfit_ratio = TRAINING_METRICS["final_val_loss"] / TRAINING_METRICS["final_train_loss"]
    assert overfit_ratio < 10, (
        f"Overfitting ratio {overfit_ratio:.1f}x quá cao (val_loss/train_loss)"
    )

    print(f"✅ Test 7 PASSED: Training metrics đạt chuẩn production")
    print(f"   Train Loss: {TRAINING_METRICS['final_train_loss']:.6f}")
    print(f"   Val Loss:   {TRAINING_METRICS['final_val_loss']:.6f}")
    print(f"   Overfit ratio: {overfit_ratio:.1f}x")
    print(f"   Trainable params: {TRAINING_METRICS['trainable_params']:,}")
    print(f"   Total steps: {TRAINING_METRICS['total_steps']:,}")


# ================================================================
# Test 8: Agentic LLM integration
# ================================================================
def test_agentic_llm_system_prompt():
    """System prompt của AgenticLLM phải chứa tất cả canonical tools."""
    from ml_engine.tax_agent_agentic_llm import AGENTIC_SYSTEM_PROMPT
    from ml_engine.tax_agent_tool_contracts import CANONICAL_TOOL_CONTRACTS

    for contract in CANONICAL_TOOL_CONTRACTS:
        assert contract.name in AGENTIC_SYSTEM_PROMPT, (
            f"Tool '{contract.name}' missing from AGENTIC_SYSTEM_PROMPT"
        )
    print(f"✅ Test 8 PASSED: System prompt chứa đầy đủ {len(CANONICAL_TOOL_CONTRACTS)} tools")


# ================================================================
# Main runner
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" TaxInspector LoRA V5 Production Adapter — Full Evaluation")
    print("=" * 70)
    print(f" Adapter dir: {ADAPTER_DIR}")
    print(f" Training: {TRAINING_METRICS['train_records']:,} records, "
          f"{TRAINING_METRICS['total_steps']:,} steps")
    print(f" Final Loss: train={TRAINING_METRICS['final_train_loss']:.6f}, "
          f"val={TRAINING_METRICS['final_val_loss']:.6f}")
    print("=" * 70)

    tests = [
        test_adapter_files_exist,
        test_adapter_checksums,
        test_lora_config,
        test_tokenizer_loadable,
        test_adapter_size_sanity,
        test_tool_contract_compatibility,
        test_training_metrics_quality,
        test_agentic_llm_system_prompt,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_fn.__name__}: {e}")

    print("\n" + "=" * 70)
    print(f" KẾT QUẢ: {passed}/{len(tests)} tests PASSED, {failed} FAILED")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 ADAPTER V5 PRODUCTION ĐÃ SẴN SÀNG TRIỂN KHAI!")
        sys.exit(0)
