"""Generate TaxInspector user-study protocol, questionnaires and simulated data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


SUS_ITEMS = [
    "Toi nghi minh se muon su dung TaxInspector thuong xuyen.",
    "Toi thay he thong nay phuc tap khong can thiet.",
    "Toi thay TaxInspector de su dung.",
    "Toi can ho tro ky thuat moi co the su dung he thong.",
    "Cac chuc nang trong TaxInspector duoc tich hop tot.",
    "Toi thay he thong co qua nhieu diem khong nhat quan.",
    "Toi nghi can bo thue co the hoc cach dung TaxInspector nhanh.",
    "Toi thay he thong kho su dung.",
    "Toi tu tin khi su dung TaxInspector.",
    "Toi can hoc rat nhieu truoc khi su dung thanh thao.",
]

TAM_ITEMS = {
    "PU": [
        "TaxInspector giup toi phan tich ho so thue nhanh hon.",
        "TaxInspector cai thien chat luong ra quyet dinh thanh tra.",
        "TaxInspector giup giam bo sot rui ro quan trong.",
        "TaxInspector giup trich dan can cu phap ly ro rang hon.",
        "TaxInspector huu ich cho cong viec nghiep vu hang ngay.",
        "TaxInspector giup uu tien nguon luc thanh tra hieu qua.",
    ],
    "PEOU": [
        "Toi thay thao tac voi TaxInspector ro rang.",
        "Toi de dang chuyen giua che do phap ly, gian lan, VAT va no dong.",
        "Toi de hieu trang thai dang xu ly/loi/hoan tat cua agent.",
        "Ket qua tra ve duoc trinh bay de kiem chung.",
        "Toi khong mat nhieu cong suc de hoc cach su dung.",
        "TaxInspector phan hoi tot voi cau hoi tu nhien cua toi.",
    ],
    "BI": [
        "Toi se de xuat dung TaxInspector trong quy trinh thu nghiem.",
        "Toi san sang dung TaxInspector cho ho so rui ro cao.",
        "Toi tin TaxInspector co the ho tro quy trinh nghiep vu trong don vi.",
    ],
}

TASK_SCENARIOS = [
    {
        "id": "T01",
        "mode": "legal",
        "title": "Tu van nguoi dan nop to khai GTGT quy bi tre han",
        "success_criteria": "Tra loi han nop, rui ro phat, buoc khac phuc va can cu Luat QLT/ND125.",
    },
    {
        "id": "T02",
        "mode": "legal",
        "title": "Giai thich mua hang tren 20 trieu thanh toan tien mat",
        "success_criteria": "Neu dieu kien khau tru VAT/chi phi duoc tru va khuyen nghi chung tu.",
    },
    {
        "id": "T03",
        "mode": "fraud",
        "title": "Tra cuu MST rui ro cao va giai thich top SHAP",
        "success_criteria": "Dung model fraud, neu score, drivers, hanh dong thanh tra.",
    },
    {
        "id": "T04",
        "mode": "vat",
        "title": "Truy vet vong hoa don VAT giua cac doanh nghiep",
        "success_criteria": "Dung GraphRAG/VAT graph, neu vong, nut trung tam, bang chung hoa don.",
    },
    {
        "id": "T05",
        "mode": "delinquency",
        "title": "Du bao no dong va de xuat bien phap cuong che",
        "success_criteria": "Dung delinquency + causal uplift, giai thich ly do va khuyen nghi.",
    },
    {
        "id": "T06",
        "mode": "macro",
        "title": "Mo phong giam VAT 2 diem phan tram trong quy toi",
        "success_criteria": "Tra workspace mo phong, tham so, sensitivity, so sanh kich ban.",
    },
    {
        "id": "T07",
        "mode": "legal",
        "title": "Ca nhan ban hang online qua san va nguong doanh thu",
        "success_criteria": "Hoi them nam tinh thue neu can, neu TT40/TT100 va nghia vu ke khai.",
    },
    {
        "id": "T08",
        "mode": "full",
        "title": "Upload CSV 50 ho so va hoi follow-up vi sao cong ty A rui ro",
        "success_criteria": "Dung snapshot upload, giai thich theo cau truoc, khong mat ngu canh.",
    },
    {
        "id": "T09",
        "mode": "legal",
        "title": "Nguoi dung go khong dau/sai chinh ta: 'xn chao hoi thue tncn'",
        "success_criteria": "Nhan dien loi chao/khong dau, dieu huong tu van TNCN dung.",
    },
    {
        "id": "T10",
        "mode": "vat",
        "title": "Kiem tra hoan thue GTGT du an dau tu co hoa don truoc giay phep",
        "success_criteria": "Neu dieu kien hoan, canh bao rui ro ho so va can cu lien quan.",
    },
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sus_score(responses: list[int]) -> float:
    total = 0
    for idx, value in enumerate(responses):
        if idx % 2 == 0:
            total += value - 1
        else:
            total += 5 - value
    return round(total * 2.5, 2)


def _mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 3) if values else 0.0


def simulate_expert_responses(experts: int = 10, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    participants = []
    task_rows = []
    for idx in range(1, experts + 1):
        role = rng.choice(["Can bo thanh tra", "Can bo ke khai", "Can bo quan ly no", "Chuyen vien phap che"])
        years = rng.randint(4, 22)
        sus = []
        for item_idx in range(10):
            positive = item_idx % 2 == 0
            base = rng.choice([4, 4, 5]) if positive else rng.choice([1, 2, 2])
            sus.append(max(1, min(5, base + rng.choice([-1, 0, 0, 1]))))
        tam = {
            group: [max(1, min(5, rng.choice([4, 4, 5]) + rng.choice([-1, 0, 0, 1]))) for _ in items]
            for group, items in TAM_ITEMS.items()
        }
        participant = {
            "participant_id": f"E{idx:02d}",
            "role": role,
            "years_experience": years,
            "sus_responses": sus,
            "sus_score": sus_score(sus),
            "tam_responses": tam,
            "tam_scores": {group: _mean(values) for group, values in tam.items()},
        }
        participants.append(participant)
        for task in TASK_SCENARIOS:
            difficulty = 1.15 if task["mode"] in {"full", "vat", "macro"} else 1.0
            accuracy = rng.random() < (0.88 if task["mode"] == "legal" else 0.84)
            if task["id"] == "T09":
                accuracy = rng.random() < 0.90
            task_rows.append(
                {
                    "participant_id": participant["participant_id"],
                    "task_id": task["id"],
                    "mode": task["mode"],
                    "completed": accuracy or rng.random() < 0.08,
                    "accuracy": accuracy,
                    "time_seconds": round(rng.uniform(80, 260) * difficulty, 1),
                    "confidence": max(1, min(5, rng.choice([3, 4, 4, 5]) + (1 if accuracy else -1))),
                }
            )
    return {"participants": participants, "task_results": task_rows}


def analyze_study(simulated: dict[str, Any]) -> dict[str, Any]:
    participants = simulated["participants"]
    tasks = simulated["task_results"]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in tasks:
        by_mode.setdefault(row["mode"], []).append(row)
    return {
        "participant_count": len(participants),
        "sus_mean": _mean([p["sus_score"] for p in participants]),
        "sus_std": round(float(statistics.pstdev([p["sus_score"] for p in participants])), 3),
        "tam_mean": {
            group: _mean([p["tam_scores"][group] for p in participants])
            for group in TAM_ITEMS
        },
        "task_completion_rate": round(sum(1 for r in tasks if r["completed"]) / max(1, len(tasks)), 4),
        "task_accuracy_rate": round(sum(1 for r in tasks if r["accuracy"]) / max(1, len(tasks)), 4),
        "mean_time_seconds": _mean([r["time_seconds"] for r in tasks]),
        "mode_accuracy": {
            mode: round(sum(1 for r in rows if r["accuracy"]) / max(1, len(rows)), 4)
            for mode, rows in sorted(by_mode.items())
        },
    }


def build_questionnaire() -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "scale": "1=rat khong dong y, 5=rat dong y",
        "sus": [{"id": f"SUS{i+1:02d}", "text": text} for i, text in enumerate(SUS_ITEMS)],
        "tam": {
            group: [{"id": f"{group}{i+1:02d}", "text": text} for i, text in enumerate(items)]
            for group, items in TAM_ITEMS.items()
        },
        "tasks": TASK_SCENARIOS,
    }


def write_outputs(out_dir: Path, experts: int, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    questionnaire = build_questionnaire()
    simulated = simulate_expert_responses(experts=experts, seed=seed)
    analysis = analyze_study(simulated)

    (out_dir / "user_study_questionnaire.json").write_text(json.dumps(questionnaire, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "user_study_simulated_responses.json").write_text(json.dumps(simulated, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "user_study_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "user_study_task_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["participant_id", "task_id", "mode", "completed", "accuracy", "time_seconds", "confidence"])
        writer.writeheader()
        writer.writerows(simulated["task_results"])

    protocol = [
        "# TaxInspector User Study Protocol",
        "",
        f"Generated at: `{_now_iso()}`",
        "",
        "## Objective",
        "Evaluate usability, perceived usefulness, and task accuracy of the TaxInspector multi-agent assistant across legal, fraud, VAT, delinquency, macro and full-analysis workflows.",
        "",
        "## Participants",
        f"Target: 5-10 tax-domain evaluators. This package includes `{experts}` simulated expert responses for dry-run analysis only.",
        "",
        "## Instruments",
        "- SUS questionnaire, Vietnamese translation, 10 items.",
        "- TAM questionnaire: PU, PEOU and BI.",
        "- 10 task-completion scenarios with time, accuracy and confidence logging.",
        "",
        "## Simulated Dry-Run Summary",
        f"- SUS mean: `{analysis['sus_mean']}`",
        f"- Task accuracy: `{analysis['task_accuracy_rate']}`",
        f"- Completion rate: `{analysis['task_completion_rate']}`",
        "",
        "## Important",
        "Simulated responses are for pipeline validation and thesis dry-runs; publication claims should use real evaluator responses.",
    ]
    (out_dir / "user_study_protocol.md").write_text("\n".join(protocol), encoding="utf-8")
    return {"questionnaire": questionnaire, "simulated": simulated, "analysis": analysis}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate user-study protocol and simulated expert responses")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "reports" / "user_study")
    parser.add_argument("--experts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = write_outputs(args.out_dir, experts=args.experts, seed=args.seed)
    print(f"[OK] wrote user study package to {args.out_dir}")
    print(json.dumps(result["analysis"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
