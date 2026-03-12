from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
sys.modules.setdefault("soundfile", None)
sys.modules.setdefault("librosa", None)

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def exact_match(reference: str, prediction: str) -> int:
    return int(reference.strip() == prediction.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on Medumba pair test set.")
    parser.add_argument("--test-file", type=Path, default=Path("outputs/pairs_test.jsonl"))
    parser.add_argument("--base-model", type=str, default="google/byt5-small")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/medumba_lora_byt5"))
    parser.add_argument("--task", type=str, default="", help="Optional: fr_to_medumba or medumba_to_fr")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--output-report", type=Path, default=Path("outputs/eval_report.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.test_file)
    if args.task.strip():
        rows = [row for row in rows if row.get("task") == args.task.strip()]

    rows = rows[: args.samples]
    if not rows:
        raise ValueError("No test rows found. Check file path and optional --task filter.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, use_safetensors=True)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    em_total = 0

    for row in rows:
        model_input = row["input"]
        reference = row["output"]
        task = row.get("task", "")

        encoded = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=128)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=4,
            )

        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        em = exact_match(reference, prediction)
        em_total += em

        predictions.append(
            {
                "id": row.get("id", ""),
                "task": task,
                "input": model_input,
                "reference": reference,
                "prediction": prediction,
                "exact_match": em,
                "source": row.get("source", ""),
            }
        )

    try:
        import sacrebleu

        references = [[row["reference"] for row in predictions]]
        hypotheses = [row["prediction"] for row in predictions]
        chrf = sacrebleu.corpus_chrf(hypotheses, references).score
    except Exception:
        chrf = None

    report = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "test_file": str(args.test_file),
        "task_filter": args.task.strip() or None,
        "rows_evaluated": len(predictions),
        "exact_match": em_total / len(predictions),
        "chrf": chrf,
        "samples": predictions[:25],
    }

    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    print(f"Wrote detailed report: {args.output_report}")


if __name__ == "__main__":
    main()
