from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
sys.modules.setdefault("soundfile", None)
sys.modules.setdefault("librosa", None)

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset(path: Path, task_filter: str | None = None) -> Dataset:
    rows = read_jsonl(path)
    if task_filter:
        rows = [row for row in rows if row.get("task") == task_filter]
    return Dataset.from_list(rows)


def tokenize_function(batch: dict[str, list[str]], tokenizer: AutoTokenizer, max_source_len: int, max_target_len: int) -> dict[str, Any]:
    model_inputs = tokenizer(
        batch["input"],
        max_length=max_source_len,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["output"],
        max_length=max_target_len,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Medumba translation pairs.")
    parser.add_argument("--train-file", type=Path, default=Path("outputs/pairs_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("outputs/pairs_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/medumba_lora_byt5"))
    parser.add_argument("--base-model", type=str, default="google/byt5-small")
    parser.add_argument("--task", type=str, default="", help="Optional: fr_to_medumba or medumba_to_fr")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-source-len", type=int, default=128)
    parser.add_argument("--max-target-len", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    task_filter = args.task.strip() or None
    train_dataset = build_dataset(args.train_file, task_filter)
    val_dataset = build_dataset(args.val_file, task_filter)

    if len(train_dataset) == 0:
        raise ValueError("No training rows found. Check file path and optional --task filter.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, use_safetensors=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "base_model": args.base_model,
        "task_filter": task_filter,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
