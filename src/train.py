#!/usr/bin/env python3
"""
Unified LLM Fine-Tuning Script (single entry point)
- Loads tokenizer/model from a checkpoint path (default: latest local checkpoint) or base model
- Trains on datasets/processed/fine_tuning_data.json
- Supports a faster configuration via --fast
- Saves final model to models/<run_dir>/final_model_<timestamp>
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(root: str = "models") -> str:
    candidates = []
    if not os.path.isdir(root):
        return ""
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-", 1)[1])
                except Exception:
                    step = -1
                candidates.append((step, os.path.join(r, d)))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_training_texts(data_path: str, subset: int = 0):
    with open(data_path, "r") as f:
        data = json.load(f)
    if subset and subset > 0:
        data = data[:subset]
    texts = []
    for item in data:
        texts.append(item["prompt"] + "\n" + item["response"] + "\n<|endoftext|>\n")
    return texts


def prepare_dataset(texts, tokenizer, max_length: int):
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        lambda ex: tokenizer(
            ex["text"], truncation=True, padding="max_length", max_length=max_length
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/processed/fine_tuning_data.json")
    parser.add_argument("--checkpoint", default="", help="Path to model or checkpoint to load")
    parser.add_argument("--base", default="microsoft/DialoGPT-medium", help="Base model if no checkpoint")
    parser.add_argument("--fast", action="store_true", help="Use faster training config")
    parser.add_argument("--subset", type=int, default=0, help="Use first N samples for quick runs")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--outdir", default="models/fine_tuned_llama_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = find_latest_checkpoint()
    if checkpoint:
        logger.info(f"Loading from checkpoint: {checkpoint}")
        model_name = checkpoint
    else:
        logger.info(f"Loading base model: {args.base}")
        model_name = args.base

    # Tokenizer/Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    # Data
    texts = load_training_texts(args.data, args.subset)
    tokenized = prepare_dataset(texts, tokenizer, args.max_length)

    # Training config
    config = {
        "learning_rate": 5e-5 if args.fast else 2e-5,
        "num_train_epochs": 2 if args.fast else 3,
        "per_device_train_batch_size": 2 if args.fast else 1,
        "per_device_eval_batch_size": 2 if args.fast else 1,
        "warmup_steps": 50 if args.fast else 100,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 100 if args.fast else 50,
        "save_steps": 1000 if args.fast else 500,
        "eval_steps": 1000 if args.fast else 500,
        "save_total_limit": 2 if args.fast else 3,
        "prediction_loss_only": True,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "report_to": None,
        "gradient_accumulation_steps": 2 if args.fast else 4,
        "lr_scheduler_type": "cosine" if args.fast else "linear",
        "warmup_ratio": 0.1 if args.fast else 0.0,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(outdir), **config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    resume_from = checkpoint if "checkpoint-" in model_name else None
    result = trainer.train(resume_from_checkpoint=resume_from)

    final_dir = outdir / ("fast_final_model_" if args.fast else "final_model_") / datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = result.metrics
    with open(final_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n🎉 Training complete")
    print(f"📁 Saved to: {final_dir}")
    print(f"📊 Metrics: {metrics}")


if __name__ == "__main__":
    main()