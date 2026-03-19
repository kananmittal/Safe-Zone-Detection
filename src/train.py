#!/usr/bin/env python3
import os
import json
import random
import argparse
import subprocess
from pathlib import Path
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-Tuning setup")
    parser.add_argument("--data", default="datasets/processed/fine_tuning_data.json")
    parser.add_argument("--model", default="mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    parser.add_argument("--outdir", default="datasets/mlx_lora")
    parser.add_argument("--iters", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--resume-adapter-file", type=str, default="", help="Path to adapter file to resume from")
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading '{args.model}' tokenizer for chat formatting...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Loading data from {args.data}...")
    with open(args.data, "r") as f:
        data = json.load(f)
        
    random.shuffle(data)
    
    # Split 90/10
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    
    def process_and_save(dataset_split, filename):
        filepath = outdir / filename
        with open(filepath, "w") as f:
            for item in dataset_split:
                messages = [
                    {"role": "system", "content": "You are a crisis response AI that analyzes voice and text for distress. You MUST respond exactly with the specified output format."},
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": item['response']}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                json.dump({"text": text}, f)
                f.write("\n")
        print(f"Saved {len(dataset_split)} records to {filepath}")
        
    process_and_save(train_data, "train.jsonl")
    process_and_save(valid_data, "valid.jsonl")
    
    print("\nStarting MLX LoRA Fine-Tuning...\n")
    # Build mlx_lm.lora command
    cmd = [
        ".venv/bin/python", "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(outdir),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", "adapters" # Default adapter output directory
    ]
    
    if args.resume_adapter_file:
        cmd.extend(["--resume-adapter-file", args.resume_adapter_file])
    
    print("Executing command:")
    print(" ".join(cmd))
    
    try:
        # Run MLX LoRA training as a subprocess
        subprocess.run(cmd, check=True)
        print("\n✅ Fine-Tuning completed successfully!")
        print("Adapters saved to the 'adapters' directory. Inference script will automatically load them.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Fine-Tuning failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()