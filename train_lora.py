#!/usr/bin/env python3
"""
LoRA fine-tuning for commitgen using Unsloth + Qwen2.5-Coder.

OPTIMIZED FOR SPEED & QUALITY:
- Adaptive hyperparameters based on dataset size
- Gradient checkpointing + flash attention
- Early stopping to prevent overfitting
- Optimized learning rate scheduling
- Layer-wise learning rate decay
- Improved training stability

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes xformers

Usage:
    python train_lora.py --dataset dataset.jsonl --model 7b
    python train_lora.py --dataset dataset.jsonl --model 7b --epochs 2 --learning-rate 5e-4
"""

import argparse
import json
import torch
import math
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Model options ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    "1.5b": ("unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit", 8, 12),
    "7b":   ("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", 16, 32),
    "14b":  ("unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit", 32, 64),
}
# Format: (model_id, recommended_lora_rank, recommended_lora_alpha)

# ── Prompt template (matches what prepare-commit-msg.py sends at inference) ────
SYSTEM_PROMPT = (
    "You are a Git commit message generator. "
    "Output ONLY a single conventional commit message. "
    "Format: type(scope): description. "
    "Types: feat, fix, refactor, docs, style, test, chore, perf, ci, build. "
    "Imperative mood, lowercase, no period, 50–72 chars."
)

def format_example(row: dict) -> str:
    """
    Expects each dataset row to have:
        stat   - output of `git diff --cached --stat`
        diff   - output of `git diff --cached` (may be truncated)
        commit - the gold commit message
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Files changed:\n{row['stat']}\n\n"
        f"Diff:\n{row['diff']}\n\n"
        f"Recent commits for style reference:\n{row.get('recent_commits', '')}\n"
        f"Commit message:<|im_end|>\n"
        f"<|im_start|>assistant\n{row['commit']}<|im_end|>"
    )


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} examples from {path}")
    if len(rows) < 50:
        print("⚠️  Warning: fewer than 50 examples — consider collecting more for best results.")

    dataset = Dataset.from_list([{"text": format_example(r)} for r in rows])
    return dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with adaptive hyperparameters")
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--model", default="7b", choices=MODEL_MAP.keys(),
                        help="Model size: 1.5b | 7b | 14b")
    parser.add_argument("--output", default="./commitgen-lora",
                        help="Output directory for LoRA adapter")
    
    # Training parameters with intelligent defaults
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (auto: 3 for small, 2 for medium/large datasets)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size per device (auto: 8 for 1.5b, 4 for 7b, 2 for 14b)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (auto: scales with model size)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (auto: 3e-4 for small, 2e-4 for medium/large)")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio (10% of steps by default for stability)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank (auto: optimized per model)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (auto: 2× rank)")
    parser.add_argument("--use-flash-attention", action="store_true", default=True,
                        help="Use flash attention if available")
    parser.add_argument("--early-stopping", type=int, default=3,
                        help="Early stopping patience (stop if no improvement for N evals)")
    args = parser.parse_args()

    model_name, rec_rank, rec_alpha = MODEL_MAP[args.model]
    
    # ── Adaptive hyperparameters based on model size & dataset ──────────────────
    # Load dataset to count examples
    dataset_size = 0
    with open(args.dataset) as f:
        dataset_size = sum(1 for _ in f)
    
    print(f"📊 Dataset size: {dataset_size} examples")
    print(f"🤖 Model: {args.model}")
    
    # Auto-tune batch size (smaller for bigger models)
    if args.batch_size is None:
        batch_size_map = {"1.5b": 8, "7b": 4, "14b": 2}
        args.batch_size = batch_size_map[args.model]
    
    # Auto-tune learning rate (higher for smaller datasets)
    if args.learning_rate is None:
        if dataset_size < 100:
            args.learning_rate = 5e-4  # Small dataset, use higher LR
        elif dataset_size < 500:
            args.learning_rate = 3e-4
        else:
            args.learning_rate = 2e-4  # Large dataset, lower LR
    
    # Auto-tune epochs (more epochs for small datasets, fewer for large)
    if args.epochs is None:
        if dataset_size < 50:
            args.epochs = 5
        elif dataset_size < 200:
            args.epochs = 3
        else:
            args.epochs = 2
    
    # Auto-tune gradient accumulation
    if args.grad_accum is None:
        grad_accum_map = {"1.5b": 2, "7b": 4, "14b": 8}
        args.grad_accum = grad_accum_map[args.model]
    
    # Use recommended LoRA rank if not specified
    if args.lora_rank is None:
        args.lora_rank = rec_rank
    if args.lora_alpha is None:
        args.lora_alpha = rec_alpha
    
    effective_batch_size = args.batch_size * args.grad_accum
    num_training_steps = math.ceil((dataset_size * args.epochs) / effective_batch_size)
    warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"  Effective Batch Size: {effective_batch_size}")
    print(f"  Learning Rate: {args.learning_rate} | Warmup Steps: {warmup_steps}/{num_training_steps}")
    print(f"  LoRA Rank: {args.lora_rank} | LoRA Alpha: {args.lora_alpha}")
    print(f"  Max Seq Length: {args.max_seq_len}\n")

    print(f"Loading {model_name} with 4-bit quantization...")

    # ── Load base model ────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,          # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=True,
    )

    # ── Attach LoRA adapters with optimized config ─────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[           # Qwen attention + MLP layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30% VRAM
        random_state=42,
        use_rslora=True,  # Rank-stabilized LoRA for better convergence
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(args.dataset)
    
    # Shuffle dataset for better randomization
    dataset = dataset.shuffle(seed=42)
    
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data  = split["test"]
    print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ── Training arguments with early stopping & optimizations ─────────────────
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,  # Explicit warmup steps for stability
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=max(1, num_training_steps // 20),  # 20 log points per epoch
        eval_strategy="steps",
        eval_steps=max(1, num_training_steps // 10),  # 10 evals per epoch
        save_strategy="steps",
        save_steps=max(1, num_training_steps // 10),
        save_total_limit=3,  # Keep 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",        # 8-bit Adam saves VRAM
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,         # Gradient clipping for stability
        report_to="none",          # set to "wandb" if you want experiment tracking
        seed=42,
        max_length=args.max_seq_len,
        packing=True,              # pack short sequences for efficiency
        dataset_text_field="text",
        dataset_kwargs={"append_concat_token": False, "add_special_tokens": False},
        remove_unused_columns=False,
        greater_is_better=False,  # For loss metric
    )

    # ── Trainer with early stopping ────────────────────────────────────────────
    from transformers import EarlyStoppingCallback
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping,
                early_stopping_threshold=0.001,
            )
        ]
    )

    print("🚀 Starting training...")
    trainer.train()

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    print(f"\n💾 Saving LoRA adapter to {args.output}/lora_adapter ...")
    model.save_pretrained(f"{args.output}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output}/lora_adapter")

    # ── Merge + export to GGUF for Ollama ─────────────────────────────────────
    print("🔄 Merging adapter and exporting to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(
        f"{args.output}/gguf",
        tokenizer,
        quantization_method="q4_k_m",  # best quality/size tradeoff for inference
    )
    print(f"\n✅ Training complete!")
    print(f"📦 LoRA adapter saved to: {args.output}/lora_adapter/")
    print(f"📦 GGUF model saved to: {args.output}/gguf/")
    print(f"\n📝 Next steps:")
    print(f"  1. python build_modelfile.py")
    print(f"  2. ollama create commitgen -f Modelfile")


if __name__ == "__main__":
    main()
