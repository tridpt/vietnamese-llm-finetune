"""
Training Script — QLoRA Fine-tuning
=====================================
Main training pipeline for fine-tuning LLMs on Vietnamese
instruction data using QLoRA (4-bit quantization + LoRA).

Usage:
    # From Python
    from src.train import main
    main("configs/training_config.yaml")
    
    # From command line
    python -m src.train --config configs/training_config.yaml
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from src.data_preparation import load_and_prepare_dataset, preview_dataset
from src.utils import (
    load_config,
    print_gpu_info,
    print_model_info,
    print_training_summary,
    get_torch_dtype,
    ensure_dir,
)


# ──────────────────────────────────────────────
#  Model Loading (4-bit Quantized)
# ──────────────────────────────────────────────

def load_quantized_model(config: dict):
    """
    Load a pre-trained model with 4-bit quantization (BitsAndBytes).
    
    This drastically reduces memory usage (~4x), enabling 7B models
    to fit on a single A100/V100 GPU for training.
    
    Args:
        config: Full configuration dictionary.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    
    model_name = model_cfg["name"]
    torch_dtype = get_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    
    print(f"📥 Loading model: {model_name}")
    print(f"   ↳ Quantization: 4-bit ({quant_cfg.get('bnb_4bit_quant_type', 'nf4')})")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch_dtype,
        device_map="auto",
        attn_implementation=model_cfg.get("attn_implementation", None),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    
    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"  # Required for SFTTrainer
    
    print(f"✅ Model loaded successfully!")
    print(f"   ↳ Vocab size: {len(tokenizer):,}")
    print(f"   ↳ Model dtype: {torch_dtype}")
    
    return model, tokenizer


# ──────────────────────────────────────────────
#  LoRA Configuration
# ──────────────────────────────────────────────

def setup_lora(model, config: dict):
    """
    Apply LoRA (Low-Rank Adaptation) adapters to the model.
    
    LoRA freezes the original weights and injects small trainable
    matrices, reducing trainable parameters by ~97%.
    
    Args:
        model: The quantized base model.
        config: Full configuration dictionary.
    
    Returns:
        Model with LoRA adapters applied.
    """
    lora_cfg = config["lora"]
    
    print(f"🔧 Applying LoRA adapters...")
    print(f"   ↳ Rank: {lora_cfg['r']}, Alpha: {lora_cfg['lora_alpha']}")
    print(f"   ↳ Target modules: {lora_cfg['target_modules']}")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    print(f"✅ LoRA adapters applied!")
    return model, peft_config


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train(config: dict) -> None:
    """
    Execute the full training pipeline.
    
    Steps:
        1. Print GPU info
        2. Load model with 4-bit quantization
        3. Apply LoRA adapters
        4. Load and format dataset
        5. Train with SFTTrainer
        6. Save model and (optionally) push to HuggingFace Hub
    
    Args:
        config: Full configuration dictionary.
    """
    train_cfg = config["training"]
    data_cfg = config["dataset"]
    hub_cfg = config.get("hub", {})
    
    # Ensure output directory exists
    output_dir = train_cfg["output_dir"]
    ensure_dir(output_dir)
    
    # ── Step 1: GPU info ──
    print_gpu_info()
    print_training_summary(config)
    
    # ── Step 2: Load model ──
    model, tokenizer = load_quantized_model(config)
    
    # ── Step 3: Apply LoRA ──
    model, peft_config = setup_lora(model, config)
    print_model_info(model)
    
    # ── Step 4: Load dataset ──
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=data_cfg["name"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
        test_size=data_cfg.get("test_size", 0.05),
        seed=data_cfg.get("seed", 42),
    )
    preview_dataset(train_dataset, num_examples=2)
    
    # ── Step 5: Training arguments ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        group_by_length=train_cfg.get("group_by_length", True),
        report_to=train_cfg.get("report_to", "wandb"),
        seed=train_cfg.get("seed", 42),
        push_to_hub=hub_cfg.get("push_to_hub", False),
        hub_model_id=hub_cfg.get("hub_model_id"),
        hub_token=hub_cfg.get("hub_token"),
    )
    
    # ── Step 6: Initialize SFTTrainer ──
    print("🏋️ Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        max_seq_length=train_cfg.get("max_seq_length", 2048),
    )
    
    # ── Step 7: Train! ──
    print("\n" + "=" * 60)
    print("  🚀 Starting Training!")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # ── Step 8: Save final model ──
    final_model_dir = f"{output_dir}/final_model"
    print(f"\n💾 Saving final model to: {final_model_dir}")
    
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # ── Step 9: Push to Hub (optional) ──
    if hub_cfg.get("push_to_hub", False):
        print(f"\n📤 Pushing model to HuggingFace Hub: {hub_cfg.get('hub_model_id')}")
        trainer.push_to_hub()
    
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print(f"  📁 Model saved at: {final_model_dir}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
#  Entry Points
# ──────────────────────────────────────────────

def main(config_path: str = "configs/training_config.yaml") -> None:
    """
    Main entry point for the training pipeline.
    
    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()
    main(args.config)
