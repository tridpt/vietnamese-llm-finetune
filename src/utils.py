"""
Utility Functions
==================
Shared utilities for configuration loading, model info, and logging.
"""

import yaml
import torch
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load training configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Dictionary containing all configuration parameters.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"⚙️  Loaded config from: {config_path}")
    return config


def print_gpu_info() -> None:
    """Print information about available GPU(s)."""
    print("\n" + "=" * 50)
    print("  GPU Information")
    print("=" * 50)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  CUDA Available: ✅")
        print(f"  Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            
            print(f"\n  GPU {i}: {name}")
            print(f"    Total Memory:     {memory_total:.1f} GB")
            print(f"    Reserved Memory:  {memory_reserved:.1f} GB")
            print(f"    Allocated Memory: {memory_allocated:.1f} GB")
            print(f"    Free Memory:      {memory_total - memory_reserved:.1f} GB")
    else:
        print("  CUDA Available: ❌")
        print("  ⚠️  No GPU detected. Training will be very slow on CPU.")
    
    if torch.backends.mps.is_available():
        print("  MPS (Apple Silicon): ✅")
    
    print("=" * 50 + "\n")


def print_model_info(model) -> None:
    """
    Print model architecture summary with trainable parameters.
    
    Args:
        model: The loaded model (with or without LoRA adapters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100
    
    print("\n" + "=" * 50)
    print("  Model Parameters")
    print("=" * 50)
    print(f"  Total Parameters:     {total_params:>14,}")
    print(f"  Trainable Parameters: {trainable_params:>14,}  ({trainable_pct:.2f}%)")
    print(f"  Frozen Parameters:    {frozen_params:>14,}  ({100 - trainable_pct:.2f}%)")
    print(f"  Model Size (approx):  {total_params * 2 / (1024**3):>10.1f} GB (FP16)")
    print("=" * 50 + "\n")


def print_training_summary(config: dict) -> None:
    """
    Print a formatted summary of training configuration.
    
    Args:
        config: The loaded configuration dictionary.
    """
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("dataset", {})
    
    # Calculate effective batch size
    batch_size = train_cfg.get("per_device_train_batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    effective_batch = batch_size * grad_accum
    
    print("\n" + "=" * 60)
    print("  Training Configuration Summary")
    print("=" * 60)
    print(f"  Model:              {model_cfg.get('name', 'N/A')}")
    print(f"  Dataset:            {data_cfg.get('name', 'N/A')}")
    print(f"  LoRA Rank:          {lora_cfg.get('r', 'N/A')}")
    print(f"  LoRA Alpha:         {lora_cfg.get('lora_alpha', 'N/A')}")
    print(f"  Learning Rate:      {train_cfg.get('learning_rate', 'N/A')}")
    print(f"  Epochs:             {train_cfg.get('num_train_epochs', 'N/A')}")
    print(f"  Batch Size:         {batch_size} (effective: {effective_batch})")
    print(f"  Max Seq Length:     {train_cfg.get('max_seq_length', 'N/A')}")
    print(f"  Optimizer:          {train_cfg.get('optim', 'N/A')}")
    print(f"  Scheduler:          {train_cfg.get('lr_scheduler_type', 'N/A')}")
    print(f"  Output Dir:         {train_cfg.get('output_dir', 'N/A')}")
    print("=" * 60 + "\n")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch.dtype.
    
    Args:
        dtype_str: String representation (e.g., "bfloat16", "float16").
    
    Returns:
        Corresponding torch.dtype.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Choose from: {list(dtype_map.keys())}"
        )
    
    return dtype_map[dtype_str]


def ensure_dir(path: str) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    
    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
