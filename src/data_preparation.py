"""
Data Preparation Module
========================
Handles loading, formatting, and splitting Vietnamese instruction datasets
for fine-tuning with the SFTTrainer.
"""

from datasets import load_dataset, Dataset
from typing import Optional


# ──────────────────────────────────────────────
#  Chat Template Formatting
# ──────────────────────────────────────────────

def format_instruction_alpaca(example: dict) -> dict:
    """
    Format a single example from Alpaca-style dataset into chat format.
    
    Alpaca format has: instruction, input (optional), output
    We convert it to a structured prompt for instruction-following.
    
    Args:
        example: Dictionary with 'instruction', 'input', 'output' keys.
    
    Returns:
        Dictionary with 'text' key containing the formatted conversation.
    """
    system_message = (
        "Bạn là một trợ lý AI thông minh và hữu ích. "
        "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu bằng tiếng Việt."
    )
    
    # Build the user message
    if example.get("input") and example["input"].strip():
        user_message = f"{example['instruction']}\n\nContext: {example['input']}"
    else:
        user_message = example["instruction"]
    
    assistant_message = example["output"]
    
    # Format as chat messages (compatible with most chat models)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    
    return {"messages": messages}


def format_instruction_chat(example: dict) -> dict:
    """
    Format a single example from a chat-style dataset.
    
    Expects: 'conversations' key with list of {"from": ..., "value": ...}
    
    Args:
        example: Dictionary with 'conversations' key.
    
    Returns:
        Dictionary with 'messages' key containing formatted messages.
    """
    role_mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
    }
    
    messages = []
    for turn in example["conversations"]:
        role = role_mapping.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    
    # Add system message if not present
    if not messages or messages[0]["role"] != "system":
        system_message = (
            "Bạn là một trợ lý AI thông minh và hữu ích. "
            "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu bằng tiếng Việt."
        )
        messages.insert(0, {"role": "system", "content": system_message})
    
    return {"messages": messages}


# ──────────────────────────────────────────────
#  Dataset Loading & Processing
# ──────────────────────────────────────────────

def load_and_prepare_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    test_size: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load a Vietnamese instruction dataset from HuggingFace Hub
    and prepare it for training.
    
    Supports two formats:
        - Alpaca format: instruction / input / output columns
        - Chat format: conversations column
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Which split to load (default: "train").
        max_samples: Maximum number of samples to use (None = all).
        test_size: Fraction of data to use for evaluation.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_dataset, eval_dataset) with 'messages' column.
    
    Example:
        >>> train_ds, eval_ds = load_and_prepare_dataset(
        ...     "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
        ...     max_samples=1000
        ... )
        >>> print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    """
    print(f"📂 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Limit samples if specified (useful for testing)
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
        print(f"   ↳ Limited to {len(dataset)} samples")
    
    # Detect format and apply appropriate formatter
    columns = dataset.column_names
    
    if "instruction" in columns and "output" in columns:
        print("   ↳ Detected Alpaca format (instruction/input/output)")
        dataset = dataset.map(
            format_instruction_alpaca,
            remove_columns=columns,
            desc="Formatting (Alpaca → messages)",
        )
    elif "conversations" in columns:
        print("   ↳ Detected Chat format (conversations)")
        dataset = dataset.map(
            format_instruction_chat,
            remove_columns=columns,
            desc="Formatting (Chat → messages)",
        )
    else:
        raise ValueError(
            f"Unknown dataset format. Expected columns with 'instruction'/'output' "
            f"or 'conversations', but got: {columns}"
        )
    
    # Split into train and eval
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"✅ Dataset ready!")
    print(f"   ↳ Train samples: {len(train_dataset):,}")
    print(f"   ↳ Eval samples:  {len(eval_dataset):,}")
    
    return train_dataset, eval_dataset


def preview_dataset(dataset: Dataset, num_examples: int = 3) -> None:
    """
    Print a few examples from the dataset for verification.
    
    Args:
        dataset: The formatted dataset to preview.
        num_examples: Number of examples to display.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset Preview ({num_examples} examples)")
    print(f"{'='*60}")
    
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        messages = example["messages"]
        
        print(f"\n--- Example {i+1} ---")
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:200]  # Truncate for display
            if len(msg["content"]) > 200:
                content += "..."
            print(f"  [{role}]: {content}")
    
    print(f"\n{'='*60}\n")


# ──────────────────────────────────────────────
#  Main (standalone testing)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with limited samples
    train_ds, eval_ds = load_and_prepare_dataset(
        dataset_name="5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
        max_samples=100,
        test_size=0.1,
    )
    preview_dataset(train_ds, num_examples=2)
