"""
Evaluation Module
==================
Evaluate the fine-tuned model using perplexity and qualitative analysis.

Usage:
    from src.evaluate import evaluate_model
    results = evaluate_model(
        model_path="outputs/qlora-vi-qwen2.5-7b/final_model",
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        config_path="configs/training_config.yaml",
    )
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.inference import ChatInference


# ──────────────────────────────────────────────
#  Vietnamese Test Prompts
# ──────────────────────────────────────────────

VIETNAMESE_TEST_PROMPTS = [
    # General knowledge
    "Giải thích machine learning cho người mới bắt đầu.",
    "Sự khác nhau giữa AI, Machine Learning và Deep Learning là gì?",
    # Reasoning
    "Tại sao bầu trời có màu xanh?",
    "Nếu tôi có 3 quả táo và cho bạn 1 quả, tôi còn mấy quả?",
    # Creative
    "Viết một đoạn thơ ngắn về Hà Nội mùa thu.",
    "Kể cho tôi một câu chuyện ngắn về một con robot biết yêu.",
    # Coding
    "Viết hàm Python để kiểm tra một số có phải số nguyên tố không.",
    "Giải thích cách hoạt động của thuật toán Binary Search.",
    # Vietnamese culture
    "Kể tên 5 món ăn nổi tiếng của Việt Nam và mô tả ngắn.",
    "Tết Nguyên Đán có ý nghĩa gì đối với người Việt Nam?",
]


# ──────────────────────────────────────────────
#  Qualitative Evaluation
# ──────────────────────────────────────────────

def evaluate_qualitative(
    model_path: str,
    base_model_name: Optional[str] = None,
    config_path: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    output_file: Optional[str] = None,
) -> list[dict]:
    """
    Run qualitative evaluation by generating responses for test prompts.
    
    Args:
        model_path: Path to the fine-tuned model.
        base_model_name: Base model name (for adapter loading).
        config_path: Path to training config.
        prompts: Custom test prompts (defaults to VIETNAMESE_TEST_PROMPTS).
        output_file: Optional path to save results as JSON.
    
    Returns:
        List of dicts with 'prompt' and 'response' keys.
    """
    if prompts is None:
        prompts = VIETNAMESE_TEST_PROMPTS

    print("\n" + "=" * 60)
    print("  📊 Qualitative Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Number of prompts: {len(prompts)}")
    print("=" * 60 + "\n")

    # Load model
    chat = ChatInference(
        model_path=model_path,
        base_model_name=base_model_name,
        config_path=config_path,
    )

    # Generate responses
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"👤 User: {prompt}")
        response = chat.generate(prompt)
        print(f"🤖 AI:   {response[:300]}{'...' if len(response) > 300 else ''}")
        results.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        })

    # Save results
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

    # Print summary
    avg_length = sum(len(r["response"]) for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"  📊 Summary")
    print(f"  Total prompts:      {len(results)}")
    print(f"  Avg response length: {avg_length:.0f} chars")
    print(f"{'='*60}\n")

    return results


# ──────────────────────────────────────────────
#  Perplexity Evaluation
# ──────────────────────────────────────────────

def evaluate_perplexity(
    model_path: str,
    eval_texts: list[str],
    base_model_name: Optional[str] = None,
) -> float:
    """
    Calculate perplexity on a set of evaluation texts.
    Lower perplexity = better model.
    
    Args:
        model_path: Path to the fine-tuned model.
        eval_texts: List of texts to evaluate on.
        base_model_name: Base model name (for adapter loading).
    
    Returns:
        Average perplexity score.
    """
    chat = ChatInference(model_path=model_path, base_model_name=base_model_name)
    model = chat.model
    tokenizer = chat.tokenizer

    total_loss = 0.0
    total_tokens = 0

    print(f"\n📊 Calculating perplexity on {len(eval_texts)} samples...")

    for i, text in enumerate(eval_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        num_tokens = inputs["input_ids"].shape[1]
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_texts)}] Running loss: {total_loss/total_tokens:.4f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"\n✅ Perplexity: {perplexity:.2f}")
    print(f"   Avg Loss:   {avg_loss:.4f}")

    return perplexity


# ──────────────────────────────────────────────
#  Full Evaluation Pipeline
# ──────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    base_model_name: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: str = "outputs/evaluation",
) -> dict:
    """
    Run full evaluation pipeline (qualitative + summary).
    
    Args:
        model_path: Path to the fine-tuned model.
        base_model_name: Base model name.
        config_path: Training config path.
        output_dir: Directory to save evaluation results.
    
    Returns:
        Dictionary with evaluation results.
    """
    print("\n" + "=" * 60)
    print("  🧪 Full Model Evaluation")
    print("=" * 60)

    # Qualitative evaluation
    qual_results = evaluate_qualitative(
        model_path=model_path,
        base_model_name=base_model_name,
        config_path=config_path,
        output_file=f"{output_dir}/qualitative_results.json",
    )

    results = {
        "model_path": model_path,
        "base_model": base_model_name,
        "timestamp": datetime.now().isoformat(),
        "qualitative": qual_results,
        "num_prompts": len(qual_results),
        "avg_response_length": sum(len(r["response"]) for r in qual_results) / len(qual_results),
    }

    # Save full results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/full_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Full results saved to: {output_dir}/")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()
    evaluate_model(args.model, args.base_model, args.config, args.output_dir)
