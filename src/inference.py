"""
Inference Module
=================
Load the fine-tuned model and generate responses.

Usage:
    from src.inference import ChatInference
    chat = ChatInference("outputs/qlora-vi-qwen2.5-7b/final_model")
    print(chat.generate("Xin chào!"))
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
from src.utils import load_config, get_torch_dtype


class ChatInference:
    """Inference wrapper for the fine-tuned Vietnamese LLM."""

    def __init__(self, model_path, base_model_name=None, config_path=None, load_in_4bit=True, device="auto"):
        self.model_path = model_path
        self.device = device
        self.gen_config = {}
        
        if config_path:
            config = load_config(config_path)
            self.gen_config = config.get("generation", {})
            if base_model_name is None:
                base_model_name = config.get("model", {}).get("name")

        self.gen_config.setdefault("max_new_tokens", 512)
        self.gen_config.setdefault("temperature", 0.7)
        self.gen_config.setdefault("top_p", 0.9)
        self.gen_config.setdefault("top_k", 50)
        self.gen_config.setdefault("repetition_penalty", 1.1)
        self.gen_config.setdefault("do_sample", True)

        self.system_prompt = (
            "Bạn là một trợ lý AI thông minh và hữu ích. "
            "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu bằng tiếng Việt."
        )

        self.model, self.tokenizer = self._load_model(model_path, base_model_name, load_in_4bit)
        print("✅ Model ready for inference!")

    def _load_model(self, model_path, base_model_name, load_in_4bit):
        """Load model — auto-detects adapter vs merged model."""
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )

        try:
            if base_model_name is None:
                raise ValueError("No base model specified")
            print(f"📥 Loading base model: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, quantization_config=bnb_config,
                torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
            )
            print(f"🔧 Loading LoRA adapters: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        except Exception:
            print(f"📥 Loading merged model: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config,
                torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

    def generate(self, prompt, system_prompt=None, **kwargs):
        """Generate a response for the given prompt."""
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_params = {**self.gen_config, **kwargs}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_params.get("max_new_tokens", 512),
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                top_k=gen_params.get("top_k", 50),
                repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                do_sample=gen_params.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()

    def interactive(self):
        """Start an interactive chat session. Type 'quit' to exit."""
        print("\n" + "=" * 60)
        print("  🤖 Vietnamese AI Assistant — Interactive Mode")
        print("  Commands: 'quit' to exit, 'clear' to reset")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("👤 Bạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Tạm biệt!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Tạm biệt!")
                break
            if user_input.lower() == "clear":
                print("🔄 Đã reset context.\n")
                continue
            print("🤖 AI: ", end="", flush=True)
            response = self.generate(user_input)
            print(response + "\n")

    def compare_responses(self, prompts):
        """Generate responses for multiple prompts (for evaluation)."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Generating...")
            results.append({"prompt": prompt, "response": self.generate(prompt)})
        return results


def merge_and_save(base_model_name, adapter_path, output_path, push_to_hub=False, hub_model_id=None):
    """Merge LoRA adapters into base model and save as standalone model."""
    print(f"📥 Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"🔧 Loading LoRA adapters: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("🔀 Merging adapters into base model...")
    model = model.merge_and_unload()
    print(f"💾 Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    if push_to_hub and hub_model_id:
        print(f"📤 Pushing to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
    print(f"✅ Merged model saved at: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    chat = ChatInference(args.model, args.base_model, args.config)
    if args.prompt:
        print(f"\n🤖 Response:\n{chat.generate(args.prompt)}")
    else:
        chat.interactive()
