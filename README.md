# 🇻🇳 Vietnamese LLM Fine-tuning with QLoRA

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/PEFT-QLoRA-green" alt="PEFT">
  <img src="https://img.shields.io/badge/Platform-Google%20Colab%20Pro-F9AB00?logo=googlecolab" alt="Colab">
</p>

Fine-tune a large language model for **Vietnamese instruction-following** using **QLoRA (Quantized Low-Rank Adaptation)** on Google Colab Pro. This project demonstrates an end-to-end LLM training pipeline — from data preparation to model evaluation and deployment.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                      │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Dataset   │───▶│ Tokenize │───▶│ SFTTrainer       │  │
│  │ (Vietnamese│    │ & Format │    │ (QLoRA + 4-bit)  │  │
│  │ Instruct) │    │          │    │                  │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                           │             │
│                                           ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Push to   │◀───│ Evaluate │◀───│ LoRA Adapters    │  │
│  │ HF Hub    │    │ Model    │    │ (Saved Weights)  │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## ✨ Features

- 🔧 **QLoRA Fine-tuning** — 4-bit quantization enabling 7B model training on single GPU
- 📊 **Weights & Biases Integration** — Real-time training metrics tracking
- 🇻🇳 **Vietnamese-focused** — Optimized for Vietnamese instruction-following tasks
- 📈 **Comprehensive Evaluation** — Perplexity, BLEU, and qualitative analysis
- 🚀 **HuggingFace Hub Deployment** — One-click model publishing
- ⚡ **Colab Pro Optimized** — Configured for A100/V100 GPU

## 📁 Project Structure

```
vietnamese-llm-finetune/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── configs/
│   └── training_config.yaml     # Training hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Dataset loading & formatting
│   ├── train.py                 # Main training script (QLoRA)
│   ├── inference.py             # Model inference & generation
│   ├── evaluate.py              # Model evaluation metrics
│   └── utils.py                 # Utility functions
├── scripts/
│   └── setup_colab.sh           # Colab environment setup
├── data/                        # Local dataset cache
└── outputs/                     # Training outputs & checkpoints
```

## 🚀 Quick Start

### 1. Open in Google Colab

Upload this project to Google Drive or clone from GitHub:

```python
# In Colab cell
!git clone https://github.com/<your-username>/vietnamese-llm-finetune.git
%cd vietnamese-llm-finetune
!bash scripts/setup_colab.sh
```

### 2. Configure Training

Edit `configs/training_config.yaml` to customize:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  
lora:
  r: 64
  alpha: 128
  
training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
```

### 3. Run Training

```python
from src.train import main
main("configs/training_config.yaml")
```

### 4. Run Inference

```python
from src.inference import ChatInference

chat = ChatInference("outputs/final_model")
response = chat.generate("Giải thích machine learning cho người mới bắt đầu")
print(response)
```

## 🧪 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-7B-Instruct |
| Method | QLoRA (4-bit NF4) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size | 4 (+ gradient accumulation 4) |
| Max Seq Length | 2048 |
| Optimizer | AdamW (paged, 8-bit) |
| Dataset | Vietnamese Instruction Data |
| GPU | NVIDIA A100 40GB (Colab Pro) |

## 📊 Results

> *Results will be updated after training is complete.*

| Metric | Before Fine-tune | After Fine-tune |
|--------|-----------------|-----------------|
| Perplexity | — | — |
| Vietnamese Fluency | — | — |
| Instruction Following | — | — |

## 🛠️ Tech Stack

- **[Transformers](https://huggingface.co/docs/transformers)** — Model loading & tokenization
- **[PEFT](https://huggingface.co/docs/peft)** — Parameter-efficient fine-tuning (LoRA/QLoRA)
- **[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)** — 4-bit quantization
- **[TRL](https://huggingface.co/docs/trl)** — Supervised fine-tuning trainer
- **[Datasets](https://huggingface.co/docs/datasets)** — Dataset management
- **[W&B](https://wandb.ai)** — Experiment tracking

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the base model
- [HuggingFace](https://huggingface.co/) for the open-source ecosystem
- Vietnamese AI community for instruction datasets
