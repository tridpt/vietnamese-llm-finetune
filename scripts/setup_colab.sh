#!/bin/bash
# ============================================================
# Google Colab Pro Setup Script
# ============================================================
# Run this script first when starting a new Colab session:
#   !bash scripts/setup_colab.sh
# ============================================================

set -e  # Exit on error

echo "=============================================="
echo "  🚀 Vietnamese LLM Fine-tuning Setup"
echo "=============================================="

# 1. Check GPU
echo ""
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. Install dependencies
echo "📦 Installing dependencies..."
pip install -q -U pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed!"

# 3. Login to HuggingFace (optional — for model download/upload)
echo ""
echo "🔑 HuggingFace Authentication:"
echo "   Run this in a Colab cell if needed:"
echo "   >>> from huggingface_hub import notebook_login"
echo "   >>> notebook_login()"

# 4. Login to Weights & Biases (optional — for experiment tracking)
echo ""
echo "📈 Weights & Biases:"
echo "   Run this in a Colab cell if needed:"
echo "   >>> import wandb"
echo "   >>> wandb.login()"

# 5. Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch
import transformers
import peft
import trl
import bitsandbytes

print(f'  PyTorch:        {torch.__version__}')
print(f'  Transformers:   {transformers.__version__}')
print(f'  PEFT:           {peft.__version__}')
print(f'  TRL:            {trl.__version__}')
print(f'  BitsAndBytes:   {bitsandbytes.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=============================================="
echo "  ✅ Setup Complete!"
echo "  Next: Run training with:"
echo "    from src.train import main"
echo "    main('configs/training_config.yaml')"
echo "=============================================="
