#!/bin/bash
set -e

echo "=== Checking Python version ==="
# lerobot >= 0.5.0 requires Python 3.12+. Older Python forces pip to resolve
# to pre-0.5.0 lerobot which lacks config fields like use_relative_actions
# (added in PR #2970), causing model loading to fail.
python -c "import sys; assert sys.version_info >= (3, 12), f'Python 3.12+ required (found {sys.version}). Recreate pod with a newer PyTorch template.'"
python --version

echo "=== Installing dependencies ==="
# Pinned versions known to work together with lerobot 0.5.1.
pip install \
    "lerobot[pi05]==0.5.1" \
    "transformers==5.3.0" \
    "huggingface-hub>=1.3.0,<2.0.0" \
    fastapi uvicorn websockets

echo "=== Validating imports ==="
# One Python process, not five — avoids re-importing torch/transformers 5x.
python - <<'PY'
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print("lerobot PI05: ok")
from lerobot.policies.factory import make_pre_post_processors
print("lerobot factory: ok")
import transformers
print(f"transformers: {transformers.__version__}")
from transformers import CONFIG_MAPPING
assert CONFIG_MAPPING["paligemma"] is not None
print("paligemma config: ok")
import torch
print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
PY

echo "=== HuggingFace auth ==="
if [ -f /root/.env ]; then
    source /root/.env
fi
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN"
    echo "HuggingFace auth: ok"
else
    echo "Warning: HF_TOKEN not set. Set it in /root/.env or run: HF_TOKEN=<token> bash setup_env.sh"
fi

echo "=== Setup complete. Start server with: ==="
echo "bash /root/remote_inference/deploy/start_server.sh"
