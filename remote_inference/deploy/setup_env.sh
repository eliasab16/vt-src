#!/bin/bash
set -e


echo "=== Installing dependencies ==="
pip install "lerobot[pi05]" fastapi uvicorn websockets
pip install transformers==5.3.0

echo "=== Validating imports ==="
python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('lerobot PI05: ok')"
python -c "from lerobot.policies.factory import make_pre_post_processors; print('lerobot factory: ok')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "from transformers import CONFIG_MAPPING; assert CONFIG_MAPPING['paligemma'] is not None; print('paligemma config: ok')"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

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
