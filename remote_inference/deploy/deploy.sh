#!/bin/bash
# Usage: bash deploy.sh "<ssh-command>" <tcp-port-for-8080>
# Example: bash deploy.sh "ssh root@<SERVER_IP> -p <SSH_PORT> -i ~/.ssh/<KEY>" <TCP_PORT>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash deploy.sh \"<ssh-command>\" <tcp-port-for-8080>"
    echo "Example: bash deploy.sh \"ssh root@<SERVER_IP> -p <SSH_PORT> -i ~/.ssh/<KEY>\" <TCP_PORT>"
    exit 1
fi

TCP_PORT="$2"

SSH_CMD="$1"

# Extract user@host, port, and identity file from the SSH command
USER_HOST=$(echo "$SSH_CMD" | grep -oE '[^ ]+@[^ ]+')
PORT=$(echo "$SSH_CMD" | sed -n 's/.*-p \([0-9]*\).*/\1/p')
IDENTITY=$(echo "$SSH_CMD" | sed -n 's/.*-i \([^ ]*\).*/\1/p')

if [ -z "$PORT" ]; then
    PORT=22
fi

RSYNC_SSH="ssh -p $PORT"
SSH_OPTS="-p $PORT"
if [ -n "$IDENTITY" ]; then
    RSYNC_SSH="ssh -p $PORT -i $IDENTITY"
    SSH_OPTS="-p $PORT -i $IDENTITY"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_INF_DIR="$(dirname "$SCRIPT_DIR")"

# Load persisted values from .env.local (git-ignored) — skips re-prompting
ENV_FILE="$SCRIPT_DIR/.env.local"
if [ -f "$ENV_FILE" ]; then
    echo "=== Loading saved values from $ENV_FILE ==="
    set -a; source "$ENV_FILE"; set +a
fi

echo "=== Ensuring rsync is installed on server ==="
# Fresh pod images often ship without rsync; install it before the upload step.
ssh $SSH_OPTS $USER_HOST "command -v rsync >/dev/null 2>&1 || (apt-get update && apt-get install -y rsync)"

echo ""
echo "=== Uploading remote_inference to server ==="
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE_INF_DIR/" \
    "$USER_HOST:/root/remote_inference/"

echo ""
echo "=== Setting up HF token ==="
# Check if .env already exists on server
HAS_ENV=$(ssh $SSH_OPTS $USER_HOST "[ -f /root/.env ] && echo yes || echo no" 2>/dev/null)
if [ "$HAS_ENV" = "no" ]; then
    if [ -z "$HF_TOKEN" ]; then
        read -p "HF Token (required for first deploy): " HF_TOKEN
    else
        echo "Using HF_TOKEN from .env.local"
    fi
    if [ -n "$HF_TOKEN" ]; then
        ssh $SSH_OPTS $USER_HOST "echo 'HF_TOKEN=$HF_TOKEN' > /root/.env"
        echo "Token saved to server."
    else
        echo "Warning: No token provided. HF auth will be skipped."
    fi
else
    echo "Token already exists on server. Skipping."
fi

echo ""
echo "=== Optional policy config ==="
echo "Values are loaded from $ENV_FILE if present. Leave blank to use <PLACEHOLDERS>."
echo "Tip: save values to $ENV_FILE to skip these prompts on future runs."
echo ""
[ -z "$POLICY_PATH" ] && read -p "Policy path — HuggingFace repo ID or local dir: " POLICY_PATH || echo "POLICY_PATH=$POLICY_PATH (from .env.local)"
[ -z "$TASK_DESC" ] && read -p "Task description: " TASK_DESC || echo "TASK_DESC=$TASK_DESC (from .env.local)"
[ -z "$ACTION_DIM" ] && read -p "Action dim (e.g. 8): " ACTION_DIM || echo "ACTION_DIM=$ACTION_DIM (from .env.local)"
[ -z "$CAMERA_NAMES" ] && read -p "Camera names (comma-separated): " CAMERA_NAMES || echo "CAMERA_NAMES=$CAMERA_NAMES (from .env.local)"

echo ""
echo "=== Upload complete ==="
echo ""
echo "Now run these commands:"
echo ""
echo "1. SSH into the remote server:"
echo "   $SSH_CMD"
echo ""
echo "2. On the remote server, first-time setup (installs deps):"
echo "   bash /root/remote_inference/deploy/setup_env.sh"
echo ""
echo "3. On the remote server, start/restart server (skips install):"
echo "   bash /root/remote_inference/deploy/start_server.sh"
echo ""
IP=$(echo "$USER_HOST" | cut -d'@' -f2)
echo "4. From your MacBook, test WebSocket latency (echo-only, no inference):"
echo "   python vt_src/remote_inference/benchmarks/ws_latency_test.py ws://$IP:$TCP_PORT/ws"
echo "   Expected: p50 close to ping RTT (30-60ms). If >150ms, check region/proxy."
echo ""
echo "5. Model inference test (step 5.1 MUST run before 5.2):"
echo ""
echo "  5.1. From your MacBook, load the model via /setup (REQUIRED — ~30-60s first time):"
# Build setup payload from user input or placeholders
POLICY_PATH_JSON="${POLICY_PATH:-<POLICY_PATH>}"
TASK_JSON="${TASK_DESC:-<TASK_DESCRIPTION>}"
ACTION_DIM_JSON="${ACTION_DIM:-<ACTION_DIM>}"
if [ -n "$CAMERA_NAMES" ]; then
    # Convert comma-separated to JSON array
    CAM_JSON=$(echo "$CAMERA_NAMES" | sed 's/,/","/g')
    CAM_JSON="[\"$CAM_JSON\"]"
else
    CAM_JSON='["<CAM_1>","<CAM_2>"]'
fi
echo "    curl -X POST http://$IP:$TCP_PORT/setup \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"policy_path\":\"$POLICY_PATH_JSON\",\"action_dim\":$ACTION_DIM_JSON,\"camera_names\":$CAM_JSON,\"task\":\"$TASK_JSON\",\"device\":\"cuda\"}'"
echo "    (wait for status: ready before running 5.2)"
echo ""
echo "  5.2. From your MacBook, run full inference benchmark (realistic payload, model must be loaded):"
echo "    python vt_src/remote_inference/benchmarks/ws_inference_test.py ws://$IP:$TCP_PORT/ws"
echo "   Expected: steady-state total ~150-200ms (model ~80ms + overhead ~80ms)."
echo ""
