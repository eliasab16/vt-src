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
if [ -n "$IDENTITY" ]; then
    RSYNC_SSH="ssh -p $PORT -i $IDENTITY"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_INF_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Uploading remote_inference to server ==="
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE_INF_DIR/" \
    "$USER_HOST:/root/remote_inference/"

echo ""
echo "=== Setting up HF token ==="
# Check if .env already exists on server
SSH_OPTS="-p $PORT"
if [ -n "$IDENTITY" ]; then
    SSH_OPTS="-p $PORT -i $IDENTITY"
fi
HAS_ENV=$(ssh $SSH_OPTS $USER_HOST "[ -f /root/.env ] && echo yes || echo no" 2>/dev/null)
if [ "$HAS_ENV" = "no" ]; then
    read -p "HF Token (required for first deploy): " HF_TOKEN
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
read -p "Policy path (HF repo, optional — press Enter to skip): " POLICY_PATH
read -p "Task description (optional): " TASK_DESC
read -p "Action dim (optional, e.g. 8): " ACTION_DIM
read -p "Camera names, comma-separated (optional, e.g. wrist,overhead): " CAMERA_NAMES

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
echo "5. From your MacBook, load the model via /setup (one-time, ~30-60s):"
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
echo "   curl -X POST http://$IP:$TCP_PORT/setup \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"policy_path\":\"$POLICY_PATH_JSON\",\"action_dim\":$ACTION_DIM_JSON,\"camera_names\":$CAM_JSON,\"task\":\"$TASK_JSON\",\"device\":\"cuda\"}'"
echo ""
echo "6. From your MacBook, run full inference benchmark (realistic payload):"
echo "   python vt_src/remote_inference/benchmarks/ws_inference_test.py ws://$IP:$TCP_PORT/ws"
echo "   Expected: steady-state total ~150-200ms (model ~80ms + overhead ~80ms)."
echo ""
