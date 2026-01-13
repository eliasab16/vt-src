#!/bin/bash
# Parallel batch processing of wire detection videos
# Runs 3 jobs in parallel for optimal CoreML performance

SCRIPT="/Users/elisd/Desktop/vult/vt-src/perception/multi_stage/benchmark_pipeline_coreml_color.py"
BASE="/Users/elisd/.cache/huggingface/lerobot/eliasab16"

# Shortcuts
GREEN="$BASE/xvlm_pick_up_wire_green/videos"
RED="$BASE/xvlm_pick_up_wire_red_clean/videos"
WHITE="$BASE/xvlm_pick_up_wire_white/videos"
YELLOW="$BASE/xvlm_pick_up_wire_yellow/videos"

WT="observation.images.wrist_top/chunk-000"
WB="observation.images.wrist_bottom/chunk-000"
OH="observation.images.overhead_top/chunk-000"

# Function to process a video
process() {
    local input="$1"
    local color="$2"
    local padding="$3"
    local output="${input%.mp4}_annotated.mp4"
    echo "Processing: $(basename $(dirname $(dirname $input)))/$color (padding: $padding)"
    python "$SCRIPT" --input "$input" --output "$output" --target-colors "$color" --no-text --padding "$padding"
}

export -f process
export SCRIPT

echo "Starting parallel batch processing (3 jobs)..."
echo ""

# Generate commands and run 3 in parallel
cat << EOF | xargs -P 3 -I {} bash -c '{}'
process "$GREEN/$WT/file-000.mp4" green 15
process "$GREEN/$WB/file-000.mp4" green 15
process "$GREEN/$OH/file-000.mp4" green 10
process "$RED/$WT/file-000.mp4" red 15
process "$RED/$WB/file-000.mp4" red 15
process "$RED/$OH/file-000.mp4" red 10
process "$WHITE/$WT/file-000.mp4" white 15
process "$WHITE/$WB/file-000.mp4" white 15
process "$WHITE/$OH/file-000.mp4" white 10
process "$YELLOW/$WT/file-000.mp4" yellow 15
process "$YELLOW/$WB/file-000.mp4" yellow 15
process "$YELLOW/$OH/file-000.mp4" yellow 10
EOF

echo ""
echo "=========================================="
echo "Batch processing complete!"
echo "=========================================="
