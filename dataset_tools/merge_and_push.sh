#!/bin/bash

# Hardcoded list of source datasets
SOURCE_REPO_IDS="['eliasab16/xvlm_pick_up_wire_yellow_1_annotated', 'eliasab16/xvlm_pick_up_wire_yellow_2_annotated', 'eliasab16/xvlm_pick_up_wire_red_1_annotated', 'eliasab16/xvlm_pick_up_wire_red_2_annotated', 'eliasab16/xvlm_pick_up_wire_white_1_annotated', 'eliasab16/xvlm_pick_up_wire_white_2_annotated', 'eliasab16/xvlm_pick_up_wire_green_1_annotated', 'eliasab16/xvlm_pick_up_wire_green_2_annotated', 'eliasab16/xvlm_insert_wire_with_target_tape_v2_fixed_segmented']"

# Destination repository ID
OUTPUT_REPO_ID="eliasab16/xvla_merged_pick_up_insert_wire_v1"

echo "Merging datasets into ${OUTPUT_REPO_ID}..."

lerobot-edit-dataset \
    --repo_id "${OUTPUT_REPO_ID}" \
    --operation.type merge \
    --operation.repo_ids "${SOURCE_REPO_IDS}" \
    --push_to_hub true

if [ $? -eq 0 ]; then
    echo "✓ Successfully merged and pushed to ${OUTPUT_REPO_ID}"
else
    echo "✗ Merge failed with exit code: $?"
    exit 1
fi
