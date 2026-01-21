#!/usr/bin/env python3
"""Print episode timeframes (start/end timestamps) for a LeRobot dataset.

Usage:
    python print_episode_timeframes.py /path/to/dataset
    python print_episode_timeframes.py eliasab16/xvlm_insert_wire_with_target_tape
"""

import sys
import json
from pathlib import Path
import pandas as pd

def get_dataset_path(dataset_arg: str) -> Path:
    """Resolve dataset path from argument (local path or HF-style name)."""
    path = Path(dataset_arg)
    if path.exists():
        return path
    
    # Try HuggingFace cache
    hf_cache = Path.home() / ".cache/huggingface/lerobot" / dataset_arg
    if hf_cache.exists():
        return hf_cache
    
    raise FileNotFoundError(f"Dataset not found: {dataset_arg}")


def print_episode_timeframes(dataset_path: Path):
    """Print timeframes for all episodes in the dataset."""
    
    meta_dir = dataset_path / "meta"
    episodes_dir = meta_dir / "episodes"
    
    # Load info.json for FPS
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    total_episodes = info.get("total_episodes", "?")
    
    print(f"Dataset: {dataset_path.name}")
    print(f"Total episodes: {total_episodes}")
    print(f"FPS: {fps}")
    print("=" * 60)
    print()
    
    # Read all episode parquet files using pyarrow
    import pyarrow.parquet as pq
    
    all_episodes = []
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            table = pq.read_table(parquet_file)
            data = table.to_pydict()
            for i in range(len(data.get("episode_index", []))):
                row = {k: v[i] for k, v in data.items()}
                all_episodes.append(row)
    
    if not all_episodes:
        print("No episode data found.")
        return
    
    # Hardcode to overhead_top camera (different cameras have different video files)
    video_key = "videos/observation.images.overhead_top"
    from_col = f"{video_key}/from_timestamp"
    to_col = f"{video_key}/to_timestamp"
    file_idx_col = f"{video_key}/file_index"
    chunk_idx_col = f"{video_key}/chunk_index"
    
    print(f"Episode | Video File          | Start    | End      | Duration | Frames | Cumulative")
    print("-" * 95)
    
    def format_time(seconds):
        """Convert seconds to mm:ss format."""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
    
    cumulative_frames = 0
    current_file = None
    for idx, row in enumerate(all_episodes):
        ep_idx = row.get("episode_index", idx)
        length = row.get("length", [0])
        if isinstance(length, list):
            length = length[0]
        
        file_idx = row.get(file_idx_col, 0)
        chunk_idx = row.get(chunk_idx_col, 0)
        video_file = f"chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
        
        # Reset cumulative frames when file changes
        if video_file != current_file:
            cumulative_frames = 0
            current_file = video_file
        
        cumulative_frames += length
        
        from_ts = row[from_col]
        to_ts = row[to_col]
        
        duration = to_ts - from_ts
        
        print(f"{ep_idx:7d} | {video_file:19s} | {format_time(from_ts)} | {format_time(to_ts)} | {format_time(duration)} | {length:6d} | {cumulative_frames:10d}")
    
    print()
    print("=" * 95)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_episode_timeframes.py <dataset_path>")
        print("Example: python print_episode_timeframes.py eliasab16/xvlm_insert_wire_with_target_tape")
        sys.exit(1)
    
    dataset_path = get_dataset_path(sys.argv[1])
    print_episode_timeframes(dataset_path)
