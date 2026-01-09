#!/usr/bin/env python3
"""
Utility to display task assignments for each episode in a lerobot dataset.

Prints the episode_index and corresponding tasks (list of task names or indices)
for each episode in meta/episodes/*.parquet files.

Usage:
    python task_per_episode.py <root_dir>
"""

import argparse
import pathlib
import sys
import pyarrow.parquet as pq


def show_episode_tasks(root: pathlib.Path) -> None:
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.is_dir():
        print(f"Episodes directory not found: {episodes_dir}", file=sys.stderr)
        sys.exit(1)
    
    episode_files = sorted(episodes_dir.rglob("*.parquet"))
    if not episode_files:
        print(f"No episode files found in {episodes_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nEpisode tasks in {root.name}:")
    print("=" * 80)
    print(f"  [episode_index] tasks")
    
    all_episodes = []
    
    # Read all episode files and collect episodes
    for ep_file in episode_files:
        table = pq.read_table(ep_file)
        data = table.to_pydict()
        
        if "episode_index" in data and "tasks" in data:
            for i in range(len(data["episode_index"])):
                ep_idx = data["episode_index"][i]
                tasks = data["tasks"][i]
                all_episodes.append((ep_idx, tasks))
    
    # Sort by episode index and print
    all_episodes.sort(key=lambda x: x[0])
    
    for ep_idx, tasks in all_episodes:
        print(f"  [{ep_idx}]           {tasks}")
    
    print("=" * 80)
    print(f"Total episodes: {len(all_episodes)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display task assignments for each episode in a LeRobot dataset."
    )
    parser.add_argument("root", type=pathlib.Path, help="Root directory of the dataset")
    args = parser.parse_args()
    show_episode_tasks(args.root)
