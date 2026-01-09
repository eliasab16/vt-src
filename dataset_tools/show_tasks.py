#!/usr/bin/env python3
"""
Display all tasks in a lerobot dataset.

Prints the task_index and task name for each task in meta/tasks.parquet.

Usage:
    python show_tasks.py <dataset_root_dir>
"""

import argparse
import pathlib
import sys
import pandas as pd


def show_tasks(root: pathlib.Path) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        print(f"Tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_parquet(tasks_path)
    
    if "task_index" not in df.columns:
        print(f"Error: 'task_index' column not found in {tasks_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nTasks in {root.name}:")
    print("=" * 80)
    
    print(f"  [task_idx] <task_name>")
    for task_name in df.index:
        task_idx = df.loc[task_name, "task_index"]
        print(f"  [{task_idx}]        {task_name}")
    
    print("=" * 80)
    print(f"Total tasks: {len(df)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display all tasks in a LeRobot dataset.")
    parser.add_argument("root", type=pathlib.Path, help="Root directory of the dataset")
    args = parser.parse_args()
    show_tasks(args.root)
