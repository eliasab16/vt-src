#!/usr/bin/env python3
"""
Utility to rename a task in a LeRobot dataset.

Updates the task name in `meta/tasks.parquet` (where the task name is stored as the
DataFrame index) and in each episode parquet file under `meta/episodes/` (the
`tasks` column contains a list of task names).

Usage:
    python update_tasks.py <root_dir> <task_index> <new_name>
"""

import argparse
import pathlib
import sys
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def get_old_task_name(tasks_path: pathlib.Path, task_index: int) -> str:
    """Get the current task name for a given task_index."""
    df = pd.read_parquet(tasks_path)
    if "task_index" not in df.columns:
        print(f"Error: 'task_index' column not found in {tasks_path}", file=sys.stderr)
        sys.exit(1)
    
    matching_rows = df[df["task_index"] == task_index]
    if len(matching_rows) == 0:
        print(f"Error: No task found with task_index={task_index}", file=sys.stderr)
        sys.exit(1)
    
    # The task name is the DataFrame index
    return matching_rows.index[0]


def replace_task_in_tasks(tasks_path: pathlib.Path, old_name: str, new_name: str) -> None:
    """Update task name in tasks.parquet by renaming the DataFrame index."""
    df = pd.read_parquet(tasks_path)
    if old_name in df.index:
        df = df.rename(index={old_name: new_name})
        df.to_parquet(tasks_path)
        print(f"Updated: {tasks_path}")
    else:
        print(f"Warning: task '{old_name}' not found in tasks.parquet index", file=sys.stderr)


def replace_task_in_episode(ep_path: pathlib.Path, old_name: str, new_name: str) -> None:
    """Update task names in episode parquet using PyArrow to preserve list types."""
    table = pq.read_table(ep_path)
    data = table.to_pydict()
    
    if "tasks" in data:
        data["tasks"] = [
            [new_name if t == old_name else t for t in task_list] if isinstance(task_list, list) else task_list
            for task_list in data["tasks"]
        ]
        
        new_table = pa.Table.from_pydict(data, schema=table.schema)
        pq.write_table(new_table, ep_path)
        print(f"Updated: {ep_path}")


def main(root: pathlib.Path, task_index: int, new_name: str) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        print(f"Tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)

    old_name = get_old_task_name(tasks_path, task_index)
    print(f"Found task_index {task_index}: '{old_name}' -> '{new_name}'")

    replace_task_in_tasks(tasks_path, old_name, new_name)

    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.is_dir():
        print(f"Episodes directory not found: {episodes_dir}", file=sys.stderr)
        sys.exit(1)

    for ep_path in episodes_dir.rglob("*.parquet"):
        replace_task_in_episode(ep_path, old_name, new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename a task in a LeRobot dataset.")
    parser.add_argument("root", type=pathlib.Path, help="Root directory of the dataset")
    parser.add_argument("task_index", type=int, help="Task index to update")
    parser.add_argument("new_name", help="New task name")
    args = parser.parse_args()
    main(args.root, args.task_index, args.new_name)