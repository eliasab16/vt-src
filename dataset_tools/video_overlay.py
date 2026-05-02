#!/usr/bin/env python3
"""Overlay tools for videos.

Two subcommands:
    frame  - Average the first frame of every clip into a single image.
    video  - Average frame n of every clip at every n into a single video.

Two sources:
    --dir <dir>                       Flat directory of video files. Each file is one clip.
    --dataset <root> [--camera <c>]   LeRobot v2.1 dataset. Each episode is one clip,
                                      seeked into its packed mp4 by metadata. With no
                                      --camera, runs every camera (output must be a dir).

Examples:
    python video_overlay.py frame --dir /path/to/videos -o initial.png
    python video_overlay.py frame --dataset ~/.cache/.../ds --camera overhead -o overhead_init.png
    python video_overlay.py video --dataset ~/.cache/.../ds -o overlays/
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class Clip:
    """One unit to overlay: a video file with a start frame and a length."""
    path: Path
    start_frame: int
    length: int  # number of frames available from start_frame
    label: str   # for logging


def find_videos(directory: Path) -> list[Path]:
    videos = sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {directory}")
    return videos


def clips_from_dir(directory: Path) -> list[Clip]:
    clips = []
    for v in find_videos(directory):
        cap = cv2.VideoCapture(str(v))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        clips.append(Clip(path=v, start_frame=0, length=n, label=v.name))
    return clips


def list_cameras(dataset_path: Path) -> list[str]:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    return [k for k in info.get("features", {}) if k.startswith("observation.images.")]


def clips_from_dataset(dataset_path: Path, camera: str) -> list[Clip]:
    """Read meta/episodes/*.parquet and return one Clip per episode for the given camera."""
    import pyarrow.parquet as pq

    feature = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    fps = info["fps"]

    if feature not in info.get("features", {}):
        raise ValueError(f"Camera {feature!r} not in dataset. Available: {list_cameras(dataset_path)}")

    video_key = f"videos/{feature}"
    from_col = f"{video_key}/from_timestamp"
    file_col = f"{video_key}/file_index"
    chunk_col = f"{video_key}/chunk_index"

    rows = []
    for chunk_dir in sorted((dataset_path / "meta" / "episodes").glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            data = pq.read_table(parquet_file).to_pydict()
            for i in range(len(data["episode_index"])):
                rows.append({k: v[i] for k, v in data.items()})

    rows.sort(key=lambda r: r["episode_index"])

    clips = []
    for r in rows:
        ep = r["episode_index"]
        chunk_idx = int(r[chunk_col])
        file_idx = int(r[file_col])
        from_ts = float(r[from_col])
        length = int(r["length"][0] if isinstance(r["length"], list) else r["length"])

        path = dataset_path / "videos" / feature / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Episode {ep}: missing {path}")
        start_frame = round(from_ts * fps)
        clips.append(Clip(path=path, start_frame=start_frame, length=length, label=f"ep{ep:03d}"))

    return clips


def open_seeked(clip: Clip) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(clip.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {clip.path}")
    if clip.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
    return cap


def read_one(clip: Clip) -> np.ndarray:
    cap = open_seeked(clip)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame at {clip.start_frame} of {clip.path}")
        return frame
    finally:
        cap.release()


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    if (w, h) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    for f in frames:
        acc += f
    acc /= len(frames)
    return np.clip(acc, 0, 255).astype(np.uint8)


def do_frame(clips: list[Clip], output: Path) -> None:
    print(f"Overlaying {len(clips)} clips → {output}")
    frames = []
    target_size = None
    for c in clips:
        f = read_one(c)
        if target_size is None:
            h, w = f.shape[:2]
            target_size = (w, h)
            print(f"  reference size: {w}x{h}")
        frames.append(resize_to(f, target_size))
        print(f"  + {c.label}")
    overlay = average_frames(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    print(f"  wrote {output}")


def do_video(clips: list[Clip], output: Path, fps: float | None) -> None:
    print(f"Overlaying {len(clips)} clips → {output}")
    caps = [open_seeked(c) for c in clips]
    try:
        widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
        heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
        target_w, target_h = widths[0], heights[0]
        n_frames = min(c.length for c in clips)
        out_fps = fps if fps is not None else caps[0].get(cv2.CAP_PROP_FPS)

        print(f"  reference size: {target_w}x{target_h}")
        print(f"  min length: {n_frames} frames @ {out_fps} fps")

        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, out_fps, (target_w, target_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {output}")

        try:
            for i in range(n_frames):
                frames = []
                for cap, clip in zip(caps, clips):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Frame {i} read failed for {clip.label} ({clip.path})")
                    frames.append(resize_to(frame, (target_w, target_h)))
                writer.write(average_frames(frames))
                if (i + 1) % 50 == 0 or i + 1 == n_frames:
                    print(f"    {i + 1}/{n_frames}")
        finally:
            writer.release()
    finally:
        for c in caps:
            c.release()
    print(f"  wrote {output}")


def resolve_targets(args, mode: str) -> list[tuple[list[Clip], Path]]:
    """Return list of (clips, output_path) pairs to process."""
    suffix = ".png" if mode == "frame" else ".mp4"

    if args.dir is not None:
        clips = clips_from_dir(args.dir)
        out = args.output or Path(f"overlay{suffix}")
        if out.is_dir() or str(out).endswith("/"):
            out = out / f"overlay{suffix}"
        return [(clips, out)]

    cameras = [args.camera] if args.camera else list_cameras(args.dataset)
    if not cameras:
        raise ValueError(f"No image cameras found in {args.dataset}/meta/info.json")

    out_arg = args.output
    if out_arg is None:
        out_arg = Path("overlays")
    treat_as_dir = len(cameras) > 1 or out_arg.is_dir() or str(out_arg).endswith("/")

    targets = []
    for cam in cameras:
        short = cam.split(".")[-1]
        clips = clips_from_dataset(args.dataset, cam)
        if treat_as_dir:
            out = out_arg / f"{short}{suffix}"
        else:
            out = out_arg
        targets.append((clips, out))
    return targets


def add_source_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="Flat directory of videos (one clip per file)")
    src.add_argument("--dataset", type=Path, help="LeRobot v2.1 dataset root")
    p.add_argument("--camera", type=str, default=None,
                   help="Camera key (e.g. 'overhead' or 'observation.images.overhead'). "
                        "Default with --dataset: every camera.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output file or directory. With multiple cameras, treated as directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_frame = sub.add_parser("frame", help="Overlay first frame of every clip as one image")
    add_source_args(p_frame)

    p_video = sub.add_parser("video", help="Overlay every frame across clips into one video")
    add_source_args(p_video)
    p_video.add_argument("--fps", type=float, default=None, help="Override output FPS")

    args = parser.parse_args()

    if args.dir is not None and args.camera is not None:
        print("--camera only applies to --dataset", file=sys.stderr)
        sys.exit(1)

    targets = resolve_targets(args, args.cmd)
    for clips, out in targets:
        if args.cmd == "frame":
            do_frame(clips, out)
        else:
            do_video(clips, out, args.fps)


if __name__ == "__main__":
    main()
