import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


video_dir = "replayed_videos/real_robot_raw"

# Each key is a subfolder under video_dir.
# The slice is applied to the timeline (frame indices) of each video in that folder,
# after resampling to --fps. For example, slice(0, 25) keeps the first 25 frames.
video_slices: dict[str, slice] = {
    "20260131-154231": slice(0, 20),
    "20260131-155232": slice(0, 20),
    "20260131-160601": slice(0, 20),
    "20260131-160920": slice(0, 20),
    "20260131-161314": slice(0, 20),
    "20260131-162050": slice(0, 20),
    "20260131-162448": slice(0, 20),
    "20260131-162738": slice(0, 20),
    "20260131-163114": slice(0, 20),
    "20260131-163436": slice(0, 20),
    "20260131-163710": slice(0, 20),
    "20260131-164008": slice(0, 20),
    "20260131-164246": slice(0, 20),
    "20260131-164535": slice(0, 20),
    "20260131-164816": slice(0, 20),
    "20260131-165125": slice(0, 20),
    "20260131-165351": slice(0, 20),
    "20260131-165703": slice(0, 20),
    "20260131-165952": slice(0, 20),
    "20260131-170249": slice(0, 20),
    "20260131-170529": slice(0, 20),
    "20260131-170904": slice(0, 20),
    # "20260131-160256": slice(0),  # F
}


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class EncodeOpts:
    fps: int
    crf: int
    preset: str
    loglevel: str


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN: command not executed")
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _parse_size(s: str) -> tuple[int, int]:
    if "x" not in s:
        raise ValueError(f"Invalid size: {s!r}, expected like 640x360")
    w_str, h_str = s.split("x", 1)
    return int(w_str), int(h_str)


def _list_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files, key=lambda p: p.name)


def _normalize_video_slice(sl: slice) -> slice:
    # Users sometimes write slice(0) to mean "from frame 0".
    # In Python, slice(0) == slice(None, 0, None) (empty range), so we treat it as "from 0".
    if sl.start is None and sl.stop == 0 and sl.step is None:
        sl = slice(0, None, None)

    if sl.step not in (None, 1):
        raise ValueError(
            f"slice step is not supported (got {sl.step}); use contiguous ranges"
        )
    if sl.start is not None and sl.start < 0:
        raise ValueError(f"slice start must be >= 0 (got {sl.start})")
    if sl.stop is not None and sl.stop < 0:
        raise ValueError(f"slice stop must be >= 0 (got {sl.stop})")
    return sl


def _trim_clause(sl: slice) -> str:
    sl = _normalize_video_slice(sl)
    parts: list[str] = []
    if sl.start is not None:
        parts.append(f"start_frame={sl.start}")
    if sl.stop is not None:
        parts.append(f"end_frame={sl.stop}")
    if not parts:
        return ""
    return "trim=" + ":".join(parts) + ","


def _build_ffmpeg_grid_command(
    *,
    folders: list[tuple[str, list[Path], slice]],
    out_path: Path,
    n: int,
    cell_size: tuple[int, int],
    cam_height: int,
    fit: str,
    border: int,
    pad_color: str,
    enc: EncodeOpts,
) -> list[str]:
    expected = n * n
    if len(folders) != expected:
        raise ValueError(f"Expected {expected} folders, got {len(folders)}")

    cell_w, cell_h = cell_size
    if border < 0:
        raise ValueError("border must be >= 0")
    if 2 * border >= cell_w or 2 * border >= cell_h:
        raise ValueError(
            f"border too large for cell-size {cell_w}x{cell_h}: border={border}"
        )

    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        enc.loglevel,
        "-y",
    ]

    # Flatten all input files.
    flat_inputs: list[Path] = []
    for _, files, _ in folders:
        flat_inputs.extend(files)
    for p in flat_inputs:
        cmd += ["-i", str(p)]

    # Single filtergraph:
    # 1) per input: fps/scale/setsar -> [vIDX]
    # 2) per folder: hstack -> [tF]
    # 3) per tile: scale+pad -> [cF]
    # 4) all tiles: xstack -> [v]
    filters: list[str] = []
    input_idx = 0
    tile_labels: list[str] = []

    for folder_idx, (_, files, frame_slice) in enumerate(folders):
        local_labels: list[str] = []
        for _ in files:
            trim = _trim_clause(frame_slice)
            filters.append(
                f"[{input_idx}:v]fps={enc.fps},{trim}setpts=PTS-STARTPTS,scale=-2:{cam_height}:flags=bicubic,setsar=1[v{input_idx}]"
            )
            local_labels.append(f"[v{input_idx}]")
            input_idx += 1

        if not local_labels:
            raise ValueError(f"Folder index {folder_idx} has no input videos")

        if len(local_labels) == 1:
            filters.append(f"{local_labels[0]}null[t{folder_idx}]")
        else:
            inputs = "".join(local_labels)
            filters.append(
                f"{inputs}hstack=inputs={len(local_labels)}:shortest=1[t{folder_idx}]"
            )

        if fit == "pad":
            inner_w = cell_w - 2 * border
            inner_h = cell_h - 2 * border
            tile_chain = (
                f"[t{folder_idx}]"
                f"scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease,"
                f"pad={inner_w}:{inner_h}:(ow-iw)/2:(oh-ih)/2:color={pad_color},"
            )
            if border > 0:
                tile_chain += (
                    f"pad={cell_w}:{cell_h}:{border}:{border}:color={pad_color},"
                )
            tile_chain += f"setsar=1[c{folder_idx}]"
            filters.append(tile_chain)
        elif fit == "crop":
            filters.append(
                "".join(
                    [
                        f"[t{folder_idx}]",
                        f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase,",
                        f"crop={cell_w}:{cell_h},",
                        "setsar=1",
                        f"[c{folder_idx}]",
                    ]
                )
            )
        elif fit == "stretch":
            filters.append(
                "".join(
                    [
                        f"[t{folder_idx}]",
                        f"scale={cell_w}:{cell_h},",
                        "setsar=1",
                        f"[c{folder_idx}]",
                    ]
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown fit mode: {fit!r}")
        tile_labels.append(f"[c{folder_idx}]")

    layout = "|".join(
        f"{(i % n) * cell_w}_{(i // n) * cell_h}" for i in range(expected)
    )
    stacked_inputs = "".join(tile_labels)
    filters.append(
        f"{stacked_inputs}xstack=inputs={expected}:layout={layout}:shortest=1,format=yuv420p[v]"
    )

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(enc.crf),
        "-preset",
        enc.preset,
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return cmd


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate videos: each folder's videos are hstacked (horizontal), then the first N*N folders are "
            "arranged into an N x N grid."
        )
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Grid size N (uses first N*N folders)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="concat_grid.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--cell-size",
        type=str,
        default="640x360",
        help="Grid cell size, e.g. 640x360",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=360,
        help="Height used when horizontally stacking videos within a folder",
    )
    parser.add_argument(
        "--fit",
        type=str,
        choices=["pad", "crop", "stretch"],
        default="pad",
        help=(
            "How to fit each tile into a grid cell: "
            "pad=keep aspect ratio and pad (letterbox), "
            "crop=keep aspect ratio and center-crop (no black bars), "
            "stretch=scale to cell (may distort)"
        ),
    )
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help=(
            "Extra border in pixels around each tile (pad mode only). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--pad-color",
        type=str,
        default="black",
        help=(
            "Padding/border color for pad mode, e.g. black, white, #RRGGBB. "
            "Ignored for crop/stretch."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", type=str, default="veryfast")
    parser.add_argument("--loglevel", type=str, default="error")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print ffmpeg commands without running"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _ensure_ffmpeg()

    keys = list(video_slices.keys())
    if not keys:
        raise RuntimeError("video_slices is empty")

    if args.n is None:
        args.n = math.isqrt(len(keys))
    if args.n <= 0:
        raise ValueError("N must be >= 1")

    need = args.n * args.n
    selected = keys[:need]
    if len(selected) < need:
        raise ValueError(
            f"Need at least {need} folders for N={args.n}, but only {len(keys)} entries in video_slices"
        )

    base = Path(__file__).resolve().parent
    root = (base / video_dir).resolve()
    out_path = (base / args.out).resolve()
    cell_size = _parse_size(args.cell_size)
    enc = EncodeOpts(
        fps=args.fps, crf=args.crf, preset=args.preset, loglevel=args.loglevel
    )

    print(f"video_dir: {root}")
    print(f"Selected folders: {len(selected)} (N={args.n})")
    print(f"Output: {out_path}")

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    folders: list[tuple[str, list[Path], slice]] = []
    for k in selected:
        folder = root / k
        files = _list_videos(folder)
        if not files:
            raise FileNotFoundError(f"No videos found in {folder}")
        folders.append((k, files, video_slices[k]))

    cmd = _build_ffmpeg_grid_command(
        folders=folders,
        out_path=out_path,
        n=args.n,
        cell_size=cell_size,
        cam_height=args.cam_height,
        fit=args.fit,
        border=args.border,
        pad_color=args.pad_color,
        enc=enc,
    )
    _run(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
