"""
Lightweight reconstruction orchestrator for drone imagery.

This module focuses on the core pipeline requested: ingest photo/video data,
run a photogrammetry reconstruction (using COLMAP if available), and present the
output as an interactive 3D view via Open3D. It keeps the surface area small so
it can be expanded later with additional measurement or QA steps.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List

import cv2
import open3d as o3d


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_frames(video_path: Path, frames_dir: Path, step: int = 12, limit: int | None = None) -> List[Path]:
    """
    Extract frames from a video to a directory.

    Args:
        video_path: Input video file.
        frames_dir: Destination directory for the extracted frames.
        step: Save every `step`-th frame to control overlap and runtime.
        limit: Optional max number of frames to extract.
    Returns:
        List of frame file paths (sorted by capture order).
    """
    _ensure_dir(frames_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: List[Path] = []
    index = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if index % step == 0:
            filename = frames_dir / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            frames.append(filename)
            saved += 1
            if limit and saved >= limit:
                break
        index += 1

    cap.release()
    if not frames:
        raise RuntimeError("No frames were extracted; check the input video and step size.")
    return frames


def collect_images(input_path: Path, workspace: Path) -> Path:
    """
    Normalize input into an image directory.

    If a directory of images is provided, it is returned unchanged. If a single
    video file is provided, frames are extracted into the workspace.
    """
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        frames_dir = workspace / "frames"
        extract_frames(input_path, frames_dir)
        return frames_dir
    if input_path.is_file():
        raise ValueError("Input must be a directory of images or a video file; single images are not supported yet.")
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _colmap_available() -> bool:
    return shutil.which("colmap") is not None


def run_colmap_pipeline(images_dir: Path, workspace: Path, quality: str = "medium") -> Path:
    """
    Run a minimal COLMAP pipeline (features -> matching -> mapper -> fusion).

    Args:
        images_dir: Directory containing input images.
        workspace: Working directory for COLMAP outputs.
        quality: Preset controlling dense stereo parameters.

    Returns:
        Path to the fused dense point cloud (PLY) for visualization.
    """
    if not _colmap_available():
        raise EnvironmentError("COLMAP binary not found in PATH. Please install COLMAP to run reconstruction.")

    _ensure_dir(workspace)
    db_path = workspace / "colmap.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
    ]
    match_cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(db_path),
    ]
    map_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]

    subprocess.run(feature_cmd, check=True)
    subprocess.run(match_cmd, check=True)
    subprocess.run(map_cmd, check=True)

    models = sorted(sparse_dir.glob("*/"))
    if not models:
        raise RuntimeError("No sparse reconstruction produced by COLMAP mapper.")
    model_dir = models[0]

    undistort_cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(images_dir),
        "--input_path",
        str(model_dir),
        "--output_path",
        str(dense_dir),
    ]
    subprocess.run(undistort_cmd, check=True)

    stereo_cmd = [
        "colmap",
        "patch_match_stereo",
        "--workspace_path",
        str(dense_dir),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true" if quality != "draft" else "false",
    ]
    subprocess.run(stereo_cmd, check=True)

    fusion_cmd = [
        "colmap",
        "stereo_fusion",
        "--workspace_path",
        str(dense_dir),
        "--workspace_format",
        "COLMAP",
        "--output_path",
        str(dense_dir / "fused.ply"),
    ]
    subprocess.run(fusion_cmd, check=True)
    return dense_dir / "fused.ply"


def visualize_point_cloud(ply_path: Path, point_size: float = 2.0) -> None:
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Reconstruction Viewer", width=1280, height=720)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


def build_interactive_model(input_path: Path, output_root: Path | None = None, quality: str = "medium", view: bool = False) -> Path:
    """
    End-to-end helper: prepare inputs, run reconstruction, and optionally open the viewer.

    Returns the path to the fused point cloud PLY file.
    """
    workspace = Path(output_root) if output_root else Path(tempfile.mkdtemp(prefix="recon_"))
    images_dir = collect_images(input_path, workspace)
    ply_path = run_colmap_pipeline(images_dir, workspace, quality=quality)
    if view:
        visualize_point_cloud(ply_path)
    return ply_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal drone reconstruction pipeline")
    parser.add_argument("input", type=Path, help="Folder with images or a video file")
    parser.add_argument("--output", type=Path, default=None, help="Workspace/output directory")
    parser.add_argument("--quality", choices=["draft", "medium", "high"], default="medium", help="Dense stereo quality preset")
    parser.add_argument("--view", action="store_true", help="Launch interactive Open3D viewer after fusion")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    ply_path = build_interactive_model(args.input, args.output, quality=args.quality, view=args.view)
    print(f"Fused point cloud ready: {ply_path}")
    if not args.view:
        print("Use --view to open the interactive renderer.")


if __name__ == "__main__":
    main()
