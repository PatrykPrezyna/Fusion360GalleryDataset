import os
import argparse
from typing import Tuple

import numpy as np

from train_vae_step_geometry import (
    shape_vertices_from_step,
    vertices_to_fixed_cloud,
)


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute a simple symmetric Chamfer distance between two point clouds.

    pc1, pc2: (N, 3) and (M, 3) numpy arrays.
    Returns:
        d1_mean: mean nearest-neighbor distance from pc1 to pc2
        d2_mean: mean nearest-neighbor distance from pc2 to pc1
        d_sym:   symmetric mean distance (average of the above)
    """
    if pc1.ndim != 2 or pc2.ndim != 2 or pc1.shape[1] != 3 or pc2.shape[1] != 3:
        raise ValueError("Point clouds must have shape (N, 3) and (M, 3).")

    # Pairwise distances (N, M)
    diff = pc1[:, None, :] - pc2[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    d1 = dists.min(axis=1)  # nearest neighbor distance for each point in pc1
    d2 = dists.min(axis=0)  # nearest neighbor distance for each point in pc2

    d1_mean = float(d1.mean())
    d2_mean = float(d2.mean())
    d_sym = 0.5 * (d1_mean + d2_mean)
    return d1_mean, d2_mean, d_sym


def load_point_cloud_from_step(step_path: str, num_points: int = 512) -> np.ndarray:
    """
    Load a STEP file, mesh it, and convert the vertices into a fixed-size point cloud.
    """
    verts = shape_vertices_from_step(step_path)
    cloud = vertices_to_fixed_cloud(verts, num_points=num_points)
    return cloud


def resolve_step_path(path: str) -> str:
    """
    Convenience helper:
    - If path already exists, return it.
    - Otherwise, try to interpret it as a file inside 'output_data'.
    """
    if os.path.exists(path):
        return path

    candidate = os.path.join("output_data", path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"Could not find STEP file at '{path}' or '{candidate}'.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two STEP files by converting them to point clouds and "
            "computing a symmetric Chamfer distance."
        )
    )
    parser.add_argument(
        "step1",
        type=str,
        help="Path to the first STEP file (or filename inside 'output_data').",
    )
    parser.add_argument(
        "step2",
        type=str,
        help="Path to the second STEP file (or filename inside 'output_data').",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=512,
        help="Number of points in each sampled point cloud (default: 512).",
    )

    args = parser.parse_args()

    step_path1 = resolve_step_path(args.step1)
    step_path2 = resolve_step_path(args.step2)

    print(f"Loading STEP 1 from: {step_path1}")
    pc1 = load_point_cloud_from_step(step_path1, num_points=args.num_points)

    print(f"Loading STEP 2 from: {step_path2}")
    pc2 = load_point_cloud_from_step(step_path2, num_points=args.num_points)

    d1, d2, d_sym = chamfer_distance(pc1, pc2)

    print("\n=== Point Cloud Comparison ===")
    print(f"Number of points: {args.num_points}")
    print(f"Mean NN distance STEP1 -> STEP2: {d1:.6f}")
    print(f"Mean NN distance STEP2 -> STEP1: {d2:.6f}")
    print(f"Symmetric Chamfer distance:     {d_sym:.6f}")


if __name__ == "__main__":
    main()


