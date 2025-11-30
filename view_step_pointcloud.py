import numpy as np
import matplotlib.pyplot as plt

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopLoc import TopLoc_Location


def load_step_shape(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def shape_vertices_from_step(step_path: str, linear_deflection: float = 0.5) -> np.ndarray:
    shape = load_step_shape(step_path)

    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesh.Perform()

    vertices = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            # Access nodes directly from triangulation
            num_nodes = tri.NbNodes()
            for i in range(1, num_nodes + 1):
                pnt: gp_Pnt = tri.Node(i)
                vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
        exp.Next()

    if not vertices:
        raise RuntimeError(f"No mesh vertices extracted from STEP file: {step_path}")

    verts = np.array(vertices, dtype=np.float32)
    mean = verts.mean(axis=0, keepdims=True)
    std = verts.std(axis=0, keepdims=True) + 1e-6
    verts = (verts - mean) / std
    return verts


def vertices_to_fixed_cloud(verts: np.ndarray, num_points: int = 2048) -> np.ndarray:
    n = verts.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
        cloud = verts[idx]
    else:
        pad = np.zeros((num_points - n, 3), dtype=np.float32)
        cloud = np.concatenate([verts, pad], axis=0)
    return cloud


def main():
    step_path = "72701222-0600-11ec-872b-020dc2b44123.step"  # adjust if needed

    verts = shape_vertices_from_step(step_path)
    cloud = vertices_to_fixed_cloud(verts, num_points=512*512)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    ax.scatter(xs, ys, zs, s=1, c=zs, cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Point cloud from {step_path}")
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()