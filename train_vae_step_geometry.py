import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt


def load_step_shape(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def shape_vertices_from_step(step_path: str, linear_deflection: float = 0.5) -> np.ndarray:
    """
    Mesh the STEP shape and collect surface vertices as a simple geometric descriptor.
    Returns an array of shape (N, 3).
    """
    shape = load_step_shape(step_path)

    # Mesh the shape
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesh.Perform()

    # Collect vertices from all faces
    vertices = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = None
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            nodes = tri.Nodes()
            for i in range(1, nodes.Length() + 1):
                pnt: gp_Pnt = nodes.Value(i)
                vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
        exp.Next()

    if not vertices:
        raise RuntimeError(f"No mesh vertices extracted from STEP file: {step_path}")

    verts = np.array(vertices, dtype=np.float32)

    # Normalize: center to mean and scale to unit variance (per component)
    mean = verts.mean(axis=0, keepdims=True)
    std = verts.std(axis=0, keepdims=True) + 1e-6
    verts = (verts - mean) / std
    return verts


def vertices_to_fixed_cloud(verts: np.ndarray, num_points: int = 512) -> np.ndarray:
    """
    Convert variable-length vertex set to a fixed-size point cloud.
    - If there are more than num_points vertices, randomly sample.
    - If fewer, pad with zeros.
    Returns (num_points, 3).
    """
    n = verts.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
        cloud = verts[idx]
    else:
        pad = np.zeros((num_points - n, 3), dtype=np.float32)
        cloud = np.concatenate([verts, pad], axis=0)
    return cloud


class AssemblyStepDataset(Dataset):
    """
    Dataset that uses STEP geometry instead of pre-rendered images.

    For each entry in picture_info.json, it finds a STEP file in the same folder
    as the PNG (e.g., assembly.step) and converts the geometry into a fixed-size
    point cloud tensor.
    """

    def __init__(self, json_path: str, num_points: int = 512):
        with open(json_path, "r") as f:
            self.entries = json.load(f)

        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.entries)

    def _find_step_file(self, png_path: str) -> str:
        folder = os.path.dirname(png_path)
        # Prefer a file named assembly.step if present
        candidate = os.path.join(folder, "assembly.step")
        if os.path.exists(candidate):
            return candidate
        # Fallback: first .step file in the folder
        for name in os.listdir(folder):
            if name.lower().endswith(".step"):
                return os.path.join(folder, name)
        raise FileNotFoundError(f"No STEP file found in folder: {folder}")

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        png_path = entry["file_path"]
        step_path = self._find_step_file(png_path)

        verts = shape_vertices_from_step(step_path)
        cloud = vertices_to_fixed_cloud(verts, num_points=self.num_points)

        # (num_points, 3) -> tensor
        x = torch.from_numpy(cloud)  # float32
        return x, entry["index"]


class PointCloudVAE(nn.Module):
    """
    Simple MLP-based VAE operating on flattened point clouds (num_points * 3).
    """

    def __init__(self, num_points: int = 512, latent_dim: int = 32):
        super().__init__()
        in_dim = num_points * 3

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, in_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_step_vae(
    json_path: str,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 32,
    num_points: int = 512,
):
    os.makedirs(out_dir, exist_ok=True)

    dataset = AssemblyStepDataset(json_path, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointCloudVAE(num_points=num_points, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, _ in dataloader:
            # x: (B, num_points, 3) -> flatten
            x = x.view(x.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"[STEP VAE] Epoch {epoch}/{epochs} - loss per sample: {avg_loss:.4f}")

    # Save model
    model_path = os.path.join(out_dir, "pointcloud_vae.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved STEP-based VAE model to {model_path}")

    # Extract embeddings for all assemblies
    model.eval()
    all_indices: List[int] = []
    all_latents: List[torch.Tensor] = []
    with torch.no_grad():
        for x, idxs in dataloader:
            x = x.view(x.size(0), -1).to(device)
            mu, logvar = model.encode(x)
            z = mu  # use mean as deterministic embedding
            all_indices.extend(idxs.tolist())
            all_latents.append(z.cpu())

    embeddings = torch.cat(all_latents, dim=0)
    emb_path = os.path.join(out_dir, "step_embeddings.pt")
    idx_path = os.path.join(out_dir, "step_indices.json")
    torch.save(embeddings, emb_path)
    with open(idx_path, "w") as f:
        json.dump(all_indices, f)
    print(f"Saved STEP embeddings to {emb_path}")
    print(f"Saved STEP index list to {idx_path}")


def load_step_embeddings(emb_path: str, idx_path: str):
    emb = torch.load(emb_path, map_location="cpu")
    with open(idx_path, "r") as f:
        idx_list = json.load(f)
    return emb, idx_list


def find_most_similar_step(
    target_index: int,
    emb: torch.Tensor,
    idx_list: List[int],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Return the top_k most similar assembly indices to the given target_index,
    using cosine similarity in the latent space (STEP-based VAE).
    """
    if target_index not in idx_list:
        raise ValueError(f"Index {target_index} not found in STEP embeddings.")

    idx_tensor = torch.tensor(idx_list)
    target_pos = (idx_tensor == target_index).nonzero(as_tuple=False).item()

    # Normalize embeddings for cosine similarity
    emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    target_vec = emb_norm[target_pos : target_pos + 1]  # (1, D)
    sims = (emb_norm @ target_vec.t()).squeeze(1)  # (N,)

    # Exclude the target itself
    sims[target_pos] = -1.0
    topk_vals, topk_idx = torch.topk(sims, k=min(top_k, emb.size(0) - 1))

    results: List[Tuple[int, float]] = []
    for pos, sim in zip(topk_idx.tolist(), topk_vals.tolist()):
        results.append((idx_list[pos], float(sim)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple VAE on STEP geometry (point clouds) for assemblies from picture_info.json and compute embeddings."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="picture_info.json",
        help="Path to picture_info.json listing assemblies.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("output_data", "step_vae_embeddings"),
        help="Directory to save model and embeddings.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=512)

    args = parser.parse_args()

    train_step_vae(
        json_path=args.json_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        num_points=args.num_points,
    )


if __name__ == "__main__":
    main()


