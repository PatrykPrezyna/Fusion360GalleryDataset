"""
Simple Variational Autoencoder for Image Retrieval

This script trains a VAE on images from a folder and provides functionality
to retrieve similar images based on their latent representations.
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import argparse
from typing import List, Tuple
from pathlib import Path

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np


class ImageFolderDataset(Dataset):
    """Dataset that loads all PNG images from a folder."""
    
    def __init__(self, folder_path: str, image_size: int = 128):
        self.folder_path = folder_path
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG images found in {folder_path}")
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # Converts to [0,1] range
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, img_path


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for images."""
    
    def __init__(self, latent_dim: int = 128, image_size: int = 128):
        super().__init__()
        
        # Calculate the size after encoder
        # 128 -> 64 -> 32 -> 16 -> 8
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size: 256 * 8 * 8 = 16384
        self.enc_out_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid(),  # Output in [0,1]
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image."""
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, 
             beta: float = 1.0) -> torch.Tensor:
    """VAE loss: reconstruction loss + KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld


def train_vae(
    folder_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 128,
    image_size: int = 128,
    beta: float = 1.0,
):
    """Train the VAE on images from the folder."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = ImageFolderDataset(folder_path, image_size=image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training VAE with {len(dataset)} images...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Average loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "vae_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'image_size': image_size,
    }, model_path)
    print(f"\nSaved VAE model to {model_path}")
    
    # Extract embeddings for all images
    print("\nExtracting embeddings for all images...")
    model.eval()
    all_embeddings = []
    all_paths = []
    
    with torch.no_grad():
        for x, paths in dataloader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            z = mu  # Use mean as deterministic embedding for retrieval
            all_embeddings.append(z.cpu())
            all_paths.extend(paths)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save embeddings and paths
    emb_path = os.path.join(output_dir, "embeddings.pt")
    paths_path = os.path.join(output_dir, "image_paths.txt")
    
    torch.save(embeddings, emb_path)
    with open(paths_path, 'w') as f:
        for path in all_paths:
            f.write(f"{path}\n")
    
    print(f"Saved embeddings to {emb_path}")
    print(f"Saved image paths to {paths_path}")
    print(f"Embedding shape: {embeddings.shape}")
    
    return model, embeddings, all_paths


def load_model_and_embeddings(output_dir: str, device: str = "cpu"):
    """Load trained model and embeddings."""
    model_path = os.path.join(output_dir, "vae_model.pt")
    emb_path = os.path.join(output_dir, "embeddings.pt")
    paths_path = os.path.join(output_dir, "image_paths.txt")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ConvVAE(
        latent_dim=checkpoint['latent_dim'],
        image_size=checkpoint['image_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load embeddings
    embeddings = torch.load(emb_path, map_location=device)
    
    # Load paths
    with open(paths_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    return model, embeddings, image_paths


def retrieve_similar_images(
    query_image_path: str,
    model: ConvVAE,
    embeddings: torch.Tensor,
    image_paths: List[str],
    top_k: int = 5,
    device: str = "cpu",
    image_size: int = 128,
) -> List[Tuple[str, float]]:
    """
    Retrieve the top_k most similar images to the query image.
    
    Returns list of (image_path, similarity_score) tuples.
    """
    # Load and preprocess query image
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    query_img = Image.open(query_image_path).convert("RGB")
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    
    # Encode query image
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(query_tensor)
        query_embedding = mu  # Use mean as embedding
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    query_norm = query_embedding / (query_embedding.norm() + 1e-8)
    
    # Compute cosine similarities
    similarities = (embeddings_norm @ query_norm.t()).squeeze(1)
    
    # Get top_k most similar (excluding the query itself if it's in the dataset)
    topk_values, topk_indices = torch.topk(similarities, k=min(top_k + 1, len(image_paths)))
    
    results = []
    for val, idx in zip(topk_values, topk_indices):
        path = image_paths[idx.item()]
        # Skip if it's the same image
        if path == query_image_path:
            continue
        results.append((path, val.item()))
        if len(results) >= top_k:
            break
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a VAE for image retrieval on images from a folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="output_data copy",
        help="Folder containing PNG images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vae_retrieval_output",
        help="Directory to save model and embeddings"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    
    args = parser.parse_args()
    
    # Train the VAE
    model, embeddings, image_paths = train_vae(
        folder_path=args.folder,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        beta=args.beta,
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"\nTo use the model for retrieval, you can:")
    print(f"1. Load the model: model, embeddings, paths = load_model_and_embeddings('{args.output_dir}')")
    print(f"2. Retrieve similar images: results = retrieve_similar_images('path/to/image.png', model, embeddings, paths)")


if __name__ == "__main__":
    main()

