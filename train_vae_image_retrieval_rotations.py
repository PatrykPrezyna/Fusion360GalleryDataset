"""
Variational Autoencoder for Image Retrieval with Rotation Invariance

This script trains a VAE on images from a folder, treating images with the same
base name (different rotations/views) as the same class. It uses a contrastive
loss to ensure that different views of the same object have similar embeddings.
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import argparse
import json
import re
import time
from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np


def extract_base_name(filename: str) -> str:
    """Extract base name from filename (removes view suffix).
    
    Example: '0022a8fa-05d2-11ec-b405-02c1fc826105_back_02.png' 
    -> '0022a8fa-05d2-11ec-b405-02c1fc826105'
    
    The filename pattern is: {base_id}_{view}_{number}.png
    or: {base_id}_{view1}_{view2}_{number}.png
    
    The base_id is a UUID-like string (contains hyphens) and is always the first
    part when split by underscore.
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) < 2:
        # No underscores, return as is
        return name
    
    # The base ID is the first part (UUID-like string with hyphens)
    # Everything after is view names and numbers
    base = parts[0]
    
    return base


class RotationAwareImageDataset(Dataset):
    """Dataset that loads images and groups them by base name (same class)."""
    
    def __init__(self, folder_path: str, image_size: int = 128, data_percentage: float = 1.0, random_seed: int = 42):
        self.folder_path = folder_path
        self.image_size = image_size
        
        # Find all PNG images
        all_images = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        if len(all_images) == 0:
            raise ValueError(f"No PNG images found in {folder_path}")
        
        # Sample a percentage of the data if specified
        total_images = len(all_images)
        if data_percentage < 1.0:
            np.random.seed(random_seed)
            num_samples = int(len(all_images) * data_percentage)
            indices = np.random.choice(len(all_images), size=num_samples, replace=False)
            all_images = [all_images[i] for i in sorted(indices)]
            print(f"Sampled {len(all_images)} images ({data_percentage*100:.1f}% of {total_images} total)")
        
        # Group images by base name
        self.base_to_images: Dict[str, List[str]] = defaultdict(list)
        self.image_paths = []
        self.image_to_base = {}
        
        for img_path in all_images:
            filename = os.path.basename(img_path)
            base_name = extract_base_name(filename)
            self.base_to_images[base_name].append(img_path)
            self.image_paths.append(img_path)
            self.image_to_base[img_path] = base_name
        
        print(f"Found {len(all_images)} images in {folder_path}")
        print(f"Grouped into {len(self.base_to_images)} unique classes (base names)")
        print(f"Average {len(all_images) / len(self.base_to_images):.1f} views per class")
        
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
        base_name = self.image_to_base[img_path]
        return x, img_path, base_name
    
    def get_same_class_images(self, base_name: str) -> List[str]:
        """Get all image paths for a given base name."""
        return self.base_to_images.get(base_name, [])


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


def contrastive_loss(mu1: torch.Tensor, mu2: torch.Tensor, same_class: bool, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss to pull same-class embeddings together and push different-class apart.
    
    Args:
        mu1, mu2: Embeddings (mean vectors) from the encoder
        same_class: Whether the two embeddings are from the same class
        margin: Margin for different-class pairs
    """
    # Normalize embeddings
    mu1_norm = mu1 / (mu1.norm(dim=1, keepdim=True) + 1e-8)
    mu2_norm = mu2 / (mu2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute distance (1 - cosine similarity)
    distance = 1 - (mu1_norm * mu2_norm).sum(dim=1)
    
    if same_class:
        # Pull same-class embeddings together
        loss = distance.pow(2).sum()
    else:
        # Push different-class embeddings apart (only if distance < margin)
        loss = torch.clamp(margin - distance, min=0).pow(2).sum()
    
    return loss


def train_vae(
    train_folder_path: str,
    test_folder_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 128,
    image_size: int = 128,
    beta: float = 1.0,
    contrastive_weight: float = 0.1,
    data_percentage: float = 1.0,
    random_seed: int = 42,
):
    """Train the VAE on images from the folder, using train/test folders separately.
    
    Images with the same base name are treated as the same class, and a contrastive
    loss encourages similar embeddings for different views of the same object.
    
    Args:
        data_percentage: Percentage of data to use (0.0 to 1.0). Applied to both train and test sets.
        random_seed: Random seed for reproducible data sampling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train and test datasets
    train_dataset = RotationAwareImageDataset(train_folder_path, image_size=image_size, 
                                               data_percentage=data_percentage, random_seed=random_seed)
    test_dataset = RotationAwareImageDataset(test_folder_path, image_size=image_size,
                                             data_percentage=data_percentage, random_seed=random_seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training VAE with {len(train_dataset)} images (train), {len(test_dataset)} images (test)...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Track losses per epoch
    train_losses = []
    test_losses = []
    train_recon_losses = []
    train_kld_losses = []
    train_contrastive_losses = []
    epoch_times = []
    
    # Start training timer
    training_start_time = time.time()
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        total_contrastive_loss = 0.0
        num_train_batches = 0
        
        # Training phase
        for batch_idx, (x, paths, base_names) in enumerate(train_loader):
            x = x.to(device)
            batch_size_actual = x.size(0)
            
            optimizer.zero_grad()
            
            # Standard VAE forward pass
            recon, mu, logvar = model(x)
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss_value = recon_loss + beta * kld
            
            # Contrastive loss: sample pairs from the batch
            contrastive_loss_value = torch.tensor(0.0, device=device)
            if batch_size_actual > 1:
                # Create pairs within the batch
                num_pairs = min(batch_size_actual, 16)  # Limit pairs to avoid too much computation
                indices = torch.randperm(batch_size_actual)[:num_pairs]
                
                mu_pairs = mu[indices]
                base_names_list = [base_names[i] for i in indices]
                
                # Create positive pairs (same class) and negative pairs (different class)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        same_class = (base_names_list[i] == base_names_list[j])
                        pair_loss = contrastive_loss(
                            mu_pairs[i:i+1], 
                            mu_pairs[j:j+1], 
                            same_class=same_class
                        )
                        contrastive_loss_value += pair_loss
            
            # Total loss
            total_loss = vae_loss_value + contrastive_weight * contrastive_loss_value
            
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld.item()
            total_contrastive_loss += contrastive_loss_value.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / len(train_dataset)
        avg_recon_loss = total_recon_loss / len(train_dataset)
        avg_kld_loss = total_kld_loss / len(train_dataset)
        avg_contrastive_loss = total_contrastive_loss / num_train_batches
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_recon_loss)
        train_kld_losses.append(avg_kld_loss)
        train_contrastive_losses.append(avg_contrastive_loss)
        
        # Evaluation on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar, beta=beta)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / max(len(test_dataset), 1)
        test_losses.append(avg_test_loss)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch}/{epochs} - Train loss: {avg_train_loss:.4f} "
              f"(recon: {avg_recon_loss:.4f}, kld: {avg_kld_loss:.4f}, contrastive: {avg_contrastive_loss:.4f}) "
              f"- Test loss: {avg_test_loss:.4f} - Time: {epoch_time:.2f}s")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    print("\n" + "="*50)
    print(f"Training completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    print("="*50 + "\n")
    
    # Save model
    model_path = os.path.join(output_dir, "vae_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'image_size': image_size,
    }, model_path)
    print(f"\nSaved VAE model to {model_path}")
    
    # Extract embeddings for all images (train + test)
    print("\nExtracting embeddings for all images...")
    model.eval()
    all_embeddings = []
    all_paths = []
    
    # Combine train and test datasets for embedding extraction
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    with torch.no_grad():
        for x, paths, _ in full_loader:
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

    # Save training configuration/info
    info = {
        "train_folder_path": train_folder_path,
        "test_folder_path": test_folder_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "latent_dim": latent_dim,
        "image_size": image_size,
        "beta": beta,
        "contrastive_weight": contrastive_weight,
        "data_percentage": data_percentage,
        "random_seed": random_seed,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "num_train_classes": len(train_dataset.base_to_images),
        "num_test_classes": len(test_dataset.base_to_images),
        "device": str(device),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_recon_losses": train_recon_losses,
        "train_kld_losses": train_kld_losses,
        "train_contrastive_losses": train_contrastive_losses,
        "total_training_time_seconds": total_training_time,
        "total_training_time_minutes": total_training_time / 60,
        "total_training_time_hours": total_training_time / 3600,
        "average_epoch_time_seconds": avg_epoch_time,
        "epoch_times_seconds": epoch_times,
    }
    info_path = os.path.join(output_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    
    # Save training time to a separate file for easy access
    timing_path = os.path.join(output_dir, "training_time.txt")
    with open(timing_path, "w") as f:
        f.write("Training Time Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        f.write(f"Total training time: {total_training_time/60:.2f} minutes\n")
        f.write(f"Total training time: {total_training_time/3600:.2f} hours\n")
        f.write(f"Average time per epoch: {avg_epoch_time:.2f} seconds\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}\n")
        f.write(f"Training end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
        f.write("\nPer-epoch times:\n")
        for i, epoch_time in enumerate(epoch_times, 1):
            f.write(f"  Epoch {i}: {epoch_time:.2f} seconds\n")
    
    print(f"Saved training time to {timing_path}")
    
    # Also save losses as a separate numpy file for easy loading
    losses_path = os.path.join(output_dir, "losses.npz")
    np.savez(
        losses_path, 
        train_losses=train_losses, 
        test_losses=test_losses,
        train_recon_losses=train_recon_losses,
        train_kld_losses=train_kld_losses,
        train_contrastive_losses=train_contrastive_losses,
    )

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved image paths to {paths_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved training info to {info_path}")
    
    return model, embeddings, all_paths


def load_model_and_embeddings(output_dir: str, device: str = "cpu"):
    """Load trained model and scan folder for all PNG images, extracting embeddings on-the-fly.
    
    This function scans the output_dir folder for all PNG images and extracts embeddings
    dynamically, instead of using a saved list of paths.
    
    Args:
        output_dir: Directory containing the trained model and images
        device: Device to run model on
    
    Returns:
        model: Loaded VAE model
        embeddings: Extracted embeddings for all images found in folder
        image_paths: List of paths to all PNG images found in folder
    """
    model_path = os.path.join(output_dir, "vae_model.pt")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    latent_dim = checkpoint['latent_dim']
    image_size = checkpoint['image_size']
    
    model = ConvVAE(
        latent_dim=latent_dim,
        image_size=image_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Scan folder for all PNG images
    print(f"Scanning folder for images: {output_dir}")
    image_paths = []
    
    # Scan for PNG files in the folder and subdirectories
    for root, dirs, files in os.walk(output_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                # Normalize path
                full_path = os.path.normpath(full_path)
                if full_path not in image_paths:
                    image_paths.append(full_path)
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"No PNG images found in {output_dir}")
    
    print(f"Found {len(image_paths)} images in folder")
    
    # Extract embeddings for all images found in folder
    print("Extracting embeddings for all images...")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    all_embeddings = []
    valid_image_paths = []  # Track paths for successfully loaded images
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img)
                    batch_images.append(x)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                mu, _ = model.encode(batch_tensor)
                all_embeddings.append(mu.cpu())
                valid_image_paths.extend(batch_valid_paths)
    
    if len(all_embeddings) == 0:
        raise ValueError("No valid images found or embeddings extracted")
    
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Extracted embeddings for {len(valid_image_paths)} images. Embedding shape: {embeddings.shape}")
    
    return model, embeddings, valid_image_paths


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
    query_path_norm = os.path.normpath(query_image_path)
    
    for val, idx in zip(topk_values, topk_indices):
        path = image_paths[idx.item()]
        path_norm = os.path.normpath(path)
        # Skip if it's the same image (using normalized paths for comparison)
        if path_norm == query_path_norm:
            continue
        results.append((path, val.item()))
        if len(results) >= top_k:
            break
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a VAE for image retrieval with rotation invariance"
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        default="output_data/14_views_10000_mechanical",
        help="Folder containing training PNG images"
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="output_data/14_views_10000_mechanical/test",
        help="Folder containing test PNG images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model and embeddings (default: same as --train-folder)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.1,
        help="Weight for contrastive loss (encourages same-class embeddings to be similar)",
    )
    parser.add_argument(
        "--data-percentage",
        type=float,
        default=1.0,
        help="Percentage of data to use for training and testing (0.0 to 1.0, default: 1.0 for 100%%)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible data sampling (default: 42)",
    )
    
    args = parser.parse_args()
    
    # If no explicit output directory is given, use the train folder
    output_dir = args.output_dir if args.output_dir is not None else args.train_folder
    
    # Train the VAE
    model, embeddings, image_paths = train_vae(
        train_folder_path=args.train_folder,
        test_folder_path=args.test_folder,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        beta=args.beta,
        contrastive_weight=args.contrastive_weight,
        data_percentage=args.data_percentage,
        random_seed=args.random_seed,
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"\nTo use the model for retrieval, you can:")
    print(f"1. Load the model: model, embeddings, paths = load_model_and_embeddings('{output_dir}')")
    print(f"2. Retrieve similar images: results = retrieve_similar_images('path/to/image.png', model, embeddings, paths)")


if __name__ == "__main__":
    main()
