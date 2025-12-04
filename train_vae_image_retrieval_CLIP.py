"""
Simple Variational Autoencoder for Image Retrieval

This script trains a VAE on images from a folder and provides functionality
to retrieve similar images based on their latent representations.

The model is designed to be orientation-invariant through:
1. Data augmentation during training: random rotations (0-360°) and flips
2. Test-time augmentation during retrieval: query images are tested with
   multiple orientations and embeddings are averaged for more robust matching
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import argparse
import json
import shutil
from typing import List, Tuple
from pathlib import Path

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import numpy as np


class ImageFolderDataset(Dataset):
    """Dataset that loads all PNG images from a folder."""
    
    def __init__(self, folder_path: str, image_size: int = 128, augment: bool = False):
        self.folder_path = folder_path
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG images found in {folder_path}")
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
        
        # Base transforms
        base_transforms = [T.Resize((image_size, image_size))]
        
        # Add augmentation for training
        if augment:
            base_transforms.extend([
                T.RandomRotation(degrees=360),  # Full rotation (0-360 degrees)
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])
        
        base_transforms.append(T.ToTensor())  # Converts to [0,1] range
        self.transform = T.Compose(base_transforms)
    
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
    train_split: float = 0.8,
    use_augmentation: bool = True,
):
    """Train the VAE on images from the folder with a train/test split.

    After training and embedding extraction, test images are moved to a
    `test` subfolder under `folder_path`, and an `info.json` file with
    training parameters is written to `output_dir`.
    
    By default, data augmentation (rotations and flips) is applied during
    training to improve orientation invariance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets - training set with optional augmentation, full dataset without for embedding extraction
    train_base_dataset = ImageFolderDataset(folder_path, image_size=image_size, augment=use_augmentation)
    full_dataset = ImageFolderDataset(folder_path, image_size=image_size, augment=False)
    
    # Split into train and test subsets
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(train_base_dataset, [train_size, test_size])
    
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
    
    print(f"Training VAE with {train_size} images (train), {test_size} images (test)...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        # Training phase
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / train_size
        
        # Evaluation on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar, beta=beta)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / max(test_size, 1)
        print(f"Epoch {epoch}/{epochs} - Train loss: {avg_train_loss:.4f} - Test loss: {avg_test_loss:.4f}")
    
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
    
    # Use a full-dataset loader (no shuffling) for embeddings - use non-augmented dataset
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    with torch.no_grad():
        for x, paths in full_loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            z = mu  # Use mean as deterministic embedding for retrieval
            all_embeddings.append(z.cpu())
            all_paths.extend(paths)
    
    embeddings = torch.cat(all_embeddings, dim=0)

    # Move test images to a dedicated subfolder for future runs/organization
    test_indices_set = set(getattr(test_dataset, "indices", []))
    test_dir = os.path.join(folder_path, "test")
    if len(test_indices_set) > 0:
        os.makedirs(test_dir, exist_ok=True)
        for idx in test_indices_set:
            try:
                src = full_dataset.image_paths[idx]
            except IndexError:
                # Should not happen, but guard anyway
                continue
            filename = os.path.basename(src)
            dst = os.path.join(test_dir, filename)
            # Avoid moving if already in the right place
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"Warning: could not move {src} to {dst}: {e}")

        # Update paths to reflect new test locations
        for i, old_path in enumerate(all_paths):
            if i in test_indices_set:
                filename = os.path.basename(old_path)
                all_paths[i] = os.path.join(test_dir, filename)

    # Save embeddings and paths
    emb_path = os.path.join(output_dir, "embeddings.pt")
    paths_path = os.path.join(output_dir, "image_paths.txt")
    
    torch.save(embeddings, emb_path)
    with open(paths_path, 'w') as f:
        for path in all_paths:
            f.write(f"{path}\n")

    # Save training configuration/info
    info = {
        "folder_path": folder_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "latent_dim": latent_dim,
        "image_size": image_size,
        "beta": beta,
        "train_split": train_split,
        "train_size": train_size,
        "test_size": test_size,
        "device": str(device),
        "test_subfolder": test_dir if len(test_indices_set) > 0 else None,
        "use_augmentation": use_augmentation,
    }
    info_path = os.path.join(output_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved image paths to {paths_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved training info to {info_path}")
    
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
    use_tta: bool = True,
) -> List[Tuple[str, float]]:
    """
    Retrieve the top_k most similar images to the query image.
    
    Uses test-time augmentation (TTA) by default to handle different orientations:
    - Original image
    - 90, 180, 270 degree rotations
    - Horizontal and vertical flips
    - Combinations of rotations and flips
    
    Returns list of (image_path, similarity_score) tuples.
    """
    # Load query image
    query_img = Image.open(query_image_path).convert("RGB")
    
    # Base transform
    base_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    # Test-time augmentation: create multiple views of the query image
    query_embeddings = []
    
    if use_tta:
        # Generate multiple augmented views
        augmentations = [
            lambda img: img,  # Original
            lambda img: img.rotate(90),
            lambda img: img.rotate(180),
            lambda img: img.rotate(270),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal flip
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),  # Vertical flip
            lambda img: img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM),
            lambda img: img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.rotate(270).transpose(Image.FLIP_LEFT_RIGHT),
        ]
        
        for aug_fn in augmentations:
            aug_img = aug_fn(query_img)
            query_tensor = base_transform(aug_img).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(query_tensor)
                query_embeddings.append(mu)
        
        # Average embeddings from all augmentations
        query_embedding = torch.stack(query_embeddings).mean(dim=0)
    else:
        # No augmentation - use original image only
        query_tensor = base_transform(query_img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(query_tensor)
            query_embedding = mu
    
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
        default=None,
        help="Directory to save model and embeddings (default: same as --folder)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data used for training (rest is test)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation during training (not recommended)",
    )
    
    args = parser.parse_args()
    
    # If no explicit output directory is given, use the input folder so that
    # the model and embeddings are saved alongside the images.
    output_dir = args.output_dir if args.output_dir is not None else args.folder
    
    # Train the VAE
    model, embeddings, image_paths = train_vae(
        folder_path=args.folder,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        beta=args.beta,
        train_split=args.train_split,
        use_augmentation=not args.no_augment,
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"\nTo use the model for retrieval, you can:")
    print(f"1. Load the model: model, embeddings, paths = load_model_and_embeddings('{args.output_dir}')")
    print(f"2. Retrieve similar images: results = retrieve_similar_images('path/to/image.png', model, embeddings, paths)")


if __name__ == "__main__":
    main()

