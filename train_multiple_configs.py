"""
Train VAE with multiple hyperparameter configurations and save losses for comparison.

Supports both standard VAE training (train_vae_image_retrieval.py) and 
rotation-aware VAE training (train_vae_image_retrieval_rotations.py).
"""

import os
import subprocess
import json
from pathlib import Path

# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define different hyperparameter configurations for standard VAE
# Note: You can adjust epochs here - using 15 for faster training while still getting meaningful results
configs = [
    {
        "name": "baseline",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 1.0,
    },
    {
        "name": "high_lr",
        "epochs": 15,
        "batch_size": 32,
        "lr": 5e-3,
        "latent_dim": 128,
        "beta": 1.0,
    },
    {
        "name": "low_lr",
        "epochs": 15,
        "batch_size": 32,
        "lr": 5e-4,
        "latent_dim": 128,
        "beta": 1.0,
    },
    {
        "name": "high_beta",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 2.0,
    },
    {
        "name": "low_beta",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 0.5,
    },
    {
        "name": "large_latent",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 256,
        "beta": 1.0,
    },
    {
        "name": "small_latent",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 64,
        "beta": 1.0,
    },
]

# Define different hyperparameter configurations for rotation-aware VAE
rotation_configs = [
    {
        "name": "baseline",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 1.0,
        "contrastive_weight": 0.1,
    },
    {
        "name": "high_lr",
        "epochs": 15,
        "batch_size": 32,
        "lr": 5e-3,
        "latent_dim": 128,
        "beta": 1.0,
        "contrastive_weight": 0.1,
    },
    {
        "name": "low_lr",
        "epochs": 15,
        "batch_size": 32,
        "lr": 5e-4,
        "latent_dim": 128,
        "beta": 1.0,
        "contrastive_weight": 0.1,
    },
    {
        "name": "high_beta",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 2.0,
        "contrastive_weight": 0.1,
    },
    {
        "name": "low_beta",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 0.5,
        "contrastive_weight": 0.1,
    },
    {
        "name": "high_contrastive",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 1.0,
        "contrastive_weight": 0.5,
    },
    {
        "name": "low_contrastive",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 128,
        "beta": 1.0,
        "contrastive_weight": 0.05,
    },
    {
        "name": "large_latent",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 256,
        "beta": 1.0,
        "contrastive_weight": 0.1,
    },
    {
        "name": "small_latent",
        "epochs": 15,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 64,
        "beta": 1.0,
        "contrastive_weight": 0.1,
    },
]

def train_config(config, folder_path, base_output_dir):
    """Train a single standard VAE configuration."""
    config_name = config["name"]
    output_dir = os.path.join(base_output_dir, f"vae_{config_name}")
    
    print(f"\n{'='*60}")
    print(f"Training configuration: {config_name} (Standard VAE)")
    print(f"{'='*60}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Latent dim: {config['latent_dim']}")
    print(f"Beta: {config['beta']}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python", "train_vae_image_retrieval.py",
        "--folder", folder_path,
        "--output-dir", output_dir,
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--latent-dim", str(config["latent_dim"]),
        "--beta", str(config["beta"]),
    ]
    
    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully trained {config_name}")
        return output_dir
    else:
        print(f"\n✗ Failed to train {config_name}")
        return None

def train_rotation_config(config, train_folder, test_folder, base_output_dir, 
                          data_percentage=1.0, random_seed=42):
    """Train a single rotation-aware VAE configuration."""
    config_name = config["name"]
    output_dir = os.path.join(base_output_dir, f"vae_rotation_{config_name}")
    
    print(f"\n{'='*60}")
    print(f"Training configuration: {config_name} (Rotation-Aware VAE)")
    print(f"{'='*60}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Latent dim: {config['latent_dim']}")
    print(f"Beta: {config['beta']}")
    print(f"Contrastive weight: {config['contrastive_weight']}")
    print(f"Train folder: {train_folder}")
    print(f"Test folder: {test_folder}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python", "train_vae_image_retrieval_rotations.py",
        "--train-folder", train_folder,
        "--test-folder", test_folder,
        "--output-dir", output_dir,
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--latent-dim", str(config["latent_dim"]),
        "--beta", str(config["beta"]),
        "--contrastive-weight", str(config["contrastive_weight"]),
        "--data-percentage", str(data_percentage),
        "--random-seed", str(random_seed),
    ]
    
    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully trained {config_name} (rotation-aware)")
        return output_dir
    else:
        print(f"\n✗ Failed to train {config_name} (rotation-aware)")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train VAE with multiple hyperparameter configurations. "
                    "Supports both standard VAE and rotation-aware VAE training."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="output_data copy",
        help="Folder containing PNG images (for standard VAE training)"
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        default="output_data/14_views_10000_mechanical",
        help="Folder containing training PNG images (for rotation-aware VAE)"
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="output_data/14_views_10000_mechanical/test",
        help="Folder containing test PNG images (for rotation-aware VAE)"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="vae_experiments",
        help="Base directory to save all experiment outputs"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["standard", "rotation", "both"],
        default="both",
        help="Type of training to run: 'standard', 'rotation', or 'both' (default: both)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Specific configurations to train (by name). If not specified, trains all."
    )
    parser.add_argument(
        "--rotation-configs",
        type=str,
        nargs="+",
        default=None,
        help="Specific rotation configurations to train (by name). If not specified, trains all."
    )
    parser.add_argument(
        "--data-percentage",
        type=float,
        default=1.0,
        help="Percentage of data to use for rotation-aware training (0.0 to 1.0, default: 1.0)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible data sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    results = {}
    total_configs = 0
    
    # Train standard VAE configurations
    if args.type in ["standard", "both"]:
        if args.configs:
            configs_to_train = [c for c in configs if c["name"] in args.configs]
            if len(configs_to_train) == 0:
                print(f"Error: No matching standard configurations found. Available: {[c['name'] for c in configs]}")
                configs_to_train = []
        else:
            configs_to_train = configs
        
        if len(configs_to_train) > 0:
            total_configs += len(configs_to_train)
            print(f"\n{'='*70}")
            print(f"STANDARD VAE TRAINING")
            print(f"{'='*70}")
            print(f"Training {len(configs_to_train)} standard configurations...")
            print(f"Image folder: {args.folder}")
            print(f"Output base directory: {args.output_base_dir}\n")
            
            for config in configs_to_train:
                output_dir = train_config(config, args.folder, args.output_base_dir)
                if output_dir:
                    results[f"standard_{config['name']}"] = {
                        "output_dir": output_dir,
                        "config": config,
                        "type": "standard"
                    }
    
    # Train rotation-aware VAE configurations
    if args.type in ["rotation", "both"]:
        if args.rotation_configs:
            rotation_configs_to_train = [c for c in rotation_configs if c["name"] in args.rotation_configs]
            if len(rotation_configs_to_train) == 0:
                print(f"Error: No matching rotation configurations found. Available: {[c['name'] for c in rotation_configs]}")
                rotation_configs_to_train = []
        else:
            rotation_configs_to_train = rotation_configs
        
        if len(rotation_configs_to_train) > 0:
            total_configs += len(rotation_configs_to_train)
            print(f"\n{'='*70}")
            print(f"ROTATION-AWARE VAE TRAINING")
            print(f"{'='*70}")
            print(f"Training {len(rotation_configs_to_train)} rotation-aware configurations...")
            print(f"Train folder: {args.train_folder}")
            print(f"Test folder: {args.test_folder}")
            print(f"Output base directory: {args.output_base_dir}\n")
            
            for config in rotation_configs_to_train:
                output_dir = train_rotation_config(
                    config, 
                    args.train_folder, 
                    args.test_folder, 
                    args.output_base_dir,
                    data_percentage=args.data_percentage,
                    random_seed=args.random_seed
                )
                if output_dir:
                    results[f"rotation_{config['name']}"] = {
                        "output_dir": output_dir,
                        "config": config,
                        "type": "rotation_aware"
                    }
    
    # Save summary
    summary_path = os.path.join(args.output_base_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    if total_configs == 0:
        print("No configurations to train. Please check your arguments.")
        print(f"Available standard configs: {[c['name'] for c in configs]}")
        print(f"Available rotation configs: {[c['name'] for c in rotation_configs]}")
    else:
        print(f"Successfully trained: {len(results)}/{total_configs} configurations")
    
    # Count by type
    standard_count = sum(1 for r in results.values() if r.get("type") == "standard")
    rotation_count = sum(1 for r in results.values() if r.get("type") == "rotation_aware")
    if standard_count > 0:
        print(f"  - Standard VAE: {standard_count}")
    if rotation_count > 0:
        print(f"  - Rotation-aware VAE: {rotation_count}")
    
    print(f"Summary saved to: {summary_path}")
    print(f"\nTo visualize the results, run:")
    print(f"python plot_training_losses.py --experiments-dir {args.output_base_dir}")
    if rotation_count > 0:
        print(f"  (with --show-loss-breakdown to see reconstruction, KL, and contrastive losses)")

if __name__ == "__main__":
    main()

