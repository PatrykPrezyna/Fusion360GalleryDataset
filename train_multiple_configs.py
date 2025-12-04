"""
Train VAE with multiple hyperparameter configurations and save losses for comparison.
"""

import os
import subprocess
import json
from pathlib import Path

# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define different hyperparameter configurations
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

def train_config(config, folder_path, base_output_dir):
    """Train a single configuration."""
    config_name = config["name"]
    output_dir = os.path.join(base_output_dir, f"vae_{config_name}")
    
    print(f"\n{'='*60}")
    print(f"Training configuration: {config_name}")
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

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train VAE with multiple hyperparameter configurations"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="output_data copy",
        help="Folder containing PNG images"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="vae_experiments",
        help="Base directory to save all experiment outputs"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Specific configurations to train (by name). If not specified, trains all."
    )
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Filter configs if specified
    configs_to_train = configs
    if args.configs:
        configs_to_train = [c for c in configs if c["name"] in args.configs]
        if len(configs_to_train) == 0:
            print(f"Error: No matching configurations found. Available: {[c['name'] for c in configs]}")
            return
    
    print(f"Training {len(configs_to_train)} configurations...")
    print(f"Image folder: {args.folder}")
    print(f"Output base directory: {args.output_base_dir}\n")
    
    # Train each configuration
    results = {}
    for config in configs_to_train:
        output_dir = train_config(config, args.folder, args.output_base_dir)
        if output_dir:
            results[config["name"]] = {
                "output_dir": output_dir,
                "config": config
            }
    
    # Save summary
    summary_path = os.path.join(args.output_base_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Successfully trained: {len(results)}/{len(configs_to_train)} configurations")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTo visualize the results, run:")
    print(f"python plot_training_losses.py --experiments-dir {args.output_base_dir}")

if __name__ == "__main__":
    main()

