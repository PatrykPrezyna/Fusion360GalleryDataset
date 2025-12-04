"""
Plot training and test losses from multiple VAE training experiments on a combined graph.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_losses_from_experiment(experiment_dir):
    """Load losses from a single experiment directory."""
    losses_path = os.path.join(experiment_dir, "losses.npz")
    info_path = os.path.join(experiment_dir, "info.json")
    
    if not os.path.exists(losses_path) and not os.path.exists(info_path):
        return None
    
    # Try loading from losses.npz first (preferred)
    if os.path.exists(losses_path):
        data = np.load(losses_path)
        train_losses = data["train_losses"]
        test_losses = data["test_losses"]
    # Fallback to info.json
    elif os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        train_losses = np.array(info.get("train_losses", []))
        test_losses = np.array(info.get("test_losses", []))
    else:
        return None
    
    # Load config info if available
    config = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
            config = {
                "lr": info.get("learning_rate", "?"),
                "latent_dim": info.get("latent_dim", "?"),
                "beta": info.get("beta", "?"),
                "batch_size": info.get("batch_size", "?"),
            }
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "config": config,
    }

def plot_combined_losses(experiments_dir, save_path=None):
    """Plot combined training and test losses from multiple experiments."""
    experiments_dir = Path(experiments_dir)
    
    # Find all experiment directories
    experiment_dirs = []
    for item in experiments_dir.iterdir():
        if item.is_dir() and item.name.startswith("vae_"):
            experiment_dirs.append(item)
    
    if len(experiment_dirs) == 0:
        print(f"No experiment directories found in {experiments_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Load losses from each experiment
    experiments_data = {}
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name.replace("vae_", "")
        data = load_losses_from_experiment(exp_dir)
        if data is not None:
            experiments_data[exp_name] = data
            print(f"  ✓ Loaded {exp_name}")
        else:
            print(f"  ✗ Could not load losses from {exp_name}")
    
    if len(experiments_data) == 0:
        print("No valid experiment data found!")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training losses
    for exp_name, data in experiments_data.items():
        train_losses = data["train_losses"]
        epochs = np.arange(1, len(train_losses) + 1)
        config = data["config"]
        
        # Create label with config info
        label = f"{exp_name}\n(lr={config.get('lr', '?')}, β={config.get('beta', '?')}, dim={config.get('latent_dim', '?')})"
        ax1.plot(epochs, train_losses, label=label, linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot test losses
    for exp_name, data in experiments_data.items():
        test_losses = data["test_losses"]
        epochs = np.arange(1, len(test_losses) + 1)
        config = data["config"]
        
        label = f"{exp_name}\n(lr={config.get('lr', '?')}, β={config.get('beta', '?')}, dim={config.get('latent_dim', '?')})"
        ax2.plot(epochs, test_losses, label=label, linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Test Loss", fontsize=12)
    ax2.set_title("Test Loss Comparison", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to {save_path}")
    else:
        plt.savefig(os.path.join(experiments_dir, "loss_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to {os.path.join(experiments_dir, 'loss_comparison.png')}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined training losses from multiple VAE experiments"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="vae_experiments",
        help="Directory containing experiment outputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: experiments_dir/loss_comparison.png)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        return
    
    plot_combined_losses(args.experiments_dir, args.output)

if __name__ == "__main__":
    main()

