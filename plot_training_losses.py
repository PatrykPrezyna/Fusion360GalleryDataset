"""
Plot training and test losses from multiple VAE training experiments on a combined graph.
Styled following The Economist's visualization guidelines.

Supports both standard VAE experiments and rotation-aware VAE experiments from
train_vae_image_retrieval_rotations.py.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
import argparse

# Economist-style color palette
ECONOMIST_COLORS = {
    'primary_blue': '#0E4C7A',      # Dark blue (primary)
    'secondary_blue': '#4A90E2',    # Medium blue
    'light_blue': '#7BB3E8',        # Light blue
    'accent_red': '#E3120B',        # Red accent
    'dark_gray': '#2C3E50',         # Dark gray
    'medium_gray': '#7F8C8D',       # Medium gray
    'light_gray': '#BDC3C7',        # Light gray
    'background': '#FFFFFF',        # White background
    'grid': '#E8E8E8',              # Very light gray for grid
}

# Extended color palette for multiple lines
ECONOMIST_PALETTE = [
    '#0E4C7A',  # Dark blue
    '#E3120B',  # Red
    '#4A90E2',  # Medium blue
    '#2C3E50',  # Dark gray
    '#7BB3E8',  # Light blue
    '#7F8C8D',  # Medium gray
    '#27AE60',  # Green
    '#F39C12',  # Orange
]

def load_losses_from_experiment(experiment_dir):
    """Load losses from a single experiment directory.
    
    Supports both standard VAE experiments and rotation-aware VAE experiments.
    """
    experiment_dir = Path(experiment_dir)
    losses_path = experiment_dir / "losses.npz"
    info_path = experiment_dir / "info.json"
    
    if not losses_path.exists() and not info_path.exists():
        return None
    
    # Try loading from losses.npz first (preferred)
    train_losses = None
    test_losses = None
    train_recon_losses = None
    train_kld_losses = None
    train_contrastive_losses = None
    
    if losses_path.exists():
        data = np.load(losses_path)
        train_losses = data.get("train_losses", None)
        test_losses = data.get("test_losses", None)
        # Additional losses from rotation-aware training
        train_recon_losses = data.get("train_recon_losses", None)
        train_kld_losses = data.get("train_kld_losses", None)
        train_contrastive_losses = data.get("train_contrastive_losses", None)
    
    # Fallback to info.json
    if train_losses is None and info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        train_losses = np.array(info.get("train_losses", [])) if info.get("train_losses") else None
        test_losses = np.array(info.get("test_losses", [])) if info.get("test_losses") else None
        train_recon_losses = np.array(info.get("train_recon_losses", [])) if info.get("train_recon_losses") else None
        train_kld_losses = np.array(info.get("train_kld_losses", [])) if info.get("train_kld_losses") else None
        train_contrastive_losses = np.array(info.get("train_contrastive_losses", [])) if info.get("train_contrastive_losses") else None
    
    if train_losses is None or test_losses is None:
        return None
    
    # Load config info if available
    config = {}
    experiment_type = "standard"  # Default to standard VAE
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
            config = {
                "lr": info.get("learning_rate", "?"),
                "latent_dim": info.get("latent_dim", "?"),
                "beta": info.get("beta", "?"),
                "batch_size": info.get("batch_size", "?"),
                "contrastive_weight": info.get("contrastive_weight", None),
            }
            # Detect rotation-aware training by presence of contrastive_weight
            if info.get("contrastive_weight") is not None:
                experiment_type = "rotation_aware"
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_recon_losses": train_recon_losses,
        "train_kld_losses": train_kld_losses,
        "train_contrastive_losses": train_contrastive_losses,
        "config": config,
        "experiment_type": experiment_type,
        "experiment_dir": str(experiment_dir),
    }

def apply_economist_style(ax, title, ylabel):
    """Apply Economist-style formatting to an axis."""
    # Set background color
    ax.set_facecolor(ECONOMIST_COLORS['background'])
    
    # Format title - clean and professional
    ax.set_title(title, fontsize=16, fontweight='600', color=ECONOMIST_COLORS['dark_gray'], 
                 pad=20, loc='left')
    
    # Format labels
    ax.set_xlabel("Epoch", fontsize=12, color=ECONOMIST_COLORS['dark_gray'], fontweight='500')
    ax.set_ylabel(ylabel, fontsize=12, color=ECONOMIST_COLORS['dark_gray'], fontweight='500')
    
    # Format ticks
    ax.tick_params(colors=ECONOMIST_COLORS['medium_gray'], labelsize=10)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    
    # Format spines - remove top and right, style bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(ECONOMIST_COLORS['light_gray'])
    ax.spines['bottom'].set_color(ECONOMIST_COLORS['light_gray'])
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Grid - subtle and clean
    ax.grid(True, color=ECONOMIST_COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis to avoid too many decimal places
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}' if x < 100 else f'{x:.0f}'))
    
    return ax

def format_config_label(exp_name, config, experiment_type="standard", include_details=False):
    """Format configuration label in a clean, Economist-style way."""
    lr = config.get('lr', '?')
    beta = config.get('beta', '?')
    dim = config.get('latent_dim', '?')
    contrastive_weight = config.get('contrastive_weight', None)
    
    # Format learning rate nicely
    if isinstance(lr, (int, float)):
        if lr >= 0.001:
            lr_str = f"{lr:.0e}"
        else:
            lr_str = f"{lr:.4f}"
    else:
        lr_str = str(lr)
    
    # Create clean label
    label = f"{exp_name.replace('_', ' ').title()}"
    
    # Add experiment type prefix if rotation-aware
    if experiment_type == "rotation_aware":
        label = f"[Rotation] {label}"
    
    # Optionally add configuration details
    if include_details:
        details = f"lr={lr_str}, β={beta}, dim={dim}"
        if contrastive_weight is not None:
            details += f", contrastive={contrastive_weight}"
        label += f" ({details})"
    
    return label

def find_experiment_directories(search_dirs):
    """Find all directories containing training outputs (losses.npz or info.json).
    
    Args:
        search_dirs: List of directories to search (can be Path objects or strings)
    
    Returns:
        List of experiment directory paths
    """
    experiment_dirs = []
    
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            print(f"Warning: Search directory does not exist: {search_dir}")
            continue
        
        # If the search directory itself contains losses/info, add it
        if (search_path / "losses.npz").exists() or (search_path / "info.json").exists():
            experiment_dirs.append(search_path)
            continue
        
        # Otherwise, search for subdirectories
        # First, look for directories starting with "vae_" (standard pattern)
        for item in search_path.iterdir():
            if item.is_dir() and item.name.startswith("vae_"):
                if (item / "losses.npz").exists() or (item / "info.json").exists():
                    experiment_dirs.append(item)
        
        # Also search recursively for any directory containing losses.npz or info.json
        # (for rotation-aware training outputs that might be in different locations)
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            root_path = Path(root)
            if (root_path / "losses.npz").exists() or (root_path / "info.json").exists():
                if root_path not in experiment_dirs:
                    experiment_dirs.append(root_path)
    
    return experiment_dirs

def plot_combined_losses(experiments_dirs, save_path=None, show_config_details=False, 
                         show_loss_breakdown=False):
    """Plot combined training and test losses from multiple experiments with Economist style.
    
    Args:
        experiments_dirs: Directory or list of directories containing experiment outputs
        save_path: Optional path to save the plot
        show_config_details: If True, include hyperparameter details in legend labels
        show_loss_breakdown: If True, show additional plots for loss components (recon, KL, contrastive)
    """
    # Convert single directory to list
    if isinstance(experiments_dirs, (str, Path)):
        experiments_dirs = [experiments_dirs]
    
    # Find all experiment directories
    experiment_dirs = find_experiment_directories(experiments_dirs)
    
    if len(experiment_dirs) == 0:
        print(f"No experiment directories found in {experiments_dirs}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Load losses from each experiment
    experiments_data = {}
    for exp_dir in experiment_dirs:
        # Generate a unique name for this experiment
        exp_name = exp_dir.name
        # If it's in a parent directory, include parent name for uniqueness
        if exp_dir.parent.name and exp_dir.parent.name not in ["", "."]:
            # Only add parent if it's meaningful (not just ".")
            parent_name = exp_dir.parent.name
            if parent_name not in ["output_data", "vae_experiments"]:  # Skip common parent dirs
                exp_name = f"{parent_name}_{exp_name}"
        
        # Remove "vae_" prefix if present for cleaner names
        if exp_name.startswith("vae_"):
            exp_name = exp_name.replace("vae_", "")
        
        data = load_losses_from_experiment(exp_dir)
        if data is not None:
            # Use full path as key to ensure uniqueness
            key = str(exp_dir)
            experiments_data[key] = data
            experiments_data[key]["display_name"] = exp_name
            print(f"  ✓ Loaded {exp_name} ({data['experiment_type']})")
        else:
            print(f"  ✗ Could not load losses from {exp_dir}")
    
    if len(experiments_data) == 0:
        print("No valid experiment data found!")
        return
    
    # Set up the figure with Economist style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['figure.facecolor'] = ECONOMIST_COLORS['background']
    
    # Determine figure layout based on whether to show loss breakdown
    if show_loss_breakdown:
        # Check if any experiments have loss breakdown data
        has_breakdown = any(
            data.get("train_recon_losses") is not None 
            for data in experiments_data.values()
        )
        if has_breakdown:
            # Create figure with more subplots for breakdown
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])  # Training loss
            ax2 = fig.add_subplot(gs[0, 1])  # Test loss
            ax3 = fig.add_subplot(gs[0, 2])  # Reconstruction loss
            ax4 = fig.add_subplot(gs[1, 0])  # KL divergence loss
            ax5 = fig.add_subplot(gs[1, 1])  # Contrastive loss
            ax6 = fig.add_subplot(gs[1, 2])  # Combined breakdown (if available)
        else:
            show_loss_breakdown = False  # No breakdown data available
    
    if not show_loss_breakdown:
        # Standard layout: two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    fig.patch.set_facecolor(ECONOMIST_COLORS['background'])
    
    # Get color cycle
    colors = ECONOMIST_PALETTE * (len(experiments_data) // len(ECONOMIST_PALETTE) + 1)
    
    # Plot training losses
    for idx, (exp_key, data) in enumerate(experiments_data.items()):
        train_losses = data["train_losses"]
        epochs = np.arange(1, len(train_losses) + 1)
        config = data["config"]
        exp_name = data["display_name"]
        exp_type = data["experiment_type"]
        
        # Get color for this line
        color = colors[idx % len(ECONOMIST_PALETTE)]
        
        # Format label
        label = format_config_label(exp_name, config, exp_type, include_details=show_config_details)
        
        # Plot with Economist style - clean lines, subtle markers
        ax1.plot(epochs, train_losses, label=label, 
                linewidth=2.5, color=color, alpha=0.9,
                marker='o', markersize=5, markevery=max(1, len(epochs)//10),
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5)
    
    # Apply Economist styling
    apply_economist_style(ax1, "Training Loss", "Training Loss")
    
    # Format legend - clean and professional
    legend1 = ax1.legend(loc='upper right', frameon=True, fancybox=False, 
                         shadow=False, framealpha=0.95, edgecolor=ECONOMIST_COLORS['light_gray'],
                         facecolor=ECONOMIST_COLORS['background'], fontsize=9)
    legend1.get_frame().set_linewidth(0.5)
    for text in legend1.get_texts():
        text.set_color(ECONOMIST_COLORS['dark_gray'])
        text.set_fontweight('500')
    
    # Plot test losses
    for idx, (exp_key, data) in enumerate(experiments_data.items()):
        test_losses = data["test_losses"]
        epochs = np.arange(1, len(test_losses) + 1)
        config = data["config"]
        exp_name = data["display_name"]
        exp_type = data["experiment_type"]
        
        # Get color for this line (same as training)
        color = colors[idx % len(ECONOMIST_PALETTE)]
        
        # Format label
        label = format_config_label(exp_name, config, exp_type, include_details=show_config_details)
        
        # Plot with Economist style
        ax2.plot(epochs, test_losses, label=label,
                linewidth=2.5, color=color, alpha=0.9,
                marker='s', markersize=5, markevery=max(1, len(epochs)//10),
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5)
    
    # Apply Economist styling
    apply_economist_style(ax2, "Test Loss", "Test Loss")
    
    # Format legend
    legend2 = ax2.legend(loc='upper right', frameon=True, fancybox=False,
                         shadow=False, framealpha=0.95, edgecolor=ECONOMIST_COLORS['light_gray'],
                         facecolor=ECONOMIST_COLORS['background'], fontsize=9)
    legend2.get_frame().set_linewidth(0.5)
    for text in legend2.get_texts():
        text.set_color(ECONOMIST_COLORS['dark_gray'])
        text.set_fontweight('500')
    
    # Plot loss breakdown if requested and available
    if show_loss_breakdown:
        # Reconstruction loss
        for idx, (exp_key, data) in enumerate(experiments_data.items()):
            recon_losses = data.get("train_recon_losses")
            if recon_losses is not None:
                epochs = np.arange(1, len(recon_losses) + 1)
                exp_name = data["display_name"]
                exp_type = data["experiment_type"]
                color = colors[idx % len(ECONOMIST_PALETTE)]
                label = format_config_label(exp_name, data["config"], exp_type, include_details=False)
                ax3.plot(epochs, recon_losses, label=label, linewidth=2.5, color=color, alpha=0.9,
                        marker='o', markersize=4, markevery=max(1, len(epochs)//10))
        apply_economist_style(ax3, "Reconstruction Loss", "Reconstruction Loss")
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.95)
        
        # KL divergence loss
        for idx, (exp_key, data) in enumerate(experiments_data.items()):
            kld_losses = data.get("train_kld_losses")
            if kld_losses is not None:
                epochs = np.arange(1, len(kld_losses) + 1)
                exp_name = data["display_name"]
                exp_type = data["experiment_type"]
                color = colors[idx % len(ECONOMIST_PALETTE)]
                label = format_config_label(exp_name, data["config"], exp_type, include_details=False)
                ax4.plot(epochs, kld_losses, label=label, linewidth=2.5, color=color, alpha=0.9,
                        marker='s', markersize=4, markevery=max(1, len(epochs)//10))
        apply_economist_style(ax4, "KL Divergence Loss", "KL Divergence Loss")
        ax4.legend(loc='upper right', fontsize=8, framealpha=0.95)
        
        # Contrastive loss (only for rotation-aware experiments)
        for idx, (exp_key, data) in enumerate(experiments_data.items()):
            contrastive_losses = data.get("train_contrastive_losses")
            if contrastive_losses is not None:
                epochs = np.arange(1, len(contrastive_losses) + 1)
                exp_name = data["display_name"]
                exp_type = data["experiment_type"]
                color = colors[idx % len(ECONOMIST_PALETTE)]
                label = format_config_label(exp_name, data["config"], exp_type, include_details=False)
                ax5.plot(epochs, contrastive_losses, label=label, linewidth=2.5, color=color, alpha=0.9,
                        marker='^', markersize=4, markevery=max(1, len(epochs)//10))
        apply_economist_style(ax5, "Contrastive Loss", "Contrastive Loss")
        ax5.legend(loc='upper right', fontsize=8, framealpha=0.95)
        
        # Combined breakdown visualization (stacked area or line comparison)
        # Show all components together for experiments that have them
        for idx, (exp_key, data) in enumerate(experiments_data.items()):
            recon = data.get("train_recon_losses")
            kld = data.get("train_kld_losses")
            contrastive = data.get("train_contrastive_losses")
            
            if recon is not None and kld is not None:
                epochs = np.arange(1, len(recon) + 1)
                exp_name = data["display_name"]
                exp_type = data["experiment_type"]
                color = colors[idx % len(ECONOMIST_PALETTE)]
                label = format_config_label(exp_name, data["config"], exp_type, include_details=False)
                
                # Plot reconstruction and KL together
                ax6.plot(epochs, recon, label=f"{label} (Recon)", linewidth=2, 
                        color=color, alpha=0.7, linestyle='-')
                ax6.plot(epochs, kld, label=f"{label} (KL)", linewidth=2, 
                        color=color, alpha=0.7, linestyle='--')
                if contrastive is not None:
                    ax6.plot(epochs, contrastive, label=f"{label} (Contrastive)", linewidth=2, 
                            color=color, alpha=0.7, linestyle=':')
        
        apply_economist_style(ax6, "Loss Components Comparison", "Loss Value")
        ax6.legend(loc='upper right', fontsize=7, framealpha=0.95, ncol=1)
    
    # Add overall title with Economist style
    title = "VAE Training: Loss Comparison Across Hyperparameter Configurations"
    if show_loss_breakdown:
        title += " (with Loss Breakdown)"
    fig.suptitle(title, fontsize=18, fontweight='600', color=ECONOMIST_COLORS['dark_gray'],
                 y=0.98, x=0.02, ha='left')
    
    # Add subtle subtitle with experiment count
    exp_types = [data["experiment_type"] for data in experiments_data.values()]
    type_counts = {}
    for exp_type in exp_types:
        type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
    
    subtitle_parts = [f"Comparing {len(experiments_data)} experiments"]
    if "rotation_aware" in type_counts:
        subtitle_parts.append(f"{type_counts['rotation_aware']} rotation-aware")
    if "standard" in type_counts:
        subtitle_parts.append(f"{type_counts['standard']} standard")
    
    fig.text(0.02, 0.95, " | ".join(subtitle_parts),
             fontsize=11, color=ECONOMIST_COLORS['medium_gray'], style='italic', ha='left')
    
    # Tight layout with padding
    if show_loss_breakdown:
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=ECONOMIST_COLORS['background'])
        print(f"\nSaved plot to {save_path}")
    else:
        # Use first search directory as default output location
        default_dir = Path(experiments_dirs[0]) if isinstance(experiments_dirs[0], (str, Path)) else Path(experiments_dirs[0])
        output_path = default_dir / "loss_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=ECONOMIST_COLORS['background'])
        print(f"\nSaved plot to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined training losses from multiple VAE experiments. "
                    "Supports both standard VAE experiments and rotation-aware VAE experiments."
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        nargs="+",
        default=["vae_experiments"],
        help="Directory or directories containing experiment outputs. "
             "Can specify multiple directories. Script will search recursively for "
             "directories containing losses.npz or info.json files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: first experiments_dir/loss_comparison.png)"
    )
    parser.add_argument(
        "--show-config-details",
        action="store_true",
        help="Include hyperparameter details in legend labels"
    )
    parser.add_argument(
        "--show-loss-breakdown",
        action="store_true",
        help="Show additional plots for loss components (reconstruction, KL divergence, contrastive). "
             "Only shown if available in the training data."
    )
    
    args = parser.parse_args()
    
    # Check that at least one directory exists
    valid_dirs = []
    for exp_dir in args.experiments_dir:
        if os.path.exists(exp_dir):
            valid_dirs.append(exp_dir)
        else:
            print(f"Warning: Directory not found: {exp_dir}")
    
    if len(valid_dirs) == 0:
        print(f"Error: No valid experiment directories found!")
        return
    
    plot_combined_losses(valid_dirs, args.output, args.show_config_details, args.show_loss_breakdown)

if __name__ == "__main__":
    main()

