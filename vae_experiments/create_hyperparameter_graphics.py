"""
Professional graphics generator for VAE hyperparameter experiments.
Creates comprehensive visualizations comparing different hyperparameter configurations.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palette for experiments - vibrant and distinct colors
COLORS = {
    'baseline': '#0066CC',           # Bright Blue
    'small_latent': '#CC0066',       # Bright Magenta/Pink
    'large_latent': '#FF6600',       # Bright Orange
    'low_beta': '#FF3333',           # Bright Red
    'high_beta': '#00CC66',          # Bright Green
    'low_lr': '#9933FF',             # Bright Purple
    'high_lr': '#FF0000',            # Pure Red
    'rotation_baseline': '#00CCCC'   # Bright Cyan/Teal
}

def load_experiment_data(experiment_dir):
    """Load experiment data from info.json file."""
    info_path = Path(experiment_dir) / 'info.json'
    if not info_path.exists():
        return None
    
    with open(info_path, 'r') as f:
        data = json.load(f)
    
    # Handle NaN values in losses
    train_losses = data.get('train_losses', [])
    test_losses = data.get('test_losses', [])
    
    # Filter out NaN values
    train_losses = [x for x in train_losses if not (isinstance(x, float) and np.isnan(x))]
    test_losses = [x for x in test_losses if not (isinstance(x, float) and np.isnan(x))]
    
    if len(train_losses) == 0 or len(test_losses) == 0:
        return None
    
    return {
        'name': Path(experiment_dir).name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_recon_losses': data.get('train_recon_losses', []),
        'train_kld_losses': data.get('train_kld_losses', []),
        'train_contrastive_losses': data.get('train_contrastive_losses', []),
        'epochs': len(train_losses),
        'latent_dim': data.get('latent_dim', 'N/A'),
        'beta': data.get('beta', 'N/A'),
        'learning_rate': data.get('learning_rate', 'N/A'),
        'batch_size': data.get('batch_size', 'N/A'),
        'train_size': data.get('train_size', 'N/A'),
    }

def get_color_for_experiment(exp_name):
    """Get color for experiment, handling vae_ prefix."""
    # Remove 'vae_' prefix if present
    key = exp_name.replace('vae_', '') if exp_name.startswith('vae_') else exp_name
    return COLORS.get(key, '#000000')

def create_loss_comparison_plot(experiments, output_path):
    """Create a comprehensive loss comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VAE Training: Hyperparameter Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Filter out experiments with no valid data
    valid_experiments = [exp for exp in experiments if exp is not None]
    
    # 1. Training Loss Comparison
    ax1 = axes[0, 0]
    for exp in valid_experiments:
        epochs = range(1, exp['epochs'] + 1)
        color = get_color_for_experiment(exp['name'])
        label = f"{exp['name'].replace('vae_', '').replace('_', ' ').title()}"
        # Enhanced plotting with thicker lines and larger markers
        ax1.plot(epochs, exp['train_losses'], color=color, linewidth=3, 
                marker='o', markersize=6, label=label, alpha=0.9, 
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss Over Epochs', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='gray', shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # 2. Test Loss Comparison
    ax2 = axes[0, 1]
    for exp in valid_experiments:
        epochs = range(1, exp['epochs'] + 1)
        color = get_color_for_experiment(exp['name'])
        label = f"{exp['name'].replace('vae_', '').replace('_', ' ').title()}"
        # Enhanced plotting with thicker lines and larger markers
        ax2.plot(epochs, exp['test_losses'], color=color, linewidth=3, 
                marker='s', markersize=6, label=label, alpha=0.9,
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    ax2.set_title('Test Loss Over Epochs', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor='gray', shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    
    # 3. Final Training Loss Comparison (Bar Chart)
    ax3 = axes[1, 0]
    exp_names = []
    final_train_losses = []
    colors_list = []
    
    for exp in valid_experiments:
        if len(exp['train_losses']) > 0:
            exp_names.append(exp['name'].replace('vae_', '').replace('_', '\n').title())
            final_train_losses.append(exp['train_losses'][-1])
            colors_list.append(get_color_for_experiment(exp['name']))
    
    bars = ax3.barh(exp_names, final_train_losses, color=colors_list, alpha=0.9, 
                   edgecolor='black', linewidth=2, hatch=None)
    ax3.set_xlabel('Final Training Loss', fontweight='bold')
    ax3.set_title('Final Training Loss by Experiment', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars with better styling
    for i, (bar, val) in enumerate(zip(bars, final_train_losses)):
        ax3.text(val, i, f' {val:.2f}', va='center', fontweight='bold', 
                fontsize=10, color='white' if val > max(final_train_losses) * 0.5 else 'black')
    
    # 4. Final Test Loss Comparison (Bar Chart)
    ax4 = axes[1, 1]
    exp_names = []
    final_test_losses = []
    colors_list = []
    
    for exp in valid_experiments:
        if len(exp['test_losses']) > 0:
            exp_names.append(exp['name'].replace('vae_', '').replace('_', '\n').title())
            final_test_losses.append(exp['test_losses'][-1])
            colors_list.append(get_color_for_experiment(exp['name']))
    
    bars = ax4.barh(exp_names, final_test_losses, color=colors_list, alpha=0.9, 
                   edgecolor='black', linewidth=2, hatch=None)
    ax4.set_xlabel('Final Test Loss', fontweight='bold')
    ax4.set_title('Final Test Loss by Experiment', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars with better styling
    for i, (bar, val) in enumerate(zip(bars, final_test_losses)):
        ax4.text(val, i, f' {val:.2f}', va='center', fontweight='bold', 
                fontsize=10, color='white' if val > max(final_test_losses) * 0.5 else 'black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    # Ensure colors are saved properly - use format='png' explicitly
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', format='png', dpi=300)
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def create_latent_dim_comparison(experiments, output_path):
    """Compare experiments with different latent dimensions."""
    latent_exps = {
        'small_latent': None,
        'baseline': None,
        'large_latent': None
    }
    
    for exp in experiments:
        if exp and exp['name'] in latent_exps:
            latent_exps[exp['name']] = exp
    
    if not any(latent_exps.values()):
        print("No latent dimension experiments found", file=sys.stdout, flush=True)
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Latent Dimension Comparison', fontsize=14, fontweight='bold')
    
    # Training losses
    ax1 = axes[0]
    for name, exp in latent_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"Latent Dim: {exp['latent_dim']}"
            ax1.plot(epochs, exp['train_losses'], color=color, linewidth=2.5, 
                    marker='o', markersize=5, label=label, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss: Latent Dimension Effect', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Test losses
    ax2 = axes[1]
    for name, exp in latent_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"Latent Dim: {exp['latent_dim']}"
            ax2.plot(epochs, exp['test_losses'], color=color, linewidth=2.5, 
                    marker='s', markersize=5, label=label, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    ax2.set_title('Test Loss: Latent Dimension Effect', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def create_beta_comparison(experiments, output_path):
    """Compare experiments with different beta values."""
    beta_exps = {
        'low_beta': None,
        'baseline': None,
        'high_beta': None
    }
    
    for exp in experiments:
        if exp and exp['name'] in beta_exps:
            beta_exps[exp['name']] = exp
    
    if not any(beta_exps.values()):
        print("No beta experiments found", file=sys.stdout, flush=True)
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Beta (KL Divergence Weight) Comparison', fontsize=14, fontweight='bold')
    
    # Training losses
    ax1 = axes[0]
    for name, exp in beta_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"β = {exp['beta']}"
            ax1.plot(epochs, exp['train_losses'], color=color, linewidth=2.5, 
                    marker='o', markersize=5, label=label, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss: Beta Effect', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Test losses
    ax2 = axes[1]
    for name, exp in beta_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"β = {exp['beta']}"
            ax2.plot(epochs, exp['test_losses'], color=color, linewidth=2.5, 
                    marker='s', markersize=5, label=label, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    ax2.set_title('Test Loss: Beta Effect', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def create_lr_comparison(experiments, output_path):
    """Compare experiments with different learning rates."""
    lr_exps = {
        'low_lr': None,
        'baseline': None,
        'high_lr': None
    }
    
    for exp in experiments:
        if exp and exp['name'] in lr_exps:
            lr_exps[exp['name']] = exp
    
    # Filter out high_lr if it has NaN values
    if lr_exps['high_lr'] and len(lr_exps['high_lr']['train_losses']) == 0:
        lr_exps['high_lr'] = None
    
    if not any([lr_exps['low_lr'], lr_exps['baseline']]):
        print("No valid learning rate experiments found", file=sys.stdout, flush=True)
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Learning Rate Comparison', fontsize=14, fontweight='bold')
    
    # Training losses
    ax1 = axes[0]
    for name, exp in lr_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"LR = {exp['learning_rate']}"
            if name == 'high_lr':
                label += " (diverged)"
            ax1.plot(epochs, exp['train_losses'], color=color, linewidth=2.5, 
                    marker='o', markersize=5, label=label, alpha=0.8, linestyle='--' if name == 'high_lr' else '-')
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss: Learning Rate Effect', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Test losses
    ax2 = axes[1]
    for name, exp in lr_exps.items():
        if exp:
            epochs = range(1, exp['epochs'] + 1)
            color = get_color_for_experiment(exp['name'])
            label = f"LR = {exp['learning_rate']}"
            if name == 'high_lr':
                label += " (diverged)"
            ax2.plot(epochs, exp['test_losses'], color=color, linewidth=2.5, 
                    marker='s', markersize=5, label=label, alpha=0.8, linestyle='--' if name == 'high_lr' else '-')
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    ax2.set_title('Test Loss: Learning Rate Effect', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def create_rotation_baseline_breakdown(experiments, output_path):
    """Create detailed breakdown for rotation baseline with component losses."""
    rotation_exp = None
    for exp in experiments:
        if exp and exp['name'] == 'rotation_baseline':
            rotation_exp = exp
            break
    
    if not rotation_exp or not rotation_exp.get('train_recon_losses'):
        print("Rotation baseline with component losses not found", file=sys.stdout, flush=True)
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rotation Baseline: Loss Component Analysis', fontsize=14, fontweight='bold')
    
    epochs = range(1, rotation_exp['epochs'] + 1)
    
    # Total training loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, rotation_exp['train_losses'], color='#2E86AB', linewidth=2.5, 
            marker='o', markersize=5, label='Total Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Total Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Component losses
    ax2 = axes[0, 1]
    ax2.plot(epochs, rotation_exp['train_recon_losses'], color='#C73E1D', linewidth=2.5, 
            marker='o', markersize=5, label='Reconstruction Loss', alpha=0.8)
    ax2.plot(epochs, rotation_exp['train_kld_losses'], color='#6A994E', linewidth=2.5, 
            marker='s', markersize=5, label='KL Divergence Loss', alpha=0.8)
    ax2.plot(epochs, rotation_exp['train_contrastive_losses'], color='#7209B7', linewidth=2.5, 
            marker='^', markersize=5, label='Contrastive Loss', alpha=0.8)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Training Loss Components', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Test loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, rotation_exp['test_losses'], color='#F18F01', linewidth=2.5, 
            marker='s', markersize=5, label='Test Loss', alpha=0.8)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Test Loss', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss component comparison (final values)
    ax4 = axes[1, 1]
    components = ['Reconstruction', 'KL Divergence', 'Contrastive']
    final_values = [
        rotation_exp['train_recon_losses'][-1],
        rotation_exp['train_kld_losses'][-1],
        rotation_exp['train_contrastive_losses'][-1]
    ]
    colors = ['#C73E1D', '#6A994E', '#7209B7']
    bars = ax4.bar(components, final_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Final Loss Value', fontweight='bold')
    ax4.set_title('Final Training Loss Components', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, final_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def create_hyperparameter_summary_table(experiments, output_path):
    """Create a summary table visualization of hyperparameters."""
    valid_experiments = [exp for exp in experiments if exp is not None]
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(valid_experiments) * 0.6)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Experiment', 'Latent Dim', 'Beta', 'Learning Rate', 'Batch Size', 
               'Final Train Loss', 'Final Test Loss', 'Epochs']
    
    for exp in valid_experiments:
        row = [
            exp['name'].replace('vae_', '').replace('_', ' ').title(),
            str(exp['latent_dim']),
            str(exp['beta']),
            str(exp['learning_rate']),
            str(exp['batch_size']),
            f"{exp['train_losses'][-1]:.2f}" if exp['train_losses'] else 'N/A',
            f"{exp['test_losses'][-1]:.2f}" if exp['test_losses'] else 'N/A',
            str(exp['epochs'])
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Color data rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1)
    
    plt.title('Hyperparameter Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"Saved: {output_path}", file=sys.stdout, flush=True)

def main():
    """Main function to generate all graphics."""
    base_dir = Path(__file__).parent
    
    # Create log file for debugging
    log_file = base_dir / 'graphics_generation.log'
    with open(log_file, 'w') as f:
        f.write("Starting graphics generation...\n")
    
    # List of experiment directories
    experiment_dirs = [
        base_dir / 'vae_baseline',
        base_dir / 'vae_small_latent',
        base_dir / 'vae_large_latent',
        base_dir / 'vae_low_beta',
        base_dir / 'vae_high_beta',
        base_dir / 'vae_low_lr',
        base_dir / 'vae_high_lr',
        base_dir / 'vae_rotation_baseline',
    ]
    
    # Load all experiment data
    experiments = []
    with open(log_file, 'a') as f:
        for exp_dir in experiment_dirs:
            data = load_experiment_data(exp_dir)
            experiments.append(data)
            if data:
                msg = f"Loaded: {data['name']}\n"
                print(msg, file=sys.stdout, flush=True)
                f.write(msg)
    
    # Create output directory for graphics
    output_dir = base_dir / 'hyperparameter_graphics'
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating graphics...", file=sys.stdout, flush=True)
    
    # Generate all visualizations
    try:
        create_loss_comparison_plot(experiments, output_dir / '01_loss_comparison.png')
        create_latent_dim_comparison(experiments, output_dir / '02_latent_dimension_comparison.png')
        create_beta_comparison(experiments, output_dir / '03_beta_comparison.png')
        create_lr_comparison(experiments, output_dir / '04_learning_rate_comparison.png')
        create_rotation_baseline_breakdown(experiments, output_dir / '05_rotation_baseline_breakdown.png')
        create_hyperparameter_summary_table(experiments, output_dir / '06_hyperparameter_summary_table.png')
        
        num_files = len(list(output_dir.glob('*.png')))
        print(f"\n✓ All graphics saved to: {output_dir}", file=sys.stdout, flush=True)
        print(f"  Generated {num_files} visualization files", file=sys.stdout, flush=True)
    except Exception as e:
        print(f"Error generating graphics: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
