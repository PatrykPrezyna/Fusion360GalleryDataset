# How to Run the Hyperparameter Graphics Generator

## Quick Start

Simply run the Python script from the `vae_experiments` directory:

```bash
cd vae_experiments
python create_hyperparameter_graphics.py
```

## What It Does

The script will:
1. Load all experiment data from the `info.json` files in each experiment folder
2. Generate 6 professional graphics comparing different hyperparameters:
   - `01_loss_comparison.png` - Overall loss comparison across all experiments
   - `02_latent_dimension_comparison.png` - Comparison of different latent dimensions
   - `03_beta_comparison.png` - Comparison of different beta (KL divergence weight) values
   - `04_learning_rate_comparison.png` - Comparison of different learning rates
   - `05_rotation_baseline_breakdown.png` - Detailed breakdown of rotation baseline experiment
   - `06_hyperparameter_summary_table.png` - Summary table of all hyperparameters

## Output Location

All graphics will be saved to:
```
vae_experiments/hyperparameter_graphics/
```

## Requirements

Make sure you have the required Python packages installed:
```bash
pip install matplotlib seaborn numpy
```

## Example Output

After running, you should see output like:
```
Loaded: vae_baseline
Loaded: vae_small_latent
Loaded: vae_large_latent
...

Generating graphics...
Saved: hyperparameter_graphics/01_loss_comparison.png
Saved: hyperparameter_graphics/02_latent_dimension_comparison.png
...

✓ All graphics saved to: hyperparameter_graphics
  Generated 6 visualization files
```
