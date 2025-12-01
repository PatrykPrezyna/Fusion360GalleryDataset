
## Getting Started

This section provides step-by-step instructions to set up and use this repository.

### Prerequisites

- **Python 3.10** (tested; e.g., 3.10.19)
- **Conda** (for managing Python environments)
- **Git** (to clone the repository)

### Installation

#### Option 1: Using Conda (Recommended)

1. **Create and activate a conda environment with `pythonocc-core`:**
   ```bash
   conda create -n fusion360 -c conda-forge python=3.10 pythonocc-core -y
   conda activate fusion360
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Option 2: Using Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv myenv
   ```

2. **Activate the environment:**
   - **Windows (Command Prompt):**
     ```cmd
     .\myenv\Scripts\activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     .\myenv\Scripts\Activate.ps1
     ```
   - **Linux/Mac:**
     ```bash
     source myenv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Downloading the Dataset

To work with assembly data, download the dataset using the provided tool:

1. Follow the instructions in [`tools/assembly_download`](tools/assembly_download)
2. Place the downloaded data in the `input_data` folder

---

## Usage

### Create a data set

### Using `rotate_STEP_and save_png_all.py`

This script scans the `input_data` folder for `.step` files and exports rendered PNG images for each part in configurable batches.

- **Basic usage (first batch, default batch size 200):**

```bash
python "rotate_STEP_and save_png_all.py"
```

- **Custom batch and batch size:**

```bash
# Process batch 0 (files 1–100)
python "rotate_STEP_and save_png_all.py" --batch 0 --batch-size 100

# Process batch 1 (files 101–200) and write images to a custom folder
python "rotate_STEP_and save_png_all.py" --batch 1 --batch-size 100 --output-folder "output_data_batch1"
```



**Notes:**
- STEP files must be under the `input_data` directory.
- Exported PNGs are saved in the chosen output folder (default `output_data`).
- Each STEP file can produce multiple grayscale images with standardized views.


### VAE Image Retrieval System

This repository includes a Variational Autoencoder (VAE) for image-based retrieval of CAD assembly images. The system can learn to find similar images based on their visual features.

#### Training the VAE

Train the VAE on a folder of images:

```bash
python train_vae_image_retrieval.py --folder "output_data copy" --epochs 30
```

**Parameters:**
- `--folder`: Path to folder containing PNG images (default: `"output_data copy"`)
- `--output-dir`: Directory to save model and embeddings (default: `"vae_retrieval_output"`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 32)
- `--latent-dim`: Dimension of latent space (default: 128)
- `--image-size`: Size to resize images to (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--beta`: KL divergence weight (default: 1.0)

**Example:**
```bash
python train_vae_image_retrieval.py --folder "output_data copy" --epochs 50 --latent-dim 256 --batch-size 64
```

#### Retrieving Similar Images

After training, use the model to find similar images:

```bash
python retrieve_similar_images.py --query "path/to/image.png" --top-k 5
```

**Parameters:**
- `--query`: Path to the query image
- `--output-dir`: Directory containing trained model (default: `"vae_retrieval_output"`)
- `--top-k`: Number of similar images to retrieve (default: 5)

**Example:**
```bash
python retrieve_similar_images.py --query "output_data copy/d853e1ca-0586-11ec-816a-0690f2e5563f_rot_0.png" --top-k 10
```



---

## Troubleshooting

### OpenMP Error on Windows

If you encounter an OpenMP error like:
```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

This is already handled in the scripts by setting `KMP_DUPLICATE_LIB_OK=TRUE`. If you still see this error, you can manually set it:

**Windows (Command Prompt):**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
python train_vae_image_retrieval.py
```

**Windows (PowerShell):**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_vae_image_retrieval.py
```

**Linux/Mac:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python train_vae_image_retrieval.py
```

### CUDA/GPU Issues

The scripts automatically detect and use GPU if available. To force CPU usage, modify the device selection in the scripts or set:
```python
device = torch.device("cpu")
```

### Missing Dependencies

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For `pythonocc-core` (required for STEP file processing), use conda:
```bash
conda install -c conda-forge pythonocc-core
```


