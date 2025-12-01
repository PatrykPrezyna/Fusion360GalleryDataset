
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
   <!-- conda activate fusion-occ -->
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

### Original Data

To work with assembly data, download the dataset using the provided tool:

1. Follow the instructions in [`tools/assembly_download`](tools/assembly_download)
2. Place the downloaded data in the `input_data` folder


### Pretrained data

1. Ask Patryk for acces to the drive: https://drive.google.com/drive/folders/1kAAGXwVLHnqmGaYKKFkwWcAJUUF-CjyU?usp=drive_link 
2. Place the downloaded data in the `output_data` folder

## Usage

### Create a data set


- **Quick example:**
```bash
# Process batch 0 (files 1–100)
python "rotate_STEP_and save_png_all.py" -n 5 --output-folder "output_data/iso_5"
```

# Process batch 1 (files 101–200) and write images to a custom folder
```bash
python "rotate_STEP_and save_png_all.py" -n 5 --category "Mechanical Engineering" --output-folder "output_data/iso_500_mechanical"
```

### Train the VAE

- **Quick example:**
```bash
python train_vae_image_retrieval.py --folder "output_data/iso_5" --epochs 1 --latent-dim 256 --batch-size 64
```
- **Usage example:**
```bash
python train_vae_image_retrieval.py --folder "output_data/iso_500_mechanical" --epochs 50 --latent-dim 256 --batch-size 64
```

**Parameters:**
- `--folder`: Path to folder containing PNG images (default: `"output_data copy"`)
- `--output-dir`: Directory to save model and embeddings (default: same as `--folder`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 32)
- `--latent-dim`: Dimension of latent space (default: 128)
- `--image-size`: Size to resize images to (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--beta`: KL divergence weight (default: 1.0)
- `--train-split`: Fraction of data used for training (rest is used for test/validation, default: 0.8)



#### Retrieving Similar Images

After training, use the model to find similar images:

- **Quick example:**
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\iso_5\test\51a00400-0573-11ec-9601-06368d9f66a5_isometric_00.png" --output-dir "output_data\iso_5" --top-k 5
```

- **Usage example:**
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Screenshot 2025-12-01 143927.png" --output-dir "output_data/iso_10000_mechanical" --top-k 20 
```

```bash
 C:\sources\Fusion360GalleryDataset>python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\iso_10000_mechanical\test\0cff56ee-053e-11ec-b49c-02ef91e90f5f_isometric_00.png" --output-dir "output_data/iso_10000_mechanical" --top-k 10 
 ```


**Parameters:**
- `--query`: Path to the query image
- `--output-dir`: Directory containing trained model (default: `"vae_retrieval_output"`)
- `--top-k`: Number of similar images to retrieve (default: 5)


___

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


