
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

### Original Data

To work with assembly data, download the dataset using the provided tool:

1. Follow the instructions in [`tools/assembly_download`](tools/assembly_download)
2. Place the downloaded data in the `input_data` folder


### Pretrained data

1. Ask Patryk for acces to the drive: https://drive.google.com/drive/folders/1kAAGXwVLHnqmGaYKKFkwWcAJUUF-CjyU?usp=drive_link 
2. Place the downloaded data in the `output_data` folder

## Usage

### (optional) Create a data set


- **Quick example:**
```bash
# Process batch 0 (files 1–100)
python "rotate_STEP_and save_png_all.py" -n 5 --output-folder "output_data/iso_5"
```

# Process batch 1 (files 101–200) and write images to a custom folder
```bash
python "rotate_STEP_and save_png_all.py" -n 5 --category "Mechanical Engineering" --output-folder "output_data/iso_500_mechanical"
```

```bash
# Process STEP files from a custom input folder (default is "input_data")
python "rotate_STEP_and save_png_all.py" -n 5 --input-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query_2" --output-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query_2/new"
```

**Parameters:**
- `-n, --num-parts`: Maximum number of STEP files (parts) to process. If omitted, all found STEP files are processed.
- `-o, --output-folder`: Output folder to save the images (default: `"output_data"`)
- `-i, --input-folder`: Input folder containing STEP files (default: `"input_data"`)
- `-c, --category`: Optional category filter (e.g., 'Mechanical Engineering'). If specified, only processes STEP files from assemblies with this category. If omitted, all STEP files are processed.

### (optional) Train the VAE

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
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\2c419f18-05b8-11ec-916a-061e4e83ef1b_isometric_00.png.png" --output-dir "output_data\iso_5" --top-k 5
```

- **Usage example:**
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Screenshot 2025-12-01 143927.png" --output-dir "output_data/iso_10000_mechanical" --top-k 20 
```

```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\2cf4c2de-05b8-11ec-8876-061e4e83ef1b_left_10_01.png" --output-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool" --top-k 10 
 ```


**Parameters:**
- `--query`: Path to the query image
- `--output-dir`: Directory containing trained model (default: `"vae_retrieval_output"`)
- `--top-k`: Number of similar images to retrieve (default: 5)


Here are some test you might want to do:

0) You can use this script to rotete the query images
```bash
python "rotate_STEP_and save_png_all.py" -n 5 --input-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query_2" --output-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query_2/new"
```

1) Than you can test the score for each of them and document 
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\98128fa4-0550-11ec-b4fe-0ac51587b959_left_10_01.png" --output-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool" --top-k 10
```
