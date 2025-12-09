
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
python "rotate_STEP_and save_png_all.py" --input-folder "C:\sources\Fusion360GalleryDataset\output_data\test_query\R1" --output-folder "C:\sources\Fusion360GalleryDataset\output_data\test_query\R1\new"
```

```bash
# Process STEP files from a custom input folder (default is "input_data")
python "rotate_STEP_and save_png_all.py"  --input-folder "C:\sources\Fusion360GalleryDataset\output_data\nut_14_views" --output-folder "C:\sources\Fusion360GalleryDataset\output_data\nut_14_views/new"
```

**Parameters:**
- `-n, --num-parts`: Maximum number of STEP files (parts) to process. If omitted, all found STEP files are processed.
- `-o, --output-folder`: Output folder to save the images (default: `"output_data"`)
- `-i, --input-folder`: Input folder containing STEP files (default: `"input_data"`)
- `-c, --category`: Optional category filter (e.g., 'Mechanical Engineering'). If specified, only processes STEP files from assemblies with this category. If omitted, all STEP files are processed.

**Key Python Libraries Used:**
- **OpenCascade (pythonocc-core)**: The main library for handling 3D CAD files. It provides:
  - `read_step_file()`: Loads STEP files into 3D geometry objects
  - `init_display()`: Creates a 3D rendering window/viewer
  - `BRepBuilderAPI_Transform`: Performs 3D rotations and transformations
  - `gp_Trsf`, `gp_Ax1`, `gp_Dir`: Geometry primitives for defining rotation axes and transformations
- **Pillow (PIL)**: Image processing library used to:
  - Convert rendered images to grayscale (`img.convert('L')`)
  - Resize images to a standardized size (515x512 pixels) using high-quality resampling
- **Standard libraries**: `os` (file system operations), `json` (reading assembly metadata), `argparse` (command-line arguments), `math` (rotation calculations)

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
python "rotate_STEP_and save_png_all.py" -n 5 --input-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query\oryginal_images" --output-folder "C:\sources\Fusion360GalleryDataset\output_data\Test_query\R2"
```


After initial Installation:
```bash
conda activate fusion360
```

RETRIVE:
___
VAE
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\R2" --output-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool" --top-k 10

```

___
VAE with rotations
```bash
python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\R1" --output-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_200" --top-k 10

```

___
CLIP
```bash
python retrieve_similar_images_use_pretrained_model.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos" --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool" --top-k 10  --cache-embeddings "embeddings.pt"  --cache-paths "image_paths.txt"
```

___
DINOv2
```bash
python retrieve_similar_images_use_pretrained_model_dinov2.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt"
```

python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt"

python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_14_views.pt" --cache-paths "image_paths.txt"

```bash
python retrieve_similar_images_use_pretrained_model_dinov2.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2_1_1"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_14_views.pt" --cache-paths "image_paths.txt"
```

explainability

python visualize_similarity_explainability.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2\865ce94a-0545-11ec-a85e-020a4e46e3ef__foto_4.jpeg" --target "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views\865ce94a-0545-11ec-a85e-020a4e46e3ef_back_01.png" --output "explainability_result.png"



now here with creating a centroid
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt"
```

```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_14_views.pt" --cache-paths "image_paths.txt"
```



```bash
python retrieve_similar_images_use_pretrained_model_dinov2.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\rendering_front"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\14_views_10000_mechanical" --top-k 10  --cache-embeddings "embeddings_dinov2_14_views_10000_mechanical.pt" --cache-paths "image_paths.txt"
```



```bash
python train_vae_image_retrieval_rotations.py     --train-folder output_data/14_views_10000_mechanical 
    --test-folder output_data/14_views_10000_mechanical/test 
    --epochs 30 
    --contrastive-weight 0.1
```

do it in bathces

```bash
python batch_retrieve_similar_images.py --query-folder "output_data/Test_query/R1" --output-dir "output_data/Test_pool_VAE_with_rotations_1p" --top-k 10 --method vae
```

python train_vae_image_retrieval_rotations.py 
    --train-folder "output_data/14_views_10000_mechanical" 
    --test-folder "output_data/14_views_10000_mechanical/test" 
    --output-dir "output_data/vae_quick_test" 
    --epochs 10 
    --batch-size 32 
    --data-percentage 0.5 
    --random-seed 42


    final 


1_1
python retrieve_similar_images_use_pretrained_model_dinov2_centroid_1_to_many.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2_1_1"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt"



1_4
python retrieve_similar_images_use_pretrained_model_dinov2_centroid_1_to_many.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2_1_1"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_14_views.pt" --cache-paths "image_paths.txt"

4_1
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt"

    4_4
    python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_14_views" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_14_views.pt" --cache-paths "image_paths.txt"


1_1 VAE

python retrieve_similar_images.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\R1"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool" --top-k 10  --cache-embeddings "embeddings_vea.pt" --cache-paths "image_paths.txt"

hyper

1_1
python retrieve_similar_images_use_pretrained_model_dinov2_centroid_hyper.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINO_250" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt" --pooling-strategy weighted_mean

python retrieve_similar_images_use_pretrained_model_dinov2_centroid_hyper.py --query "C:\sources\Fusion360GalleryDataset\output_data\Test_query\fotos_2"     --image-dir "C:\sources\Fusion360GalleryDataset\output_data\Test_pool_DINOs" --top-k 10  --cache-embeddings "embeddings_dinov2_Test_pool_DINO_250.pt" --cache-paths "image_paths.txt" --image-size 256 

python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --model-variant large \
    --pooling-strategy weighted_mean \
    --temperature 0.8 \
    --batch-size 64 \
    --top-k 10