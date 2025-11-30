# Fusion 360 Gallery Dataset
![Fusion 360 Gallery Dataset](docs/images/fusion_gallery_mosaic.jpg)

The *Fusion 360 Gallery Dataset* contains rich 2D and 3D geometry data derived from parametric CAD models. The dataset is produced from designs submitted by users of the CAD package [Autodesk Fusion 360](https://www.autodesk.com/products/fusion-360/overview) to the [Autodesk Online Gallery](https://gallery.autodesk.com/fusion360). The dataset provides valuable data for learning how people design, including sequential CAD design data, designs segmented by modeling operation, and assemblies containing hierarchy and joint connectivity information.

## Datasets
From the approximately 20,000 designs available we derive several datasets focused on specific areas of research. Currently the following data subsets are available, with more to be released on an ongoing basis.

### [Assembly Dataset](docs/assembly.md) - NEW!
Multi-part CAD assemblies containing rich information on joints, contact surfaces, holes, and the underlying assembly graph structure.

![Fusion 360 Gallery Assembly Dataset](docs/images/assembly_mosaic.jpg)


### [Reconstruction Dataset](docs/reconstruction.md)
Sequential construction sequence information from a subset of simple 'sketch and extrude' designs.

![Fusion 360 Gallery Reconstruction Dataset](docs/images/reconstruction_teaser.jpg)

### [Segmentation Dataset](docs/segmentation.md)

A segmentation of 3D models based on the modeling operation used to create each face, e.g. Extrude, Fillet, Chamfer etc...

![Fusion 360 Gallery Segmentation Dataset](docs/images/segmentation_example.jpg)


## Publications
Please cite the relevant paper below if you use the Fusion 360 Gallery dataset in your research.

### Assembly Dataset
[JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints](https://arxiv.org/abs/2111.12772)

```
@article{willis2021joinable,
  title={JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints},
  author={Willis, Karl DD and Jayaraman, Pradeep Kumar and Chu, Hang and Tian, Yunsheng and Li, Yifei and Grandi, Daniele and Sanghi, Aditya and Tran, Linh and Lambourne, Joseph G and Solar-Lezama, Armando and Matusik, Wojciech},
  journal={arXiv preprint arXiv:2111.12772},
  year={2021}
}
```

### Reconstruction Dataset
[Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Construction from Human Design Sequences](https://arxiv.org/abs/2010.02392)
```
@article{willis2020fusion,
    title={Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Construction from Human Design Sequences},
    author={Karl D. D. Willis and Yewen Pu and Jieliang Luo and Hang Chu and Tao Du and Joseph G. Lambourne and Armando Solar-Lezama and Wojciech Matusik},
    journal={ACM Transactions on Graphics (TOG)},
    volume={40},
    number={4},
    year={2021},
    publisher={ACM New York, NY, USA}
}
```

### Segmentation Dataset
[BRepNet: A Topological Message Passing System for Solid Models](https://arxiv.org/abs/2104.00706)
```
@inproceedings{lambourne2021brepnet,
    author    = {Lambourne, Joseph G. and Willis, Karl D.D. and Jayaraman, Pradeep Kumar and Sanghi, Aditya and Meltzer, Peter and Shayani, Hooman},
    title     = {BRepNet: A Topological Message Passing System for Solid Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12773-12782}
}
```

## Download

| Dataset | Designs | Documentation | Download | Paper | Code |
| - | - | - | - | - | - |
| Assembly | 8,251 assemblies / 154,468 parts  | [Documentation](docs/assembly.md) | [Instructions](tools/assembly_download) | [Paper](https://arxiv.org/abs/2111.12772) | [Code](tools) |
| Assembly - Joint | 32,148 joints / 23,029 parts | [Documentation](docs/assembly_joint.md) | [j1.0.0 - 2.8 GB](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/assembly/j1.0.0/j1.0.0.7z) | [Paper](https://arxiv.org/abs/2111.12772) | [Code](https://github.com/AutodeskAILab/JoinABLe) |
| Reconstruction | 8,625 sequences | [Documentation](docs/reconstruction.md) | [r1.0.1 - 2.0 GB](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip) | [Paper](https://arxiv.org/abs/2010.02392) | [Code](tools) |
| Segmentation |  35,680 parts | [Documentation](docs/segmentation.md) | [s2.0.1 - 3.1 GB](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1.zip) | [Paper](https://arxiv.org/abs/2104.00706) | [Code](https://github.com/AutodeskAILab/BRepNet)

### Additional Downloads
- **Reconstruction Dataset Extrude Volumes** [(r1.0.1 - 152 MB)](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1_extrude_tools.zip): The extrude volumes for each extrude operation in the design timeline.
- **Reconstruction Dataset Face Extrusion Sequences** [(r1.0.1 - 41MB)](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1_regraph_05.zip): The pre-processed face extrusion sequences used to train our [reconstruction network](tools/regraphnet).
- **Segmentation Extended STEP Dataset** [(s2.0.1 - 483 MB)](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1_extended_step.zip): An extended collection of 42,912 STEP files with associated segmentation information.  This include all STEP data from s2.0.0 along with additional files for which triangle meshes with close to 2500 edges could not be created. 

## Tools
As part of the dataset we provide various tools for working with the data. These tools leverage the [Fusion 360 API](http://help.autodesk.com/view/fusion360/ENU/?guid=GUID-7B5A90C8-E94C-48DA-B16B-430729B734DC) to perform operations such as geometry reconstruction, traversing B-Rep data structures, and conversion to other formats. More information can be found in the [tools directory](tools).


## License
Please refer to the [dataset license](LICENSE.md).

---

## Getting Started

This section provides step-by-step instructions to set up and use this repository.

### Prerequisites

- **Python 3.10 or 3.11** (recommended)
- **Conda** (for managing Python environments)
- **Git** (to clone the repository)

### Installation

#### Option 1: Using Conda (Recommended)

1. **Create a conda environment:**
   ```bash
   conda create -n fusion-occ -c conda-forge python=3.10 pythonocc-core
   conda activate fusion-occ
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

#### Programmatic Usage

You can also use the VAE in your own Python code:

```python
from train_vae_image_retrieval import load_model_and_embeddings, retrieve_similar_images

# Load trained model
model, embeddings, image_paths = load_model_and_embeddings("vae_retrieval_output")

# Find similar images
results = retrieve_similar_images(
    query_image_path="path/to/query.png",
    model=model,
    embeddings=embeddings,
    image_paths=image_paths,
    top_k=5
)

# Print results
for path, similarity in results:
    print(f"{path}: {similarity:.4f}")
```

### Other Scripts

#### Working with STEP Files

1. **Create assembly visualization:**
   ```bash
   python create_assembly.py
   ```

2. **Read and process STEP files:**
   ```bash
   python read_STEP.py
   ```

3. **Rotate STEP files and save as PNG:**
   ```bash
   python rotate_STEP_and save_png_all.py
   ```

#### Viewing Point Clouds

- **View STEP point cloud:**
  ```bash
  python view_step_pointcloud.py
  ```

- **Compare STEP point clouds:**
  ```bash
  python compare_step_pointclouds.py
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

---

## Project Structure

```
Fusion360GalleryDataset/
├── train_vae_image_retrieval.py    # VAE training script
├── retrieve_similar_images.py      # Image retrieval script
├── create_assembly.py              # Assembly visualization
├── read_STEP.py                    # STEP file processing
├── rotate_STEP_and save_png_all.py # STEP rotation and PNG export
├── view_step_pointcloud.py         # Point cloud viewer
├── compare_step_pointclouds.py     # Point cloud comparison
├── input_data/                     # Input dataset folder
├── output_data/                    # Output folder for processed data
├── vae_retrieval_output/           # VAE model and embeddings
├── tools/                          # Additional tools and utilities
└── docs/                           # Documentation
```

---

## Additional Resources

- **Dataset Documentation:** See [`docs/`](docs/) for detailed dataset documentation
- **Tools Documentation:** See [`tools/README.md`](tools/README.md) for tool-specific documentation
- **Assembly Dataset:** See [`docs/assembly.md`](docs/assembly.md) for assembly dataset details