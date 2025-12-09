# Image Retrieval Web Application

A web-based interface for image retrieval using VAE and DINO models.

## Features

- **Query Image Upload**: Upload a query image to search for similar images
- **Pool Selection**: Select from available image pools in the `output_data` directory
- **Dual Model Support**: Run retrieval with VAE, DINO, or both models simultaneously
- **Results Visualization**: View top-k similar images with similarity scores
- **Modern UI**: Clean, responsive interface with real-time results

## Prerequisites

- Python 3.10+
- All dependencies from `requirements.txt` installed
- Trained VAE models (for VAE retrieval) - place `vae_model.pt` in your pool directories
- Image pools in the `output_data` directory

## Installation

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your image pools**:
   - Place image directories in the `output_data` folder
   - For VAE retrieval: Ensure each pool directory contains a trained `vae_model.pt` file
   - For DINO retrieval: No additional setup needed (uses pretrained model)

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser**:
   Navigate to `http://localhost:5000`

## Usage

1. **Upload Query Image**:
   - Click "Choose File" and select an image (PNG, JPG, or JPEG)
   - A preview will appear

2. **Select Image Pool**:
   - Choose a pool directory from the dropdown
   - Pools are automatically discovered from `output_data` folder
   - Click "🔄 Refresh Pools" to reload available pools

3. **Choose Retrieval Method**:
   - **Both (VAE + DINO)**: Run both models and compare results side-by-side
   - **VAE Only**: Use trained VAE model (requires `vae_model.pt` in pool directory)
   - **DINO Only**: Use pretrained DINOv2 model

4. **Set Top-K**:
   - Specify how many similar images to retrieve (default: 10, max: 50)

5. **Run Search**:
   - Click "🔍 Search Similar Images"
   - Wait for processing (first run may take longer as models load)
   - Results will appear below with similarity scores

## API Endpoints

### `GET /`
Main web interface

### `POST /api/upload`
Upload a query image file
- **Request**: `multipart/form-data` with `file` field
- **Response**: `{success: true, path: "...", filename: "..."}`

### `POST /api/retrieve`
Run image retrieval
- **Request**: JSON
  ```json
  {
    "query_path": "path/to/query/image.png",
    "pool_dir": "path/to/pool/directory",
    "method": "both|vae|dino",
    "top_k": 10
  }
  ```
- **Response**: JSON with results
  ```json
  {
    "vae": {"results": [...], "count": 10},
    "dino": {"results": [...], "count": 10},
    "elapsed_time": 2.34,
    "device": "cuda"
  }
  ```

### `GET /api/pools`
List available image pools
- **Response**: `{pools: [{name: "...", path: "...", image_count: 100}]}`

### `GET /api/image/<path>`
Serve images from filesystem
- Returns image file for display in results

## Notes

- **First Run**: DINO model will be downloaded from Hugging Face on first use (~330MB)
- **Caching**: Models and embeddings are cached in memory for faster subsequent queries
- **VAE Models**: Each pool directory needs its own trained VAE model (`vae_model.pt`)
- **Performance**: GPU acceleration is used automatically if CUDA is available

## Troubleshooting

**"No pools found"**:
- Ensure `output_data` directory exists and contains image folders
- Check that image folders contain PNG/JPG/JPEG files

**"No VAE model found"**:
- Train a VAE model for your pool using `train_vae_image_retrieval.py`
- Place `vae_model.pt` in the pool directory

**Images not displaying**:
- Check file paths are accessible
- Ensure images are in supported formats (PNG, JPG, JPEG)

**Slow performance**:
- First query loads models (slower)
- Subsequent queries use cached models (faster)
- Use GPU if available for better performance

## Example Workflow

1. Train a VAE model for your image pool:
   ```bash
   python train_vae_image_retrieval.py --folder "output_data/my_pool" --epochs 30
   ```

2. Start the web app:
   ```bash
   python app.py
   ```

3. Open browser to `http://localhost:5000`

4. Upload a query image, select your pool, and run retrieval!
