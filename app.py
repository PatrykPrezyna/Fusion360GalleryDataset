"""
Flask web application for image retrieval using VAE and DINO models.
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel

# Import VAE functions
from train_vae_image_retrieval import load_model_and_embeddings, retrieve_similar_images, ConvVAE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = r'C:\sources\Fusion360GalleryDataset\output_data\query_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for caching models and embeddings
vae_models_cache = {}
dino_model_cache = None
dino_processor_cache = None
pool_embeddings_cache = {}  # Cache embeddings per pool directory


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_dino_model(device="cpu"):
    """Load DINOv2 model (cached globally)."""
    global dino_model_cache, dino_processor_cache
    
    if dino_model_cache is None:
        print("Loading DINOv2 model...")
        dino_processor_cache = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dino_model_cache = AutoModel.from_pretrained("facebook/dinov2-base")
        dino_model_cache = dino_model_cache.to(device)
        dino_model_cache.eval()
        print("DINOv2 model loaded!")
    
    return dino_model_cache, dino_processor_cache


def extract_dino_embeddings(model, processor, image_paths, device="cpu", batch_size=32):
    """Extract DINO embeddings for a list of images."""
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                batch_images.append(Image.new("RGB", (224, 224), color="black"))
        
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.last_hidden_state[:, 0, :]  # CLS token
            image_features = F.normalize(image_features, p=2, dim=1)
            all_embeddings.append(image_features.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings


def get_pool_images(pool_dir):
    """Get all image paths from a pool directory."""
    image_paths = sorted(glob.glob(os.path.join(pool_dir, "*.png"))) + \
                  sorted(glob.glob(os.path.join(pool_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(pool_dir, "*.jpeg")))
    return image_paths


def retrieve_dino_similar(query_image_path, pool_dir, top_k=10, device="cpu"):
    """Retrieve similar images using DINO."""
    model, processor = load_dino_model(device)
    
    # Get pool images
    pool_images = get_pool_images(pool_dir)
    if len(pool_images) == 0:
        return []
    
    # Check cache for embeddings
    cache_key = pool_dir
    if cache_key not in pool_embeddings_cache:
        print(f"Computing DINO embeddings for pool: {pool_dir}")
        pool_embeddings_cache[cache_key] = {
            'embeddings': extract_dino_embeddings(model, processor, pool_images, device),
            'paths': pool_images
        }
    
    cached = pool_embeddings_cache[cache_key]
    embeddings = cached['embeddings']
    image_paths = cached['paths']
    
    # Process query image
    query_img = Image.open(query_image_path).convert("RGB")
    inputs = processor(images=query_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :]
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute similarities
    embeddings_device = embeddings.to(device)
    similarities = (embeddings_device @ query_embedding.t()).squeeze(1)
    
    # Get top-k
    top_k = min(top_k, len(image_paths))
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        results.append({
            'path': image_paths[idx],
            'similarity': float(sim),
            'filename': os.path.basename(image_paths[idx])
        })
    
    return results


def retrieve_vae_similar(query_image_path, pool_dir, top_k=10, device="cpu"):
    """Retrieve similar images using VAE."""
    # Check if pool directory has a trained VAE model
    model_path = os.path.join(pool_dir, "vae_model.pt")
    
    if not os.path.exists(model_path):
        return {
            'error': f'No VAE model found in pool directory. Please train a VAE model first using train_vae_image_retrieval.py with --output-dir "{pool_dir}"'
        }
    
    # Load model and embeddings (cached)
    cache_key = pool_dir
    if cache_key not in vae_models_cache:
        print(f"Loading VAE model from: {pool_dir}")
        model, embeddings, image_paths = load_model_and_embeddings(pool_dir, device=device)
        vae_models_cache[cache_key] = {
            'model': model,
            'embeddings': embeddings,
            'paths': image_paths
        }
    
    cached = vae_models_cache[cache_key]
    
    # Retrieve similar images
    results = retrieve_similar_images(
        query_image_path=query_image_path,
        model=cached['model'],
        embeddings=cached['embeddings'],
        image_paths=cached['paths'],
        top_k=top_k,
        device=device
    )
    
    # Convert to dict format
    formatted_results = []
    for path, similarity in results:
        formatted_results.append({
            'path': path,
            'similarity': float(similarity),
            'filename': os.path.basename(path)
        })
    
    return formatted_results


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/retrieve', methods=['POST'])
def api_retrieve():
    """API endpoint for image retrieval."""
    try:
        data = request.json
        query_path = data.get('query_path')
        pool_dir = data.get('pool_dir')
        method = data.get('method', 'both')  # 'vae', 'dino', or 'both'
        top_k = int(data.get('top_k', 10))
        
        if not query_path or not pool_dir:
            return jsonify({'error': 'Missing query_path or pool_dir'}), 400
        
        if not os.path.exists(query_path):
            return jsonify({'error': f'Query image not found: {query_path}'}), 404
        
        if not os.path.exists(pool_dir):
            return jsonify({'error': f'Pool directory not found: {pool_dir}'}), 404
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = {}
        
        start_time = time.time()
        
        # Run VAE retrieval
        if method in ['vae', 'both']:
            try:
                vae_results = retrieve_vae_similar(query_path, pool_dir, top_k=top_k, device=device)
                if isinstance(vae_results, dict) and 'error' in vae_results:
                    results['vae'] = vae_results
                else:
                    results['vae'] = {'results': vae_results, 'count': len(vae_results)}
            except Exception as e:
                results['vae'] = {'error': str(e)}
        
        # Run DINO retrieval
        if method in ['dino', 'both']:
            try:
                dino_results = retrieve_dino_similar(query_path, pool_dir, top_k=top_k, device=device)
                results['dino'] = {'results': dino_results, 'count': len(dino_results)}
            except Exception as e:
                results['dino'] = {'error': str(e)}
        
        elapsed_time = time.time() - start_time
        
        results['elapsed_time'] = elapsed_time
        results['device'] = device
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'path': filepath,
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/pools', methods=['GET'])
def api_pools():
    """List available pool directories."""
    # Look for pool directories in Test_pools
    base_dir = r'C:\sources\Fusion360GalleryDataset\output_data\Test_pools'
    pools = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if it has images
                images = get_pool_images(item_path)
                if len(images) > 0:
                    pools.append({
                        'name': item,
                        'path': item_path,
                        'image_count': len(images)
                    })
    
    return jsonify({'pools': pools})


@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Serve images from the filesystem."""
    from urllib.parse import unquote
    
    # Decode URL-encoded path
    image_path = unquote(image_path)
    # Normalize path separators
    image_path = os.path.normpath(image_path)
    
    # Security: ensure path exists
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # Security: prevent directory traversal - ensure path is within workspace
    abs_path = os.path.abspath(image_path)
    workspace_root = os.path.abspath('.')
    
    # Check if path is within workspace or output_data/query_images/Test_pools
    allowed_roots = [
        workspace_root,
        os.path.join(workspace_root, 'output_data'),
        r'C:\sources\Fusion360GalleryDataset\output_data\query_images',
        r'C:\sources\Fusion360GalleryDataset\output_data\Test_pools'
    ]
    
    if not any(abs_path.startswith(os.path.abspath(root)) for root in allowed_roots):
        return jsonify({'error': 'Access denied'}), 403
    
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    print("Starting Flask app for image retrieval...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
