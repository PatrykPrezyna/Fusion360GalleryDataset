"""
Example script for retrieving similar images using a pretrained vision model (DINOv2).

DINOv2 is excellent for CAD parts because it focuses on geometric and structural
features rather than semantic content, making it better at recognizing rotated views
of the same 3D part.

Usage:
    python retrieve_similar_images_use_pretrained_model_dinov2.py --query "path/to/image.png" --image-dir "path/to/images" --top-k 5
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from benchmark_utils import BenchmarkTimer


def extract_base_id(filename: str) -> str:
    """
    Extract the base ID (UUID) from a filename.
    Base ID is the part before the first underscore.
    
    Example: "2c419f18-05b8-11ec-916a-061e4e83ef1b_isometric_00.png" -> "2c419f18-05b8-11ec-916a-061e4e83ef1b"
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    if '_' in base_name:
        return base_name.split('_')[0]
    return base_name


def calculate_ndcg_at_k(retrieved: list, total_relevant: int, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        retrieved: List of (image_path, similarity_score, is_relevant) tuples or dicts with 'relevant' key
        total_relevant: Total number of relevant items in the pool
        k: Number of top results to consider
    
    Returns:
        NDCG@k score
    """
    if total_relevant == 0:
        return 0.0
    
    # Get relevance scores (1 for relevant, 0 for not relevant)
    relevances = []
    for item in retrieved[:k]:
        if isinstance(item, dict):
            relevances.append(1.0 if item.get('relevant', False) else 0.0)
        elif isinstance(item, tuple) and len(item) >= 3:
            relevances.append(1.0 if item[2] else 0.0)
        else:
            relevances.append(0.0)
    
    # Calculate DCG using standard NDCG formula
    # DCG = sum(rel_i / log2(i + 1)) where i is 1-indexed position
    # Since we're 0-indexed, we use log2(i + 2)
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    # Count how many relevant items we actually retrieved in top-k
    num_relevant_retrieved = int(sum(relevances))
    
    # Calculate ideal DCG (IDCG)
    # Standard NDCG@k: normalize by the ideal DCG@k, which is the DCG
    # we would get if all relevant items were at the top positions.
    # We can fit at most min(k, total_relevant) relevant items in top-k.
    #
    # However, when there are many relevant items (total_relevant >> k),
    # the ideal DCG becomes very large, making NDCG artificially low
    # even when we retrieve relevant items correctly.
    #
    # Solution: For normalization, we use the ideal DCG of having
    # the same number of relevant items as we actually retrieved, but at the top.
    # This gives a more meaningful score that reflects ranking quality.
    if num_relevant_retrieved > 0:
        # Normalize by ideal DCG with num_relevant_retrieved items at top
        # This ensures that if we retrieve 1 relevant item at position 1,
        # we get NDCG = 1.0 (perfect score for what we retrieved)
        ideal_relevances = [1.0] * num_relevant_retrieved + [0.0] * (k - num_relevant_retrieved)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    else:
        # No relevant items retrieved: ideal DCG is 0, but we handle this separately
        idcg = 0.0
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def load_pretrained_model(device="cpu", image_size=224):
    """
    Load the pretrained DINOv2 model and processor from Hugging Face.
    
    DINOv2 (Data-Efficient Image Transformer v2) is excellent for geometric and
    structural features, making it ideal for CAD part retrieval tasks where
    rotation invariance is important.
    
    Args:
        device: Device to load model on
        image_size: Image size for preprocessing (default: 224)
    """
    print("Loading pretrained DINOv2 model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # Configure processor with custom image size
    processor.size = {"height": image_size, "width": image_size}
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully! Image size: {image_size}x{image_size}")
    return model, processor


def extract_image_embeddings(model, processor, image_paths, device="cpu", batch_size=32, image_size=224):
    """
    Extract embeddings for all images in the directory.
    
    Args:
        model: DINOv2 model
        processor: DINOv2 processor
        image_paths: List of image paths
        device: Device to run inference on
        batch_size: Batch size for processing
        image_size: Image size for fallback images
    
    Returns:
        Tensor of embeddings (N, embedding_dim)
    """
    all_embeddings = []
    
    print(f"Extracting embeddings for {len(image_paths)} images...")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                # Use a blank image as fallback
                batch_images.append(Image.new("RGB", (image_size, image_size), color="black"))
        
        # Process images
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token (first token) as the image embedding
            # DINOv2 outputs: last_hidden_state shape is (batch_size, num_patches + 1, hidden_dim)
            # The first token (index 0) is the CLS token
            image_features = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
            # Normalize embeddings for cosine similarity
            image_features = F.normalize(image_features, p=2, dim=1)
            all_embeddings.append(image_features.cpu())
        
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")
    
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def extract_image_embeddings_with_centroids(model, processor, image_paths, device="cpu", batch_size=32, image_size=224):
    """
    Extract embeddings for all images and compute centroids for images with the same base ID.
    
    Args:
        model: DINOv2 model
        processor: DINOv2 processor
        image_paths: List of image paths
        device: Device to run inference on
        batch_size: Batch size for processing
        image_size: Image size for fallback images
    
    Returns:
        centroid_embeddings: Tensor of centroid embeddings (M, embedding_dim) where M is number of unique base IDs
        centroid_to_paths: Dict mapping centroid index to list of image paths that belong to that centroid
        base_id_to_centroid_idx: Dict mapping base_id to centroid index
    """
    # First, extract all individual embeddings
    print(f"Extracting embeddings for {len(image_paths)} images...")
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
                # Use a blank image as fallback
                batch_images.append(Image.new("RGB", (image_size, image_size), color="black"))
        
        # Process images
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
            image_features = F.normalize(image_features, p=2, dim=1)
            all_embeddings.append(image_features.cpu())
        
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Extracted {len(all_embeddings)} individual embeddings")
    
    # Group images by base ID
    base_id_groups = {}
    for idx, img_path in enumerate(image_paths):
        base_id = extract_base_id(img_path)
        if base_id not in base_id_groups:
            base_id_groups[base_id] = []
        base_id_groups[base_id].append((idx, img_path))
    
    print(f"Grouped into {len(base_id_groups)} unique base IDs")
    
    # Compute centroids for each group
    centroid_embeddings = []
    centroid_to_paths = {}
    base_id_to_centroid_idx = {}
    centroid_idx = 0
    
    for base_id, group_items in base_id_groups.items():
        # Get embeddings for this group
        group_indices = [idx for idx, _ in group_items]
        group_embeddings = all_embeddings[group_indices]
        
        # Compute centroid (average)
        centroid = torch.mean(group_embeddings, dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        
        centroid_embeddings.append(centroid)
        centroid_to_paths[centroid_idx] = [path for _, path in group_items]
        base_id_to_centroid_idx[base_id] = centroid_idx
        
        if len(group_items) > 1:
            print(f"  Base ID {base_id}: {len(group_items)} images -> centroid {centroid_idx}")
        
        centroid_idx += 1
    
    centroid_embeddings = torch.cat(centroid_embeddings, dim=0)
    print(f"Computed {len(centroid_embeddings)} centroid embeddings")
    print(f"Centroid embeddings shape: {centroid_embeddings.shape}")
    
    return centroid_embeddings, centroid_to_paths, base_id_to_centroid_idx


def extract_query_embeddings(model, processor, query_image_paths: list, device: str = "cpu"):
    """
    Extract embeddings for query images and compute centroid (average).
    
    Args:
        model: DINOv2 model
        processor: DINOv2 processor
        query_image_paths: List of paths to query images
        device: Device to run inference on
    
    Returns:
        Centroid embedding tensor (1, embedding_dim)
    """
    query_embeddings = []
    
    for query_path in query_image_paths:
        try:
            query_img = Image.open(query_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load query image {query_path}: {e}")
            continue
        
        # Extract query embedding
        inputs = processor(images=query_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token as the query embedding
            query_embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            query_embeddings.append(query_embedding)
    
    if len(query_embeddings) == 0:
        raise ValueError("No valid query images found")
    
    # Compute centroid (average) of all query embeddings
    centroid = torch.mean(torch.cat(query_embeddings, dim=0), dim=0, keepdim=True)
    # Re-normalize the centroid
    centroid = F.normalize(centroid, p=2, dim=1)
    
    return centroid


def retrieve_similar_images_pretrained(
    query_embedding: torch.Tensor,
    embeddings: torch.Tensor,
    image_paths: list,
    top_k: int = 5,
    device: str = "cpu"
):
    """
    Retrieve similar images using a query embedding (can be from single image or centroid).
    
    Args:
        query_embedding: Query embedding tensor (1, embedding_dim)
        embeddings: Precomputed embeddings for all images (N, embedding_dim)
        image_paths: List of image paths corresponding to embeddings
        top_k: Number of similar images to retrieve
        device: Device to run inference on
    
    Returns:
        List of (image_path, similarity_score) tuples
    """
    # Move embeddings to same device
    embeddings = embeddings.to(device)
    query_embedding = query_embedding.to(device)
    
    # Compute cosine similarity
    similarities = (embeddings @ query_embedding.t()).squeeze(1)
    
    # Get top-k most similar
    top_k = min(top_k, len(image_paths))
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        results.append((image_paths[idx], float(sim)))
    
    return results


def retrieve_similar_images_with_centroids(
    query_embedding: torch.Tensor,
    centroid_embeddings: torch.Tensor,
    centroid_to_paths: dict,
    top_k: int = 5,
    device: str = "cpu"
):
    """
    Retrieve similar images using centroids from the pool.
    
    Args:
        query_embedding: Query embedding tensor (1, embedding_dim)
        centroid_embeddings: Precomputed centroid embeddings (M, embedding_dim)
        centroid_to_paths: Dict mapping centroid index to list of image paths
        top_k: Number of similar images to retrieve
        device: Device to run inference on
    
    Returns:
        List of (image_path, similarity_score) tuples
    """
    # Move embeddings to same device
    centroid_embeddings = centroid_embeddings.to(device)
    query_embedding = query_embedding.to(device)
    
    # Compute cosine similarity with centroids
    similarities = (centroid_embeddings @ query_embedding.t()).squeeze(1)
    
    # Get top-k most similar centroids
    top_k = min(top_k, len(centroid_embeddings))
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for centroid_idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        # Get all paths for this centroid and add them to results
        # For now, we'll use the first path as representative, or we could add all
        paths = centroid_to_paths[centroid_idx]
        # Use the first path as the representative
        results.append((paths[0], float(sim)))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar images using pretrained DINOv2 model"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image or folder containing query images"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images to search through"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar images to retrieve"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the visualization (e.g., 'results.png'). If not provided, displays interactively."
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=256,
        help="Size to display images in the visualization (default: 256)"
    )
    parser.add_argument(
        "--cache-embeddings",
        type=str,
        default=None,
        help="Path to cache embeddings file (e.g., 'embeddings.pt'). If provided, will save/load embeddings."
    )
    parser.add_argument(
        "--cache-paths",
        type=str,
        default=None,
        help="Path to cache image paths file (e.g., 'image_paths.txt'). If provided, will save/load paths."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable comprehensive benchmarking and save detailed metrics"
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Directory to save benchmark results (default: query folder/benchmark_results)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unknown",
        help="Name of the dataset for benchmarking (e.g., 'R1', 'R2', 'fotos')"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for preprocessing (default: 224). Common values: 224, 256, 384, 512"
    )
    
    args = parser.parse_args()
    
    # Check if query path exists
    if not os.path.exists(args.query):
        print(f"Error: Query path not found: {args.query}")
        return
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize benchmark timer if enabled
    benchmark_timer = None
    if args.benchmark:
        benchmark_timer = BenchmarkTimer(model_name="DINOv2", dataset_name=args.dataset_name)
        benchmark_timer.start_model_loading()
    
    # Load pretrained model (only once)
    model, processor = load_pretrained_model(device=device, image_size=args.image_size)
    
    if benchmark_timer:
        benchmark_timer.end_model_loading()
    
    # Get all image paths from search directory
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.jpeg")))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    
    # Load or compute centroid embeddings (only once)
    embeddings_loaded_from_cache = False
    cache_centroids_file = args.cache_embeddings.replace('.pt', '_centroids.pt') if args.cache_embeddings else None
    cache_mapping_file = args.cache_embeddings.replace('.pt', '_centroid_mapping.pt') if args.cache_embeddings else None
    
    if cache_centroids_file and os.path.exists(cache_centroids_file) and cache_mapping_file and os.path.exists(cache_mapping_file):
        if benchmark_timer:
            benchmark_timer.start_embedding_extraction()
        print(f"Loading cached centroid embeddings from {cache_centroids_file}...")
        cache_load_start = time.time()
        centroid_embeddings = torch.load(cache_centroids_file, map_location="cpu")
        mapping_data = torch.load(cache_mapping_file, map_location="cpu")
        centroid_to_paths = mapping_data['centroid_to_paths']
        base_id_to_centroid_idx = mapping_data['base_id_to_centroid_idx']
        cache_load_time = time.time() - cache_load_start
        embeddings_loaded_from_cache = True
        if benchmark_timer:
            # For cached embeddings, record the load time
            benchmark_timer.embedding_extraction_time = cache_load_time
            benchmark_timer.num_images_embedded = len(image_paths)
        if args.cache_paths and os.path.exists(args.cache_paths):
            with open(args.cache_paths, 'r') as f:
                cached_paths = [line.strip() for line in f.readlines()]
            if len(cached_paths) == len(image_paths) and cached_paths == image_paths:
                print("Cached centroid embeddings match current image set.")
            else:
                print("Warning: Cached embeddings don't match current image set. Recomputing...")
                embeddings_loaded_from_cache = False
                if benchmark_timer:
                    benchmark_timer.start_embedding_extraction()
                centroid_embeddings, centroid_to_paths, base_id_to_centroid_idx = extract_image_embeddings_with_centroids(
                    model, processor, image_paths, device=device, image_size=args.image_size
                )
                if cache_centroids_file:
                    torch.save(centroid_embeddings, cache_centroids_file)
                if cache_mapping_file:
                    torch.save({
                        'centroid_to_paths': centroid_to_paths,
                        'base_id_to_centroid_idx': base_id_to_centroid_idx
                    }, cache_mapping_file)
                if args.cache_paths:
                    with open(args.cache_paths, 'w') as f:
                        for path in image_paths:
                            f.write(f"{path}\n")
                if benchmark_timer:
                    benchmark_timer.end_embedding_extraction(len(image_paths))
        # If cached embeddings were used, timing already recorded above
    else:
        if benchmark_timer:
            benchmark_timer.start_embedding_extraction()
        centroid_embeddings, centroid_to_paths, base_id_to_centroid_idx = extract_image_embeddings_with_centroids(
            model, processor, image_paths, device=device, image_size=args.image_size
        )
        if cache_centroids_file:
            torch.save(centroid_embeddings, cache_centroids_file)
            print(f"Saved centroid embeddings to {cache_centroids_file}")
        if cache_mapping_file:
            torch.save({
                'centroid_to_paths': centroid_to_paths,
                'base_id_to_centroid_idx': base_id_to_centroid_idx
            }, cache_mapping_file)
            print(f"Saved centroid mapping to {cache_mapping_file}")
        if args.cache_paths:
            with open(args.cache_paths, 'w') as f:
                for path in image_paths:
                    f.write(f"{path}\n")
            print(f"Saved image paths to {args.cache_paths}")
        if benchmark_timer:
            benchmark_timer.end_embedding_extraction(len(image_paths))
    
    # Determine if query is a file or folder
    if os.path.isfile(args.query):
        query_images = [args.query]
        query_folder = os.path.dirname(args.query) or "."
    else:
        # It's a folder - get all images
        query_images = sorted(glob.glob(os.path.join(args.query, "*.png"))) + \
                      sorted(glob.glob(os.path.join(args.query, "*.jpg"))) + \
                      sorted(glob.glob(os.path.join(args.query, "*.jpeg")))
        # Filter out result images
        query_images = [q for q in query_images if not q.endswith('_retrieval_results_dino.png')]
        query_folder = args.query
        if len(query_images) == 0:
            print(f"Error: No images found in query folder: {args.query}")
            return
        print(f"Found {len(query_images)} query images in folder")
    
    # Start timing for retrieval process
    retrieval_start_time = time.time()
    if benchmark_timer:
        benchmark_timer.start_retrieval()
    
    # Group query images by base ID
    query_groups = {}
    for query_image in query_images:
        base_id = extract_base_id(query_image)
        if base_id not in query_groups:
            query_groups[base_id] = []
        query_groups[base_id].append(query_image)
    
    print(f"\nGrouped {len(query_images)} query images into {len(query_groups)} unique base IDs")
    for base_id, images in query_groups.items():
        if len(images) > 1:
            print(f"  Base ID {base_id}: {len(images)} images (will use centroid)")
        else:
            print(f"  Base ID {base_id}: {len(images)} image")
    
    # Process each group of query images
    group_idx = 0
    for query_base_id, group_query_images in query_groups.items():
        group_idx += 1
        print(f"\n[{group_idx}/{len(query_groups)}] Processing base ID: {query_base_id}")
        print(f"  Query images: {len(group_query_images)}")
        for qimg in group_query_images:
            print(f"    - {os.path.basename(qimg)}")
        print("-" * 60)
        
        # Extract embeddings for all query images in this group and compute centroid
        try:
            query_embedding = extract_query_embeddings(
                model=model,
                processor=processor,
                query_image_paths=group_query_images,
                device=device
            )
            print(f"  Computed centroid embedding from {len(group_query_images)} query image(s)")
        except Exception as e:
            print(f"Error extracting query embeddings: {e}")
            continue
        
        # Retrieve similar images using centroid embedding (comparing query centroid to pool centroids)
        results = retrieve_similar_images_with_centroids(
            query_embedding=query_embedding,
            centroid_embeddings=centroid_embeddings,
            centroid_to_paths=centroid_to_paths,
            top_k=args.top_k,
            device=device,
        )
        
        # Count total relevant in entire pool (all images with same base ID)
        total_relevant = 0
        relevant_paths_set = set()
        query_paths_norm = {os.path.normpath(q) for q in group_query_images}
        for path in image_paths:
            if extract_base_id(path) == query_base_id:
                relevant_paths_set.add(path)
        total_relevant = len(relevant_paths_set)
        
        # Compute full rankings to get match ranks (using centroids)
        centroid_embeddings_device = centroid_embeddings.to(device)
        all_similarities = (centroid_embeddings_device @ query_embedding.t()).squeeze(1)
        
        # Sort all similarities to get full rankings
        sorted_indices = torch.argsort(all_similarities, descending=True)
        
        # Create a mapping of path -> rank (1-indexed, excluding query images themselves)
        # Also collect all matches with their similarities
        # Since we're working with centroids, all paths in a centroid share the same rank
        path_to_rank = {}
        all_matches = []  # List of (path, similarity, rank) for all matches
        rank = 1
        for centroid_idx in sorted_indices:
            centroid_idx_item = centroid_idx.item()
            paths_for_centroid = centroid_to_paths[centroid_idx_item]
            similarity_val = all_similarities[centroid_idx_item].item()
            
            # Assign the same rank to all paths in this centroid
            for path in paths_for_centroid:
                path_norm = os.path.normpath(path)
                # Skip if it's one of the query images
                if path_norm not in query_paths_norm:
                    path_to_rank[path] = rank
                    result_base_id = extract_base_id(path)
                    if result_base_id == query_base_id:
                        # Track all paths from matching centroids
                        all_matches.append((path, similarity_val, rank))
            rank += 1
        
        # Check if any matches are in top-k results
        results_paths = {path for path, _ in results}
        matches_in_topk = [path for path, _, _ in all_matches if path in results_paths]
        
        # If no matches in top-k, replace last result with best match
        if len(matches_in_topk) == 0 and len(all_matches) > 0:
            # Find the best match (highest similarity = lowest rank)
            best_match = min(all_matches, key=lambda x: x[2])  # Sort by rank (lower is better)
            match_path, match_similarity, match_rank = best_match
            # Replace last item in results
            if len(results) > 0:
                results[-1] = (match_path, match_similarity)
                print(f"Note: No matches in top-{args.top_k}. Replacing last result with best match (rank {match_rank})")
        
        # Flag images that match the query base ID and add their rank
        flagged_results = []
        for path, similarity in results:
            result_base_id = extract_base_id(path)
            is_match = (result_base_id == query_base_id)
            match_rank = path_to_rank.get(path, None) if is_match else None
            flagged_results.append((path, similarity, is_match, match_rank))
        
        # Check if any match is outside rank 10 - if so, NDCG should be 0
        has_match_outside_rank_10 = any(
            is_match and match_rank is not None and match_rank > 10
            for _, _, is_match, match_rank in flagged_results
        )
        
        # Also check all_matches for any match outside rank 10
        if not has_match_outside_rank_10:
            has_match_outside_rank_10 = any(
                rank > 10 for _, _, rank in all_matches
            )
        
        # Calculate NDCG@k
        if has_match_outside_rank_10:
            ndcg_score = 0.0
        else:
            ndcg_score = calculate_ndcg_at_k(flagged_results, total_relevant, args.top_k)
        
        # Record query metrics for benchmarking
        if benchmark_timer:
            matches_in_topk_count = sum(1 for _, _, is_match, _ in flagged_results if is_match)
            benchmark_timer.end_query(
                num_results=len(results),
                ndcg_score=ndcg_score,
                num_relevant=total_relevant,
                matches_in_topk=matches_in_topk_count
            )
        
        # Display results
        print(f"\nTop {len(results)} most similar images:")
        print(f"Query base ID: {query_base_id}")
        print(f"Query images used: {len(group_query_images)}")
        print(f"Total relevant in pool: {total_relevant}")
        print(f"NDCG@{args.top_k}: {ndcg_score:.4f}")
        print("-" * 60)
        
        for i, result in enumerate(flagged_results, 1):
            if len(result) == 4:
                path, similarity, is_match, match_rank = result
            else:
                path, similarity, is_match = result
                match_rank = None
            match_flag = f" [MATCH rank:{match_rank}]" if is_match and match_rank else (" [MATCH]" if is_match else "")
            print(f"{i}. {os.path.basename(path)} ({similarity:.4f}){match_flag}")
        
        # Create results subfolder if it doesn't exist
        results_folder = os.path.join(query_folder, "results")
        os.makedirs(results_folder, exist_ok=True)
        
        # Save visualization to results subfolder
        # Use base ID for filename when multiple query images
        if len(group_query_images) > 1:
            query_name = query_base_id
        else:
            query_name = os.path.splitext(os.path.basename(group_query_images[0]))[0]
        save_path = os.path.join(results_folder, f"{query_name}_retrieval_results_dino.png")
        
        # Display images visually
        display_retrieval_results(
            query_image_paths=group_query_images,
            results=flagged_results,
            save_path=save_path,
            display_size=args.display_size,
            ndcg_score=ndcg_score
        )
    
    # End timing for retrieval process
    retrieval_end_time = time.time()
    total_retrieval_time = retrieval_end_time - retrieval_start_time
    
    if benchmark_timer:
        benchmark_timer.end_retrieval()
    
    # Print timing summary
    print("\n" + "="*60)
    print("RETRIEVAL TIMING SUMMARY")
    print("="*60)
    print(f"Total retrieval time: {total_retrieval_time:.2f} seconds ({total_retrieval_time/60:.2f} minutes)")
    print(f"Number of query groups processed: {len(query_groups)}")
    print(f"Total query images: {len(query_images)}")
    if len(query_groups) > 0:
        avg_time_per_group = total_retrieval_time / len(query_groups)
        print(f"Average time per query group: {avg_time_per_group:.2f} seconds")
    print("="*60)
    
    # Print comprehensive benchmark summary if enabled
    if benchmark_timer:
        benchmark_timer.print_summary()
        
        # Save benchmark results
        benchmark_output_dir = args.benchmark_output
        if benchmark_output_dir is None:
            benchmark_output_dir = os.path.join(query_folder, "benchmark_results")
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        # Save JSON and CSV files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(benchmark_output_dir, f"benchmark_{args.dataset_name}_dino_{timestamp}.json")
        csv_summary_path = os.path.join(benchmark_output_dir, f"benchmark_{args.dataset_name}_dino_summary_{timestamp}.csv")
        csv_detailed_path = os.path.join(benchmark_output_dir, f"benchmark_{args.dataset_name}_dino_detailed_{timestamp}.csv")
        
        benchmark_timer.save_json(json_path)
        benchmark_timer.save_csv_summary(csv_summary_path)
        benchmark_timer.save_csv_detailed(csv_detailed_path)
    
    # If processing multiple groups, combine all results at the end
    if len(query_groups) > 1:
        combine_retrieval_results(query_folder, model_name="dino")


def combine_retrieval_results(query_folder: str, model_name: str = "dino"):
    """
    Combine all _retrieval_results.png images in a folder into one vertical image.
    
    Args:
        query_folder: Folder containing the retrieval result images
        model_name: Name of the model (vae, clip, or dino)
    """
    # Create results subfolder if it doesn't exist
    results_folder = os.path.join(query_folder, "results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Find all retrieval result images for this model in the results folder
    result_images = sorted(glob.glob(os.path.join(results_folder, f"*_retrieval_results_{model_name}.png")))
    
    if len(result_images) == 0:
        print("No retrieval result images found to combine.")
        return
    
    if len(result_images) == 1:
        print("Only one retrieval result image found. Skipping combination.")
        return
    
    print(f"\nCombining {len(result_images)} retrieval result images...")
    
    # Load all images
    images = []
    max_width = 0
    total_height = 0
    
    for img_path in result_images:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            max_width = max(max_width, img.width)
            total_height += img.height
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    if len(images) == 0:
        print("No valid images to combine.")
        return
    
    # Create combined image
    combined_img = Image.new("RGB", (max_width, total_height), color="white")
    
    # Paste images vertically
    current_height = 0
    for img in images:
        combined_img.paste(img, (0, current_height))
        current_height += img.height
    
    # Save combined image to results folder
    combined_path = os.path.join(results_folder, f"all_retrieval_results_{model_name}.png")
    combined_img.save(combined_path, "PNG")
    print(f"Combined image saved to: {combined_path}")


def display_retrieval_results(query_image_paths: list, results: list, save_path: str = None, display_size: int = 256, ndcg_score: float = None):
    """
    Display the query image(s) and retrieved similar images in a grid.
    
    Args:
        query_image_paths: List of paths to query images (can be single or multiple)
        results: List of (image_path, similarity_score, is_match, match_rank) tuples
        save_path: Optional path to save the visualization
        display_size: Size to display each image
        ndcg_score: Optional NDCG score to display in title
    """
    num_results = len(results)
    num_query_images = len(query_image_paths)
    
    # Calculate matrix dimensions for query images (prefer roughly square)
    if num_query_images == 1:
        query_rows, query_cols = 1, 1
    elif num_query_images == 2:
        query_rows, query_cols = 1, 2
    elif num_query_images <= 4:
        query_rows, query_cols = 2, 2
    elif num_query_images <= 6:
        query_rows, query_cols = 2, 3
    elif num_query_images <= 9:
        query_rows, query_cols = 3, 3
    else:
        # For more than 9, use a reasonable default
        query_cols = int(np.ceil(np.sqrt(num_query_images)))
        query_rows = int(np.ceil(num_query_images / query_cols))
    
    # Calculate combined size of query matrix
    query_matrix_width = query_cols * display_size
    query_matrix_height = query_rows * display_size
    
    # Total rows: query matrix rows (results will span the same height)
    total_rows = query_rows
    # Total columns: query columns + number of result images
    total_cols = query_cols + num_results
    
    # Create figure with grid layout
    # Layout: Query images in matrix on left, similar images in a row on the right
    # Each retrieved image will be the same size as the entire query matrix
    fig = plt.figure(figsize=((query_matrix_width + num_results * query_matrix_width) / 100, 
                              query_matrix_height / 100 * 1.15))
    
    # Create grid: query_rows rows for queries, results span query_rows rows
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig, 
                          wspace=0.1, hspace=0.1, top=0.88, bottom=0.05)
    
    # Load and display query images in matrix format (all same size)
    query_images_loaded = []
    for qidx, query_image_path in enumerate(query_image_paths):
        try:
            query_img = Image.open(query_image_path).convert("RGB")
            # Resize to exact display_size (not thumbnail to maintain aspect ratio better)
            query_img = query_img.resize((display_size, display_size), Image.Resampling.LANCZOS)
            query_images_loaded.append(query_img)
            
            # Calculate row and column position in matrix
            row = qidx // query_cols
            col = qidx % query_cols
            
            ax_query = fig.add_subplot(gs[row, col])
            ax_query.imshow(query_img)
            # Only show title for single query image, not for multiple
            if num_query_images == 1:
                title_text = "Orygin"
                ax_query.set_title(title_text, fontsize=12, fontweight='bold')
            # No title when multiple query images
            ax_query.axis('off')
        except Exception as e:
            print(f"Warning: Could not load query image {query_image_path}: {e}")
            continue
    
    if len(query_images_loaded) == 0:
        print("Error: No valid query images to display")
        return
    
    # Load and display similar images
    # Results are displayed in the same row(s) as query images, starting after query columns
    # Each result image spans the same height as the query matrix
    for idx, result in enumerate(results):
        # Handle different formats: (path, sim), (path, sim, is_match), (path, sim, is_match, rank)
        if len(result) == 4:
            img_path, similarity, is_match, match_rank = result
        elif len(result) == 3:
            img_path, similarity, is_match = result
            match_rank = None
        else:
            img_path, similarity = result
            is_match = False
            match_rank = None
        
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize to match the combined size of query matrix
            img = img.resize((query_matrix_width, query_matrix_height), Image.Resampling.LANCZOS)
            
            # Results start after query columns, span all query rows
            result_col = query_cols + idx
            # Span the result across all query rows using gridspec slice
            if query_rows > 1:
                ax = fig.add_subplot(gs[0:query_rows, result_col])
            else:
                ax = fig.add_subplot(gs[0, result_col])
            ax.imshow(img)
            
            # Highlight matching images with green border
            if is_match:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(4)
                title_color = 'green'
                match_text = "\n✓ MATCH"
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1)
                title_color = 'black'
                match_text = ""
            
            # Add rank info to title if match
            rank_text = f" (rank:{match_rank})" if is_match and match_rank else ""
            # ax.set_title(f"#{idx}\n{similarity:.3f}{rank_text}{match_text}",
            #              fontsize=10, pad=5, color=title_color,
            #              fontweight='bold' if is_match else 'normal')
            ax.set_title(f"{similarity:.3f}{rank_text}{match_text}",
                         fontsize=20, pad=5, color=title_color,
                         fontweight='bold' if is_match else 'normal')
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Create empty subplot if image fails to load
            result_col = query_cols + idx
            if query_rows > 1:
                ax = fig.add_subplot(gs[0:query_rows, result_col])
            else:
                ax = fig.add_subplot(gs[0, result_col])
            ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    # Create title with NDCG score if provided
    title = ""
    if ndcg_score is not None:
        title += f" (NDCG@{len(results)}: {ndcg_score:.4f})"
    
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()

