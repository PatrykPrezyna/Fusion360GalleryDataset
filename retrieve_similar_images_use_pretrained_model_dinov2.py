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
    
    # Calculate DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    # Calculate ideal DCG (IDCG) - all relevant items at the top
    ideal_relevances = [1.0] * min(total_relevant, k) + [0.0] * max(0, k - total_relevant)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def load_pretrained_model(device="cpu"):
    """
    Load the pretrained DINOv2 model and processor from Hugging Face.
    
    DINOv2 (Data-Efficient Image Transformer v2) is excellent for geometric and
    structural features, making it ideal for CAD part retrieval tasks where
    rotation invariance is important.
    """
    print("Loading pretrained DINOv2 model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def extract_image_embeddings(model, processor, image_paths, device="cpu", batch_size=32):
    """
    Extract embeddings for all images in the directory.
    
    Args:
        model: DINOv2 model
        processor: DINOv2 processor
        image_paths: List of image paths
        device: Device to run inference on
        batch_size: Batch size for processing
    
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
                batch_images.append(Image.new("RGB", (224, 224), color="black"))
        
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


def retrieve_similar_images_pretrained(
    query_image_path: str,
    model,
    processor,
    embeddings: torch.Tensor,
    image_paths: list,
    top_k: int = 5,
    device: str = "cpu"
):
    """
    Retrieve similar images using pretrained DINOv2 model.
    
    Args:
        query_image_path: Path to query image
        model: DINOv2 model
        processor: DINOv2 processor
        embeddings: Precomputed embeddings for all images (N, embedding_dim)
        image_paths: List of image paths corresponding to embeddings
        top_k: Number of similar images to retrieve
        device: Device to run inference on
    
    Returns:
        List of (image_path, similarity_score) tuples
    """
    # Load and process query image
    try:
        query_img = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load query image: {e}")
    
    # Extract query embedding
    inputs = processor(images=query_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token as the query embedding
        query_embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Move embeddings to same device
    embeddings = embeddings.to(device)
    
    # Compute cosine similarity
    similarities = (embeddings @ query_embedding.t()).squeeze(1)
    
    # Get top-k most similar (excluding the query image itself if it's in the pool)
    top_k = min(top_k, len(image_paths))
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        results.append((image_paths[idx], float(sim)))
    
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
    model, processor = load_pretrained_model(device=device)
    
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
    
    # Load or compute embeddings (only once)
    embeddings_loaded_from_cache = False
    if args.cache_embeddings and os.path.exists(args.cache_embeddings):
        if benchmark_timer:
            benchmark_timer.start_embedding_extraction()
        print(f"Loading cached embeddings from {args.cache_embeddings}...")
        cache_load_start = time.time()
        embeddings = torch.load(args.cache_embeddings, map_location="cpu")
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
                print("Cached embeddings match current image set.")
            else:
                print("Warning: Cached embeddings don't match current image set. Recomputing...")
                embeddings_loaded_from_cache = False
                if benchmark_timer:
                    benchmark_timer.start_embedding_extraction()
                embeddings = extract_image_embeddings(model, processor, image_paths, device=device)
                if args.cache_embeddings:
                    torch.save(embeddings, args.cache_embeddings)
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
        embeddings = extract_image_embeddings(model, processor, image_paths, device=device)
        if args.cache_embeddings:
            torch.save(embeddings, args.cache_embeddings)
            print(f"Saved embeddings to {args.cache_embeddings}")
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
    
    # Process each query image
    for idx, query_image in enumerate(query_images, 1):
        print(f"\n[{idx}/{len(query_images)}] Processing: {os.path.basename(query_image)}")
        print("-" * 60)
        
        # Retrieve similar images
        results = retrieve_similar_images_pretrained(
            query_image_path=query_image,
            model=model,
            processor=processor,
            embeddings=embeddings,
            image_paths=image_paths,
            top_k=args.top_k,
            device=device,
        )
        
        # Extract base ID from query image
        query_base_id = extract_base_id(query_image)
        
        # Count total relevant in entire pool (all images with same base ID)
        total_relevant = 0
        relevant_paths_set = set()
        for path in image_paths:
            if extract_base_id(path) == query_base_id:
                relevant_paths_set.add(path)
        total_relevant = len(relevant_paths_set)
        
        # Compute full rankings to get match ranks
        query_img = Image.open(query_image).convert("RGB")
        inputs = processor(images=query_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Compute cosine similarities for all images
        embeddings_device = embeddings.to(device)
        all_similarities = (embeddings_device @ query_embedding.t()).squeeze(1)
        
        # Sort all similarities to get full rankings
        sorted_indices = torch.argsort(all_similarities, descending=True)
        query_path_norm = os.path.normpath(query_image)
        
        # Create a mapping of path -> rank (1-indexed, excluding query itself)
        # Also collect all matches with their similarities
        path_to_rank = {}
        all_matches = []  # List of (path, similarity, rank) for all matches
        rank = 1
        for idx in sorted_indices:
            path = image_paths[idx.item()]
            path_norm = os.path.normpath(path)
            # Skip if it's the same image
            if path_norm != query_path_norm:
                path_to_rank[path] = rank
                similarity_val = all_similarities[idx.item()].item()
                result_base_id = extract_base_id(path)
                if result_base_id == query_base_id:
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
        query_name = os.path.splitext(os.path.basename(query_image))[0]
        save_path = os.path.join(results_folder, f"{query_name}_retrieval_results_dino.png")
        
        # Display images visually
        display_retrieval_results(
            query_image_path=query_image,
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
    print(f"Number of queries processed: {len(query_images)}")
    if len(query_images) > 0:
        avg_time_per_query = total_retrieval_time / len(query_images)
        print(f"Average time per query: {avg_time_per_query:.2f} seconds")
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
    
    # If processing a folder, combine all results at the end
    if len(query_images) > 1:
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


def display_retrieval_results(query_image_path: str, results: list, save_path: str = None, display_size: int = 256, ndcg_score: float = None):
    """
    Display the query image and retrieved similar images in a grid.
    
    Args:
        query_image_path: Path to the query image
        results: List of (image_path, similarity_score, is_match, match_rank) tuples
        save_path: Optional path to save the visualization
        display_size: Size to display each image
        ndcg_score: Optional NDCG score to display in title
    """
    num_results = len(results)
    
    # Create figure with grid layout
    # Layout: Query image on left, similar images in a row on the right
    # Increase height slightly to accommodate title spacing
    fig = plt.figure(figsize=(display_size * (num_results + 1) / 100, display_size / 100 * 1.15))
    gs = gridspec.GridSpec(1, num_results + 1, figure=fig, wspace=0.1, hspace=0.1, top=0.88, bottom=0.05)
    
    # Load and display query image
    try:
        query_img = Image.open(query_image_path).convert("RGB")
        query_img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
        
        ax_query = fig.add_subplot(gs[0, 0])
        ax_query.imshow(query_img)
        ax_query.set_title("Orygin", fontsize=12, fontweight='bold')
        ax_query.axis('off')
    except Exception as e:
        print(f"Warning: Could not load query image: {e}")
        return
    
    # Load and display similar images
    for idx, result in enumerate(results, 1):
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
            img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            ax = fig.add_subplot(gs[0, idx])
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
            ax.set_title(f"#{idx}\n{similarity:.3f}{rank_text}{match_text}",
                         fontsize=10, pad=5, color=title_color,
                         fontweight='bold' if is_match else 'normal')
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Create empty subplot if image fails to load
            ax = fig.add_subplot(gs[0, idx])
            ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    # Create title with NDCG score if provided
    title = "Image Retrieval Results"
    if ndcg_score is not None:
        title += f" (NDCG@{len(results)}: {ndcg_score:.4f})"
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()

