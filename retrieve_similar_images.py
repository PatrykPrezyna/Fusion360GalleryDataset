"""
Example script for retrieving similar images using the trained VAE.

Usage:
    python retrieve_similar_images.py --query "path/to/image.png" --output-dir "vae_retrieval_output" --top-k 5
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
from train_vae_image_retrieval import load_model_and_embeddings, retrieve_similar_images


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


def combine_retrieval_results(query_folder: str, model_name: str = "vae"):
    """
    Combine all _retrieval_results.png images in a folder into one vertical image.
    
    Args:
        query_folder: Folder containing the retrieval result images
        model_name: Name of the model (vae, clip, or dino)
    """
    import glob
    from PIL import Image
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar images using trained VAE"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image or folder containing query images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vae_retrieval_output",
        help="Directory containing trained model and embeddings"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar images to retrieve"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size used during training"
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
    
    args = parser.parse_args()
    
    # Check if query path exists
    if not os.path.exists(args.query):
        print(f"Error: Query path not found: {args.query}")
        return
    
    # Check if model exists
    model_path = os.path.join(args.output_dir, "vae_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_vae_image_retrieval.py")
        return
    
    # Load model and embeddings (only once)
    print(f"Loading model and embeddings from {args.output_dir}...")
    import torch
    import glob
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embeddings, image_paths = load_model_and_embeddings(args.output_dir, device=device)
    print(f"Loaded {len(image_paths)} image embeddings")
    
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
        query_images = [q for q in query_images if not q.endswith('_retrieval_results_vae.png')]
        query_folder = args.query
        if len(query_images) == 0:
            print(f"Error: No images found in query folder: {args.query}")
            return
        print(f"Found {len(query_images)} query images in folder")
    
    # Start timing for retrieval process
    retrieval_start_time = time.time()
    
    # Track metrics across all queries
    all_ndcg_scores = []
    all_avg_similarities = []
    all_match_ranks = []
    # Track all matched parts (not just top-k)
    all_matched_similarities = []
    all_matched_ranks = []
    
    # Process each query image
    for idx, query_image in enumerate(query_images, 1):
        print(f"\n[{idx}/{len(query_images)}] Processing: {os.path.basename(query_image)}")
        print("-" * 60)
        
        # Retrieve similar images
        results = retrieve_similar_images(
            query_image_path=query_image,
            model=model,
            embeddings=embeddings,
            image_paths=image_paths,
            top_k=args.top_k,
            device=device,
            image_size=args.image_size,
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
        import torch
        from PIL import Image as PILImage
        import torchvision.transforms as T
        
        # Compute full similarities for all images
        transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
        ])
        
        query_img = PILImage.open(query_image).convert("RGB")
        query_tensor = transform(query_img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(query_tensor)
            query_embedding = mu
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        query_norm = query_embedding / (query_embedding.norm() + 1e-8)
        
        # Compute cosine similarities for all images
        all_similarities = (embeddings_norm @ query_norm.t()).squeeze(1)
        
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
        
        # Collect metrics for averaging
        all_ndcg_scores.append(ndcg_score)
        
        # Calculate average similarity for this query
        avg_similarity = np.mean([sim for _, sim, _, _ in flagged_results])
        all_avg_similarities.append(avg_similarity)
        
        # Collect match ranks (if any matches found)
        match_ranks_for_query = [rank for _, _, is_match, rank in flagged_results if is_match and rank is not None]
        if match_ranks_for_query:
            all_match_ranks.extend(match_ranks_for_query)
        
        # Collect all matched parts' similarities and ranks (not just top-k)
        if len(all_matches) > 0:
            all_matched_similarities.extend([sim for _, sim, _ in all_matches])
            all_matched_ranks.extend([rank for _, _, rank in all_matches])
        
        # Display results
        print(f"\nTop {len(results)} most similar images:")
        print(f"Query base ID: {query_base_id}")
        print(f"Total relevant in pool: {total_relevant}")
        print(f"NDCG@{args.top_k}: {ndcg_score:.4f}")
        print(f"Average similarity: {avg_similarity:.4f}")
        if match_ranks_for_query:
            avg_rank = np.mean(match_ranks_for_query)
            print(f"Average match rank: {avg_rank:.2f}")
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
        save_path = os.path.join(results_folder, f"{query_name}_retrieval_results_vae.png")
        
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
    
    # Print performance metrics summary
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    if len(all_ndcg_scores) > 0:
        avg_ndcg = np.mean(all_ndcg_scores)
        print(f"Average NDCG@{args.top_k}: {avg_ndcg:.4f}")
    if len(all_avg_similarities) > 0:
        overall_avg_similarity = np.mean(all_avg_similarities)
        print(f"Average similarity: {overall_avg_similarity:.4f}")
    if len(all_match_ranks) > 0:
        avg_rank = np.mean(all_match_ranks)
        print(f"Average rank: {avg_rank:.2f}")
    else:
        print("Average rank: N/A (no matches found)")
    if len(all_matched_similarities) > 0:
        avg_matched_similarity = np.mean(all_matched_similarities)
        print(f"Average similarity of all matched parts: {avg_matched_similarity:.4f}")
    else:
        print("Average similarity of all matched parts: N/A (no matches found)")
    if len(all_matched_ranks) > 0:
        avg_matched_rank = np.mean(all_matched_ranks)
        print(f"Average rank of all matched parts: {avg_matched_rank:.2f}")
    else:
        print("Average rank of all matched parts: N/A (no matches found)")
    print("="*60)
    
    # If processing a folder, combine all results at the end
    if len(query_images) > 1:
        combine_retrieval_results(query_folder, model_name="vae")


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
        ax_query.set_title("Query Image", fontsize=12, fontweight='bold')
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

