"""
Example script for retrieving similar images using the trained VAE.

Usage:
    python retrieve_similar_images.py --query "path/to/image.png" --output-dir "vae_retrieval_output" --top-k 5
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar images using trained VAE"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image"
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
    
    # Check if query image exists
    if not os.path.exists(args.query):
        print(f"Error: Query image not found: {args.query}")
        return
    
    # Check if model exists
    model_path = os.path.join(args.output_dir, "vae_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_vae_image_retrieval.py")
        return
    
    # Load model and embeddings
    print(f"Loading model and embeddings from {args.output_dir}...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, embeddings, image_paths = load_model_and_embeddings(args.output_dir, device=device)
    print(f"Loaded {len(image_paths)} image embeddings")
    
    # Retrieve similar images
    print(f"\nRetrieving {args.top_k} most similar images to: {args.query}")
    print("-" * 60)
    
    results = retrieve_similar_images(
        query_image_path=args.query,
        model=model,
        embeddings=embeddings,
        image_paths=image_paths,
        top_k=args.top_k,
        device=device,
        image_size=args.image_size,
    )
    
    # Extract base ID from query image
    query_base_id = extract_base_id(args.query)
    
    # Count total relevant in entire pool (all images with same base ID)
    total_relevant = 0
    relevant_paths_set = set()
    for path in image_paths:
        if extract_base_id(path) == query_base_id:
            relevant_paths_set.add(path)
    total_relevant = len(relevant_paths_set)
    
    # Flag images that match the query base ID
    flagged_results = []
    for path, similarity in results:
        result_base_id = extract_base_id(path)
        is_match = (result_base_id == query_base_id)
        flagged_results.append((path, similarity, is_match))
    
    # Calculate NDCG@k
    ndcg_score = calculate_ndcg_at_k(flagged_results, total_relevant, args.top_k)
    
    # Display results
    print(f"\nTop {len(results)} most similar images:")
    print(f"Query base ID: {query_base_id}")
    print(f"Total relevant in pool: {total_relevant}")
    print(f"NDCG@{args.top_k}: {ndcg_score:.4f}")
    print("-" * 60)
    
    for i, (path, similarity, is_match) in enumerate(flagged_results, 1):
        match_flag = " [MATCH]" if is_match else ""
        print(f"{i}. {os.path.basename(path)} (similarity: {similarity:.4f}){match_flag}")
        print(f"   Full path: {path}")
    
    # Display images visually
    display_retrieval_results(
        query_image_path=args.query,
        results=flagged_results,
        save_path=args.save,
        display_size=args.display_size,
        ndcg_score=ndcg_score
    )


def display_retrieval_results(query_image_path: str, results: list, save_path: str = None, display_size: int = 256, ndcg_score: float = None):
    """
    Display the query image and retrieved similar images in a grid.
    
    Args:
        query_image_path: Path to the query image
        results: List of (image_path, similarity_score, is_match) tuples
        save_path: Optional path to save the visualization
        display_size: Size to display each image
        ndcg_score: Optional NDCG score to display in title
    """
    num_results = len(results)
    
    # Create figure with grid layout
    # Layout: Query image on left, similar images in a row on the right
    fig = plt.figure(figsize=(display_size * (num_results + 1) / 100, display_size / 100))
    gs = gridspec.GridSpec(1, num_results + 1, figure=fig, wspace=0.1, hspace=0.1)
    
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
        # Handle both old format (path, similarity) and new format (path, similarity, is_match)
        if len(result) == 3:
            img_path, similarity, is_match = result
        else:
            img_path, similarity = result
            is_match = False
        
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
            
            ax.set_title(f"#{idx}\nSimilarity: {similarity:.3f}{match_text}",
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
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()

