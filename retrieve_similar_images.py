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
from train_vae_image_retrieval import load_model_and_embeddings, retrieve_similar_images


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
    
    # Display results
    print(f"\nTop {len(results)} most similar images:")
    for i, (path, similarity) in enumerate(results, 1):
        print(f"{i}. {os.path.basename(path)} (similarity: {similarity:.4f})")
        print(f"   Full path: {path}")
    
    # Display images visually
    display_retrieval_results(
        query_image_path=args.query,
        results=results,
        save_path=args.save,
        display_size=args.display_size
    )


def display_retrieval_results(query_image_path: str, results: list, save_path: str = None, display_size: int = 256):
    """
    Display the query image and retrieved similar images in a grid.
    
    Args:
        query_image_path: Path to the query image
        results: List of (image_path, similarity_score) tuples
        save_path: Optional path to save the visualization
        display_size: Size to display each image
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
    for idx, (img_path, similarity) in enumerate(results, 1):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(img)
            ax.set_title(f"#{idx}\n{os.path.basename(img_path)}\nSimilarity: {similarity:.3f}", 
                        fontsize=10, pad=5)
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Create empty subplot if image fails to load
            ax = fig.add_subplot(gs[0, idx])
            ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    plt.suptitle("Image Retrieval Results", fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()

