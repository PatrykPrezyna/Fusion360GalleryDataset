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


if __name__ == "__main__":
    main()

