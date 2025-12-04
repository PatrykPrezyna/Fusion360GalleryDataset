"""
Explain image retrieval using SHAP (SHapley Additive exPlanations).

This script retrieves similar images and uses SHAP to visualize which pixels
or features are responsible for the similarity between the query image and
retrieved images.

Usage:
    python explain_image_retrieval_shap.py --query "path/to/image.png" --output-dir "vae_retrieval_output" --top-k 3
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    print("Error: SHAP is not installed. Please install it using:")
    print("  pip install shap")
    exit(1)

from train_vae_image_retrieval import load_model_and_embeddings, retrieve_similar_images, ConvVAE


def create_similarity_function(retrieved_embedding, model, device, image_size):
    """
    Create a function that computes similarity between a (possibly masked) query image
    and the retrieved image embedding.
    
    Args:
        retrieved_embedding: The embedding of the retrieved image (target to compare against)
        model: The VAE model
        device: Device to run on
        image_size: Size of images
    
    Returns:
        A function that takes an image array and returns similarity score
    """
    def similarity_fn(img_array):
        """
        Compute similarity between masked query image and retrieved image.
        
        Args:
            img_array: Image array in NHWC format (numpy array)
        
        Returns:
            Similarity score (cosine similarity between embeddings)
        """
        # Convert to tensor if needed
        if isinstance(img_array, np.ndarray):
            # Handle both single image and batch
            if img_array.ndim == 3:
                img_array = img_array[np.newaxis, ...]  # Add batch dimension
            
            # Convert from NHWC to NCHW
            img_tensor = torch.from_numpy(img_array).float()
            if img_tensor.shape[-1] == 3:  # NHWC format
                img_tensor = img_tensor.permute(0, 3, 1, 2)  # Convert to NCHW
        else:
            img_tensor = img_array
        
        # Normalize to [0, 1] if needed (assuming input is in [0, 255])
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Resize if needed
        if img_tensor.shape[-1] != image_size or img_tensor.shape[-2] != image_size:
            # For batch processing, we need to resize each image
            resized_images = []
            for i in range(img_tensor.shape[0]):
                img = img_tensor[i]
                # Convert to PIL for resizing
                if img.shape[0] == 3:  # CHW format
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = img.cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil = img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img_resized = np.array(img_pil).astype(np.float32) / 255.0
                img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)  # HWC to CHW
                resized_images.append(img_resized)
            img_tensor = torch.stack(resized_images)
        
        img_tensor = img_tensor.to(device)
        
        # Encode the (masked) query image
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(img_tensor)
            query_emb = mu  # Use mean as embedding
        
        # Normalize embeddings for cosine similarity
        query_emb_norm = query_emb / (query_emb.norm(dim=1, keepdim=True) + 1e-8)
        retrieved_emb_norm = retrieved_embedding / (retrieved_embedding.norm() + 1e-8)
        
        # Compute cosine similarity
        similarity = (query_emb_norm @ retrieved_emb_norm.t()).squeeze()
        
        # Convert to numpy if needed
        if isinstance(similarity, torch.Tensor):
            similarity = similarity.cpu().numpy()
        
        # Handle batch output
        if isinstance(similarity, np.ndarray) and similarity.ndim > 0:
            return similarity
        return float(similarity)
    
    return similarity_fn


def explain_retrieval_with_shap(
    query_image_path: str,
    retrieved_image_path: str,
    model: ConvVAE,
    device: str,
    image_size: int,
    max_evals: int = 1000,
    batch_size: int = 50,
):
    """
    Use SHAP to explain why the retrieved image is similar to the query image.
    
    Args:
        query_image_path: Path to query image
        retrieved_image_path: Path to retrieved similar image
        model: VAE model
        device: Device to run on
        image_size: Image size used in model
        max_evals: Number of evaluations for SHAP
        batch_size: Batch size for SHAP evaluations
    
    Returns:
        SHAP values and original image data
    """
    # Load and preprocess images
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    query_img = Image.open(query_image_path).convert("RGB")
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    
    retrieved_img = Image.open(retrieved_image_path).convert("RGB")
    retrieved_tensor = transform(retrieved_img).unsqueeze(0).to(device)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        query_mu, _ = model.encode(query_tensor)
        retrieved_mu, _ = model.encode(retrieved_tensor)
        query_embedding = query_mu
        retrieved_embedding = retrieved_mu
    
    # Convert query image to numpy for SHAP (NHWC format)
    query_np = query_tensor[0].cpu().permute(1, 2, 0).numpy()  # CHW to HWC
    # Normalize to [0, 1] if needed
    if query_np.max() > 1.0:
        query_np = query_np / 255.0
    
    # Create similarity function
    similarity_fn = create_similarity_function(
        retrieved_embedding, model, device, image_size
    )
    
    # Create masker for SHAP
    # Use blur masker to mask out parts of the image
    masker = shap.maskers.Image("blur(128,128)", query_np.shape)
    
    # Create explainer
    explainer = shap.Explainer(
        similarity_fn,
        masker,
        output_names=[f"Similarity to {os.path.basename(retrieved_image_path)}"]
    )
    
    # Explain the query image
    print(f"Computing SHAP values (this may take a while with {max_evals} evaluations)...")
    shap_values = explainer(
        query_np[np.newaxis, ...],  # Add batch dimension
        max_evals=max_evals,
        batch_size=batch_size,
    )
    
    return shap_values, query_np


def visualize_shap_explanations(
    query_image_path: str,
    retrieved_image_path: str,
    shap_values,
    query_image_np: np.ndarray,
    similarity_score: float,
    save_path: str = None,
):
    """
    Visualize SHAP explanations for image retrieval.
    
    Args:
        query_image_path: Path to query image
        retrieved_image_path: Path to retrieved image
        shap_values: SHAP values from explainer
        query_image_np: Original query image as numpy array
        similarity_score: Similarity score between images
        save_path: Optional path to save visualization
    """
    # Prepare data for visualization
    # SHAP Explanation object has .values and .data attributes
    if hasattr(shap_values, 'values'):
        shap_vals = shap_values.values
        if isinstance(shap_vals, torch.Tensor):
            shap_vals = shap_vals.cpu().numpy()
        
        # Handle different shapes - SHAP Partition explainer returns (batch, H, W, C, outputs)
        if shap_vals.ndim == 5:  # (1, H, W, C, outputs)
            shap_vals = shap_vals[0]  # Remove batch dimension -> (H, W, C, outputs)
            if shap_vals.ndim == 4:  # (H, W, C, outputs)
                shap_vals = shap_vals[..., 0]  # Take first output -> (H, W, C)
        elif shap_vals.ndim == 4:  # (1, H, W, C) or (H, W, C, outputs)
            if shap_vals.shape[0] == 1:  # Batch dimension
                shap_vals = shap_vals[0]  # Remove batch -> (H, W, C)
        
        # Convert to list of arrays for each output (SHAP image_plot expects list)
        if shap_vals.ndim == 3:  # (H, W, C)
            shap_vals = [shap_vals]
        elif shap_vals.ndim == 4:  # (H, W, C, outputs)
            shap_vals = [shap_vals[..., i] for i in range(shap_vals.shape[-1])]
    else:
        # Fallback if not a proper SHAP Explanation object
        shap_vals = [shap_values] if isinstance(shap_values, np.ndarray) else [np.array(shap_values)]
    
    # Get pixel values from SHAP data if available, otherwise use query image
    if hasattr(shap_values, 'data'):
        pixel_data = shap_values.data
        if isinstance(pixel_data, torch.Tensor):
            pixel_data = pixel_data.cpu().numpy()
        # Remove batch dimension if present
        if pixel_data.ndim == 4 and pixel_data.shape[0] == 1:
            pixel_data = pixel_data[0]
        # Convert from CHW to HWC if needed
        if pixel_data.shape[0] == 3 and pixel_data.ndim == 3:
            pixel_data = pixel_data.transpose(1, 2, 0)
        pixel_values = pixel_data
    else:
        # Use query image
        if query_image_np.max() <= 1.0:
            pixel_values = (query_image_np * 255).astype(np.uint8)
        else:
            pixel_values = query_image_np.astype(np.uint8)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Query image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(pixel_values)
    ax1.set_title("Query Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Retrieved image
    retrieved_img = Image.open(retrieved_image_path).convert("RGB")
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(retrieved_img)
    ax2.set_title(f"Retrieved Image\n(Similarity: {similarity_score:.4f})", 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: SHAP explanation
    ax3 = plt.subplot(1, 3, 3)
    
    # Use SHAP's image plot if available, otherwise manual plotting
    try:
        shap.image_plot(
            shap_values=shap_vals,
            pixel_values=pixel_values,
            labels=[f"Similarity: {similarity_score:.4f}"],
            show=False
        )
    except Exception as e:
        print(f"Warning: Could not use shap.image_plot, using manual plot: {e}")
        # Manual plotting
        if len(shap_vals) > 0:
            # Sum SHAP values across channels for visualization
            shap_sum = np.sum(shap_vals[0], axis=2) if shap_vals[0].ndim == 3 else shap_vals[0]
            
            im = ax3.imshow(shap_sum, cmap='RdBu', vmin=-np.abs(shap_sum).max(), 
                           vmax=np.abs(shap_sum).max())
            ax3.set_title("SHAP Values\n(Red = increases similarity)", 
                         fontsize=12, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046)
    
    plt.suptitle("SHAP Explanation for Image Retrieval", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Explain image retrieval using SHAP"
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
        default=3,
        help="Number of similar images to retrieve and explain"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size used during training"
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=1000,
        help="Number of evaluations for SHAP (more = more accurate but slower)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for SHAP evaluations"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the visualization (e.g., 'shap_explanation.png'). If not provided, displays interactively."
    )
    parser.add_argument(
        "--explain-all",
        action="store_true",
        help="Explain all top-k retrieved images (default: only explain the first one)"
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
    
    # Explain retrieved images
    num_to_explain = len(results) if args.explain_all else 1
    
    for i, (retrieved_path, similarity_score) in enumerate(results[:num_to_explain]):
        print(f"\n{'='*60}")
        print(f"Explaining retrieval #{i+1}: {os.path.basename(retrieved_path)}")
        print(f"Similarity score: {similarity_score:.4f}")
        print(f"{'='*60}")
        
        # Compute SHAP explanations
        shap_values, query_image_np = explain_retrieval_with_shap(
            query_image_path=args.query,
            retrieved_image_path=retrieved_path,
            model=model,
            device=device,
            image_size=args.image_size,
            max_evals=args.max_evals,
            batch_size=args.batch_size,
        )
        
        # Visualize
        save_path = args.save
        if save_path and num_to_explain > 1:
            # Add index to filename if explaining multiple
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_{i+1}{ext}"
        
        visualize_shap_explanations(
            query_image_path=args.query,
            retrieved_image_path=retrieved_path,
            shap_values=shap_values,
            query_image_np=query_image_np,
            similarity_score=similarity_score,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()

