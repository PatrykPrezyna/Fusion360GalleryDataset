"""
Enhanced script for retrieving similar CAD part images with rotation invariance.

This script supports multiple pretrained models optimized for geometric/structural tasks:
- DINOv2: Best for geometric features, better rotation handling
- CLIP: General purpose, good for semantic similarity
- ResNet50: Classic baseline with rotation augmentation

Usage:
    python retrieve_similar_images_rotation_invariant.py \
        --query "path/to/image.png" \
        --image-dir "path/to/images" \
        --top-k 5 \
        --model dinov2
"""

import os
# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


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


def load_dinov2_model(device="cpu"):
    """
    Load DINOv2 model - excellent for geometric/structural features.
    DINOv2 is better than CLIP for CAD parts because it focuses on geometric
    structure rather than semantic content.
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
        print("Loading DINOv2 model from Hugging Face...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        model = model.to(device)
        model.eval()
        print("DINOv2 model loaded successfully!")
        return model, processor
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")


def load_clip_model(device="cpu"):
    """
    Load CLIP model - good for general image similarity.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP model from Hugging Face...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)
        model.eval()
        print("CLIP model loaded successfully!")
        return model, processor
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")


def load_resnet50_model(device="cpu"):
    """
    Load ResNet50 model - classic baseline with good rotation handling when augmented.
    """
    print("Loading ResNet50 model...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final classification layer to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Standard ImageNet preprocessing
    processor = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("ResNet50 model loaded successfully!")
    return model, processor


def extract_embedding_dinov2(model, processor, image, device="cpu"):
    """Extract embedding using DINOv2."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token or mean pooling
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def extract_embedding_clip(model, processor, image, device="cpu"):
    """Extract embedding using CLIP."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def extract_embedding_resnet(model, processor, image, device="cpu"):
    """Extract embedding using ResNet50."""
    # Convert PIL to tensor
    if isinstance(image, Image.Image):
        image_tensor = processor(image).unsqueeze(0).to(device)
    else:
        image_tensor = processor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor)
        # ResNet outputs (batch, channels, H, W), need to pool
        embedding = F.adaptive_avg_pool2d(embedding, (1, 1))
        embedding = embedding.view(embedding.size(0), -1)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def extract_embedding_with_rotation_augmentation(
    model, processor, image, model_type="dinov2", device="cpu", num_rotations=4
):
    """
    Extract embedding with rotation augmentation for better rotation invariance.
    
    This function rotates the image multiple times, extracts embeddings for each,
    and averages them. This makes the final embedding more rotation-invariant.
    """
    embeddings = []
    
    for angle in range(0, 360, 360 // num_rotations):
        if angle == 0:
            rotated_img = image
        else:
            rotated_img = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        
        if model_type == "dinov2":
            emb = extract_embedding_dinov2(model, processor, rotated_img, device)
        elif model_type == "clip":
            emb = extract_embedding_clip(model, processor, rotated_img, device)
        elif model_type == "resnet50":
            emb = extract_embedding_resnet(model, processor, rotated_img, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        embeddings.append(emb)
    
    # Average the embeddings from different rotations
    avg_embedding = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
    avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
    return avg_embedding


def extract_image_embeddings(
    model, processor, image_paths, model_type="dinov2", device="cpu", 
    batch_size=32, use_rotation_aug=False, num_rotations=4
):
    """
    Extract embeddings for all images in the directory.
    
    Args:
        model: The model to use
        processor: The processor/preprocessor
        image_paths: List of image paths
        model_type: Type of model ("dinov2", "clip", "resnet50")
        device: Device to run inference on
        batch_size: Batch size for processing
        use_rotation_aug: Whether to use rotation augmentation
        num_rotations: Number of rotation angles to use for augmentation
    
    Returns:
        Tensor of embeddings (N, embedding_dim)
    """
    all_embeddings = []
    
    print(f"Extracting embeddings for {len(image_paths)} images...")
    if use_rotation_aug:
        print(f"  Using rotation augmentation with {num_rotations} rotations per image")
    
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
        
        if use_rotation_aug:
            # Process with rotation augmentation
            batch_embeddings = []
            for img in batch_images:
                emb = extract_embedding_with_rotation_augmentation(
                    model, processor, img, model_type, device, num_rotations
                )
                batch_embeddings.append(emb.cpu())
            batch_embedding = torch.cat(batch_embeddings, dim=0)
        else:
            # Standard batch processing
            if model_type == "dinov2":
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_embedding = outputs.last_hidden_state[:, 0, :]
                    batch_embedding = F.normalize(batch_embedding, p=2, dim=1).cpu()
            elif model_type == "clip":
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    batch_embedding = model.get_image_features(**inputs)
                    batch_embedding = F.normalize(batch_embedding, p=2, dim=1).cpu()
            elif model_type == "resnet50":
                batch_tensors = torch.stack([processor(img) for img in batch_images]).to(device)
                with torch.no_grad():
                    features = model(batch_tensors)
                    batch_embedding = F.adaptive_avg_pool2d(features, (1, 1))
                    batch_embedding = batch_embedding.view(batch_embedding.size(0), -1)
                    batch_embedding = F.normalize(batch_embedding, p=2, dim=1).cpu()
        
        all_embeddings.append(batch_embedding)
        
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")
    
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def retrieve_similar_images(
    query_image_path: str,
    model,
    processor,
    embeddings: torch.Tensor,
    image_paths: list,
    model_type: str = "dinov2",
    top_k: int = 5,
    device: str = "cpu",
    use_rotation_aug: bool = False,
    num_rotations: int = 4
):
    """
    Retrieve similar images using pretrained model.
    """
    # Load and process query image
    try:
        query_img = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load query image: {e}")
    
    # Extract query embedding
    if use_rotation_aug:
        query_embedding = extract_embedding_with_rotation_augmentation(
            model, processor, query_img, model_type, device, num_rotations
        )
    else:
        if model_type == "dinov2":
            query_embedding = extract_embedding_dinov2(model, processor, query_img, device)
        elif model_type == "clip":
            query_embedding = extract_embedding_clip(model, processor, query_img, device)
        elif model_type == "resnet50":
            query_embedding = extract_embedding_resnet(model, processor, query_img, device)
    
    # Move embeddings to same device
    embeddings = embeddings.to(device)
    
    # Compute cosine similarity
    similarities = (embeddings @ query_embedding.t()).squeeze(1)
    
    # Get top-k most similar
    top_k = min(top_k, len(image_paths))
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
        results.append((image_paths[idx], float(sim)))
    
    return results


def display_retrieval_results(query_image_path: str, results: list, save_path: str = None, 
                              display_size: int = 256, ndcg_score: float = None):
    """
    Display the query image and retrieved similar images in a grid.
    """
    num_results = len(results)
    
    fig = plt.figure(figsize=(display_size * (num_results + 1) / 100, display_size / 100))
    gs = gridspec.GridSpec(1, num_results + 1, figure=fig, wspace=0.1, hspace=0.1)
    
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
            
            ax.set_title(f"#{idx}\n{similarity:.3f}{match_text}",
                         fontsize=10, pad=5, color=title_color,
                         fontweight='bold' if is_match else 'normal')
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            ax = fig.add_subplot(gs[0, idx])
            ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar CAD part images with rotation invariance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Recommendations for CAD Parts:
  - dinov2: BEST for geometric/structural features, handles rotations well
  - clip: Good for general similarity, semantic understanding
  - resnet50: Classic baseline, works well with rotation augmentation

Rotation Augmentation:
  Use --rotation-aug to make embeddings more rotation-invariant by averaging
  embeddings from multiple rotated versions of each image.
        """
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image"
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
        "--model",
        type=str,
        default="dinov2",
        choices=["dinov2", "clip", "resnet50"],
        help="Model to use: dinov2 (best for CAD), clip, or resnet50"
    )
    parser.add_argument(
        "--rotation-aug",
        action="store_true",
        help="Use rotation augmentation for better rotation invariance"
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=4,
        help="Number of rotation angles for augmentation (default: 4 = 0, 90, 180, 270 degrees)"
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
    
    args = parser.parse_args()
    
    # Check if query image exists
    if not os.path.exists(args.query):
        print(f"Error: Query image not found: {args.query}")
        return
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")
    if args.rotation_aug:
        print(f"Rotation augmentation: ENABLED ({args.num_rotations} rotations)")
    
    # Load pretrained model
    if args.model == "dinov2":
        model, processor = load_dinov2_model(device=device)
    elif args.model == "clip":
        model, processor = load_clip_model(device=device)
    elif args.model == "resnet50":
        model, processor = load_resnet50_model(device=device)
    
    # Get all image paths
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.jpeg")))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    
    # Load or compute embeddings
    cache_key = f"{args.model}_{args.rotation_aug}_{args.num_rotations}"
    cache_emb_path = args.cache_embeddings or f"embeddings_{cache_key}.pt"
    cache_paths_path = args.cache_paths or f"image_paths_{cache_key}.txt"
    
    if os.path.exists(cache_emb_path):
        print(f"Loading cached embeddings from {cache_emb_path}...")
        embeddings = torch.load(cache_emb_path, map_location="cpu")
        if os.path.exists(cache_paths_path):
            with open(cache_paths_path, 'r') as f:
                cached_paths = [line.strip() for line in f.readlines()]
            if len(cached_paths) == len(image_paths) and cached_paths == image_paths:
                print("Cached embeddings match current image set.")
            else:
                print("Warning: Cached embeddings don't match current image set. Recomputing...")
                embeddings = extract_image_embeddings(
                    model, processor, image_paths, args.model, device=device,
                    use_rotation_aug=args.rotation_aug, num_rotations=args.num_rotations
                )
                torch.save(embeddings, cache_emb_path)
                with open(cache_paths_path, 'w') as f:
                    for path in image_paths:
                        f.write(f"{path}\n")
        else:
            embeddings = extract_image_embeddings(
                model, processor, image_paths, args.model, device=device,
                use_rotation_aug=args.rotation_aug, num_rotations=args.num_rotations
            )
            torch.save(embeddings, cache_emb_path)
            with open(cache_paths_path, 'w') as f:
                for path in image_paths:
                    f.write(f"{path}\n")
    else:
        embeddings = extract_image_embeddings(
            model, processor, image_paths, args.model, device=device,
            use_rotation_aug=args.rotation_aug, num_rotations=args.num_rotations
        )
        torch.save(embeddings, cache_emb_path)
        print(f"Saved embeddings to {cache_emb_path}")
        with open(cache_paths_path, 'w') as f:
            for path in image_paths:
                f.write(f"{path}\n")
        print(f"Saved image paths to {cache_paths_path}")
    
    # Retrieve similar images
    print(f"\nRetrieving {args.top_k} most similar images to: {args.query}")
    print("-" * 60)
    
    results = retrieve_similar_images(
        query_image_path=args.query,
        model=model,
        processor=processor,
        embeddings=embeddings,
        image_paths=image_paths,
        model_type=args.model,
        top_k=args.top_k,
        device=device,
        use_rotation_aug=args.rotation_aug,
        num_rotations=args.num_rotations
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
        print(f"{i}. {os.path.basename(path)} ({similarity:.4f}){match_flag}")
        print(f"   Full path: {path}")
    
    # Display images visually
    display_retrieval_results(
        query_image_path=args.query,
        results=flagged_results,
        save_path=args.save,
        display_size=args.display_size,
        ndcg_score=ndcg_score
    )


if __name__ == "__main__":
    main()

