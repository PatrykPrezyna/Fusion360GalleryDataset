"""
Script to visualize which pixels/regions in images are responsible for similarity matching.
Uses attention maps and gradient-based methods to show explainability of the DINOv2 model.

Usage:
    python visualize_similarity_explainability.py --query "path/to/query.png" --target "path/to/target.png" --output "explainability.png"
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoImageProcessor, AutoModel
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available. Using PIL for resizing instead.")
    cv2 = None


def load_pretrained_model(device="cpu"):
    """Load the pretrained DINOv2 model and processor."""
    print("Loading pretrained DINOv2 model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def get_attention_maps(model, inputs, device="cpu"):
    """
    Extract attention maps from DINOv2 model.
    
    DINOv2 uses a Vision Transformer architecture. We extract attention from the CLS token
    to all image patches, which shows which parts of the image the model focuses on.
    
    Args:
        model: DINOv2 model
        inputs: Processed image inputs
        device: Device to run inference on
    
    Returns:
        Attention maps averaged across layers and heads, or None if not available
    """
    model.eval()
    
    # Method 1: Try to get attention directly from model output
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions
            
            # attentions is a tuple of tensors, one per layer
            # Each tensor shape: (batch, num_heads, seq_len, seq_len)
            if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
                # Stack all layer attentions: (num_layers, batch, num_heads, seq_len, seq_len)
                attentions = torch.stack(attentions)
                
                # Average across heads: (num_layers, batch, num_heads, seq_len, seq_len) -> (num_layers, batch, seq_len, seq_len)
                attentions = attentions.mean(dim=2)
                
                # Average across layers: (num_layers, batch, seq_len, seq_len) -> (batch, seq_len, seq_len)
                attentions = attentions.mean(dim=0)
                
                # Get CLS token attention to patches
                # CLS token is at index 0, so we take attention from CLS to all tokens
                # attentions shape: (batch, seq_len, seq_len)
                cls_attention = attentions[0, 0, :]  # (seq_len,) - attention from CLS to all tokens
                
                # Remove CLS token itself (index 0) to get patch attention
                patch_attention = cls_attention[1:].cpu().numpy()  # (num_patches,)
                
                return patch_attention
    except Exception as e:
        # If output_attentions doesn't work, try hook-based approach
        pass
    
    # Method 2: Use hooks to capture attention from self-attention layers
    attention_weights = []
    
    def attention_hook(module, input, output):
        """Hook to capture attention weights"""
        # The output from self-attention is typically (batch, seq_len, hidden_dim)
        # But we need the attention weights, which are usually stored in the module
        if hasattr(module, 'attention_probs'):
            # Some implementations store attention probs
            attn = module.attention_probs
            if attn is not None and isinstance(attn, torch.Tensor):
                attention_weights.append(attn.detach())
    
    # Register hooks on attention modules
    hooks = []
    for name, module in model.named_modules():
        # Look for self-attention modules in the encoder
        if 'self' in name.lower() and 'attn' in name.lower():
            if hasattr(module, 'forward'):
                hooks.append(module.register_forward_hook(attention_hook))
    
    # Forward pass to trigger hooks
    try:
        with torch.no_grad():
            _ = model(**inputs)
    except:
        pass
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # If we captured attention via hooks, process it
    if len(attention_weights) > 0:
        attentions = torch.stack(attention_weights)
        if len(attentions.shape) == 5:  # (num_layers, batch, num_heads, seq_len, seq_len)
            attentions = attentions.mean(dim=2).mean(dim=0)  # Average heads and layers
        elif len(attentions.shape) == 4:  # (num_layers, batch, seq_len, seq_len)
            attentions = attentions.mean(dim=0)  # Average layers
        
        cls_attention = attentions[0, 0, :]
        patch_attention = cls_attention[1:].cpu().numpy()
        return patch_attention
    
    # If neither method worked, return None
    # This is common with some DINOv2 implementations that don't expose attention
    return None


def compute_gradient_saliency(query_img_tensor, target_embedding, model, processor, device="cpu"):
    """
    Compute gradient-based saliency map showing which pixels contribute to similarity.
    
    Args:
        query_img_tensor: Input image tensor (requires_grad=True)
        target_embedding: Target embedding to compare against
        model: DINOv2 model
        processor: Image processor
        device: Device to run inference on
    
    Returns:
        Saliency map as numpy array
    """
    model.eval()
    
    # Ensure gradients are enabled
    query_img_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(pixel_values=query_img_tensor)
    query_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = (query_embedding @ target_embedding.t()).squeeze()
    
    # Backward pass
    similarity.backward()
    
    # Get gradients
    gradients = query_img_tensor.grad.data
    
    # Compute saliency as absolute gradient values
    saliency = torch.abs(gradients)
    saliency = saliency.squeeze(0)  # Remove batch dimension
    saliency = saliency.max(dim=0)[0]  # Max across channels
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency.cpu().numpy()


def visualize_attention_on_image(image, attention_map, patch_size=14):
    """
    Visualize attention map overlaid on the original image.
    
    Args:
        image: PIL Image or numpy array
        attention_map: 1D attention map (num_patches,)
        patch_size: Size of each patch (DINOv2 uses 14x14 patches)
    
    Returns:
        Visualization as numpy array
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    h, w = img_array.shape[:2]
    
    # Reshape attention map to 2D
    # DINOv2 base uses 224x224 images with 14x14 patches = 16x16 patches
    num_patches_per_side = int(np.sqrt(len(attention_map)))
    attention_2d = attention_map.reshape(num_patches_per_side, num_patches_per_side)
    
    # Resize attention map to image size
    if cv2 is not None:
        attention_resized = cv2.resize(attention_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        # Fallback to PIL
        from PIL import Image as PILImage
        attention_img = PILImage.fromarray((attention_2d * 255).astype(np.uint8))
        attention_resized = np.array(attention_img.resize((w, h), PILImage.Resampling.BICUBIC)) / 255.0
    
    # Normalize attention map
    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.jet(attention_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    alpha = 0.5
    overlay = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
    
    return overlay


def visualize_saliency_on_image(image, saliency_map):
    """
    Visualize saliency map overlaid on the original image.
    
    Args:
        image: PIL Image or numpy array
        saliency_map: 2D saliency map (H, W)
    
    Returns:
        Visualization as numpy array
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    h, w = img_array.shape[:2]
    
    # Resize saliency to match image size
    if cv2 is not None:
        saliency_resized = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        # Fallback to PIL
        from PIL import Image as PILImage
        saliency_img = PILImage.fromarray((saliency_map * 255).astype(np.uint8))
        saliency_resized = np.array(saliency_img.resize((w, h), PILImage.Resampling.BICUBIC)) / 255.0
    
    # Normalize
    saliency_resized = (saliency_resized - saliency_resized.min()) / (saliency_resized.max() - saliency_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.hot(saliency_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    alpha = 0.6
    overlay = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
    
    return overlay


def create_explainability_visualization(query_path, target_path, model, processor, device="cpu", output_path=None):
    """
    Create a comprehensive explainability visualization showing which regions contribute to similarity.
    
    Args:
        query_path: Path to query image
        target_path: Path to target (similar) image
        model: DINOv2 model
        processor: Image processor
        device: Device to run inference on
        output_path: Path to save visualization
    """
    # Load images
    query_img = Image.open(query_path).convert("RGB")
    target_img = Image.open(target_path).convert("RGB")
    
    # Process images
    query_inputs = processor(images=query_img, return_tensors="pt")
    target_inputs = processor(images=target_img, return_tensors="pt")
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    target_inputs = {k: v.to(device) for k, v in target_inputs.items()}
    
    # Get target embedding
    with torch.no_grad():
        target_outputs = model(**target_inputs)
        target_embedding = target_outputs.last_hidden_state[:, 0, :]
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
    
    # Compute similarity
    with torch.no_grad():
        query_outputs = model(**query_inputs)
        query_embedding = query_outputs.last_hidden_state[:, 0, :]
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        similarity = (query_embedding @ target_embedding.t()).item()
    
    print(f"Similarity score: {similarity:.4f}")
    
    # Get attention maps
    print("Computing attention maps...")
    query_attention = get_attention_maps(model, query_inputs, device)
    target_attention = get_attention_maps(model, target_inputs, device)
    
    # Compute gradient saliency
    print("Computing gradient saliency...")
    query_img_tensor = query_inputs['pixel_values'].clone().detach().requires_grad_(True)
    query_saliency = compute_gradient_saliency(query_img_tensor, target_embedding, model, processor, device)
    
    target_img_tensor = target_inputs['pixel_values'].clone().detach().requires_grad_(True)
    target_saliency = compute_gradient_saliency(target_img_tensor, query_embedding, model, processor, device)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Query image
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    if query_attention is not None:
        query_att_overlay = visualize_attention_on_image(query_img, query_attention)
        axes[0, 1].imshow(query_att_overlay)
        axes[0, 1].set_title("Query: Attention Map", fontsize=12)
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, "Attention not available", ha='center', va='center')
        axes[0, 1].axis('off')
    
    query_sal_overlay = visualize_saliency_on_image(query_img, query_saliency)
    axes[0, 2].imshow(query_sal_overlay)
    axes[0, 2].set_title("Query: Gradient Saliency", fontsize=12)
    axes[0, 2].axis('off')
    
    # Combined visualization for query
    if query_attention is not None:
        query_combined = visualize_attention_on_image(query_img, query_attention)
    else:
        query_combined = np.array(query_img)
    axes[0, 3].imshow(query_combined)
    axes[0, 3].set_title("Query: Combined Visualization", fontsize=12)
    axes[0, 3].axis('off')
    
    # Row 2: Target image
    axes[1, 0].imshow(target_img)
    axes[1, 0].set_title("Target Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    if target_attention is not None:
        target_att_overlay = visualize_attention_on_image(target_img, target_attention)
        axes[1, 1].imshow(target_att_overlay)
        axes[1, 1].set_title("Target: Attention Map", fontsize=12)
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, "Attention not available", ha='center', va='center')
        axes[1, 1].axis('off')
    
    target_sal_overlay = visualize_saliency_on_image(target_img, target_saliency)
    axes[1, 2].imshow(target_sal_overlay)
    axes[1, 2].set_title("Target: Gradient Saliency", fontsize=12)
    axes[1, 2].axis('off')
    
    # Combined visualization for target
    if target_attention is not None:
        target_combined = visualize_attention_on_image(target_img, target_attention)
    else:
        target_combined = np.array(target_img)
    axes[1, 3].imshow(target_combined)
    axes[1, 3].set_title("Target: Combined Visualization", fontsize=12)
    axes[1, 3].axis('off')
    
    # Add overall title
    fig.suptitle(f"Similarity Explainability Visualization (Similarity: {similarity:.4f})", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize which pixels/regions contribute to image similarity"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target (similar) image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization (default: query_name_explainability.png)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified."
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.query):
        print(f"Error: Query image not found: {args.query}")
        return
    
    if not os.path.exists(args.target):
        print(f"Error: Target image not found: {args.target}")
        return
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    model, processor = load_pretrained_model(device=device)
    
    # Determine output path
    if args.output is None:
        query_name = os.path.splitext(os.path.basename(args.query))[0]
        target_name = os.path.splitext(os.path.basename(args.target))[0]
        output_dir = os.path.dirname(args.query) or "."
        args.output = os.path.join(output_dir, f"{query_name}_vs_{target_name}_explainability.png")
    
    # Create visualization
    create_explainability_visualization(
        query_path=args.query,
        target_path=args.target,
        model=model,
        processor=processor,
        device=device,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

