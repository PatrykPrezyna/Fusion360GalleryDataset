# DINOv2 Image Retrieval Tuning Guide

This guide explains how to tune the DINOv2-based image retrieval system to achieve better results for your specific use case.

## Overview

The retrieval script (`retrieve_similar_images_use_pretrained_model_dinov2_centroid.py`) uses pretrained DINOv2 models for finding similar CAD part images. DINOv2 is excellent for geometric and structural features, making it ideal for recognizing rotated views of the same 3D part.

## Quick Start

### Basic Usage
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/query/images" \
    --image-dir "path/to/search/pool" \
    --top-k 5
```

### With Caching (Recommended for Repeated Experiments)
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/query/images" \
    --image-dir "path/to/search/pool" \
    --cache-embeddings "embeddings_dinov2.pt" \
    --cache-paths "image_paths.txt" \
    --top-k 5
```

---

## Tunable Parameters

### 1. Model Variant (`--model-variant`)

**What it does:** Controls the size and capacity of the DINOv2 model.

**Options:**
- `small`: 21M parameters, 384-dim embeddings, fastest inference
- `base`: 86M parameters, 768-dim embeddings, default (good balance)
- `large`: 300M parameters, 1024-dim embeddings, better accuracy
- `giant`: 1.1B parameters, 1536-dim embeddings, best accuracy, slowest

**Default:** `base`

**How to Change:**
```bash
# Use smaller model for faster processing
--model-variant small

# Use larger model for better accuracy
--model-variant large
```

**Expected Results:**
- **Small → Base:** Expect 2-5% improvement in retrieval accuracy, ~30% slower
- **Base → Large:** Expect 3-8% improvement in retrieval accuracy, ~50-70% slower
- **Large → Giant:** Expect 2-4% improvement, ~100% slower (often not worth it)

**When to Use:**
- **Small:** When speed is critical and you have limited GPU memory
- **Base:** Default choice for most use cases (recommended starting point)
- **Large:** When accuracy is more important than speed, and you have GPU resources
- **Giant:** Only for final evaluations when you need the absolute best results

**Strategy:**
1. Start with `base` and establish a baseline
2. If results are insufficient, try `large`
3. Compare accuracy vs. speed tradeoff
4. Only try `giant` if you have significant computational resources

---

### 2. Pooling Strategy (`--pooling-strategy`)

**What it does:** Controls how patch-level features are aggregated into a single image embedding.

**Options:**
- `cls`: Uses only the CLS token (first token), default, fastest
- `mean`: Averages all patch tokens (excludes CLS token)
- `cls+mean`: Concatenates CLS token + mean of patches (larger embedding)
- `weighted_mean`: Center-weighted average (higher weight for central patches)
- `max`: Max pooling across all patch tokens

**Default:** `cls`

**How to Change:**
```bash
# Use mean pooling for fine-grained features
--pooling-strategy mean

# Combine CLS and mean for richer representation
--pooling-strategy cls+mean

# Use center-weighted pooling for CAD parts
--pooling-strategy weighted_mean
```

**Expected Results:**

| Strategy | Embedding Size | Best For | Expected Improvement Over CLS |
|----------|---------------|----------|------------------------------|
| `cls` | 768 (base) | General semantic features | Baseline |
| `mean` | 768 (base) | Fine-grained geometric details | +2-5% for geometric shapes |
| `cls+mean` | 1536 (base) | Rich feature representation | +3-7% overall, slower |
| `weighted_mean` | 768 (base) | CAD parts with central focus | +2-4% for centered objects |
| `max` | 768 (base) | Capturing most prominent features | Varies, often worse |

**When to Use:**
- **CLS:** Default, good for semantic similarity, fastest
- **Mean:** Best for geometric shapes, rotated views, CAD parts
- **CLS+Mean:** When you need both semantic and geometric features
- **Weighted Mean:** For CAD parts where the center is most important
- **Max:** Rarely recommended, may lose important details

**Strategy:**
1. Start with `cls` to establish baseline
2. For CAD/geometric retrieval, try `mean` (often better)
3. If you have computational resources, try `cls+mean` (concatenated)
4. For centered CAD objects, test `weighted_mean`
5. Compare NDCG scores across strategies

**Example Tuning Process:**
```bash
# Step 1: Baseline with CLS
python ... --pooling-strategy cls --benchmark

# Step 2: Try mean pooling
python ... --pooling-strategy mean --benchmark

# Step 3: Compare results (check NDCG scores)
# If mean is better, continue with it
# If similar, stick with cls (faster)
```

---

### 3. Temperature Scaling (`--temperature`)

**What it does:** Controls the sharpness of similarity score distributions.

**Options:** Any positive float value
- `1.0`: No scaling (default)
- `< 1.0`: Sharper distribution (emphasizes top matches more)
- `> 1.0`: Softer distribution (makes scores more uniform)

**Default:** `1.0`

**How to Change:**
```bash
# Make top matches more distinct (sharper)
--temperature 0.7

# Make scores more uniform (softer)
--temperature 1.5
```

**Expected Results:**

| Temperature | Effect | Use Case | Expected Change |
|-------------|--------|----------|-----------------|
| 0.5-0.7 | Very sharp, top-1 dominates | Clear, distinct objects | Top-1 more confident, less diversity |
| 0.8-0.9 | Moderately sharp | Standard retrieval | Slight improvement in top-k ranking |
| 1.0 | No scaling (default) | Baseline | No change |
| 1.2-1.5 | Softer scores | Diverse results needed | More balanced rankings |
| 2.0+ | Very soft | Exploratory search | Less distinction between matches |

**When to Use:**
- **Low (0.7-0.9):** When you want to emphasize the best matches and have high confidence in queries
- **Default (1.0):** General purpose, recommended starting point
- **High (1.2-1.5):** When you want more diverse results or have ambiguous queries

**Strategy:**
1. Start with default `1.0`
2. If top matches are too similar in score, try `0.7-0.9` (sharper)
3. If you need more diverse results, try `1.2-1.5` (softer)
4. Monitor NDCG@k scores to find optimal value

**Example Tuning Process:**
```bash
# Baseline
python ... --temperature 1.0 --benchmark

# Try sharper (if top matches are unclear)
python ... --temperature 0.8 --benchmark

# Try softer (if results are too similar)
python ... --temperature 1.2 --benchmark

# Compare NDCG@k scores across temperatures
```

---

### 4. Top-K (`--top-k`)

**What it does:** Number of similar images to retrieve.

**Default:** `5`

**How to Change:**
```bash
# Retrieve more results
--top-k 10

# Retrieve fewer results
--top-k 3
```

**Expected Results:**
- Larger `top-k` may improve recall but lower precision
- Smaller `top-k` focuses on highest-confidence matches
- NDCG@k scales with k (larger k = potentially higher NDCG)

**When to Use:**
- **Small (3-5):** When displaying to users or focusing on precision
- **Medium (10-20):** For evaluation metrics or broader exploration
- **Large (50+):** For comprehensive search or evaluation

---

### 5. Batch Size (Internal, not currently exposed)

**Current default:** `32`

**Note:** This is hardcoded in the script. For very large datasets or limited GPU memory, you may need to modify the `batch_size` parameter in the code.

**Expected Impact:**
- Larger batches: Faster processing, more GPU memory
- Smaller batches: Slower but lower memory usage

---

## Systematic Tuning Strategy

### Step-by-Step Optimization Process

#### Phase 1: Establish Baseline
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --cache-embeddings "embeddings_baseline.pt" \
    --cache-paths "paths.txt" \
    --model-variant base \
    --pooling-strategy cls \
    --temperature 1.0 \
    --top-k 5 \
    --benchmark \
    --dataset-name baseline
```

**Record:** NDCG@5, retrieval time, model loading time

---

#### Phase 2: Optimize Model Size (if needed)

**Test 2a: Try Larger Model**
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --cache-embeddings "embeddings_large.pt" \
    --model-variant large \
    --pooling-strategy cls \
    --temperature 1.0 \
    --top-k 5 \
    --benchmark \
    --dataset-name large_model
```

**Compare:**
- NDCG improvement vs. speed degradation
- If improvement < 3%, stick with `base`
- If improvement > 5%, consider using `large`

---

#### Phase 3: Optimize Pooling Strategy

**Test 3a: Mean Pooling**
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --cache-embeddings "embeddings_mean.pt" \
    --pooling-strategy mean \
    --temperature 1.0 \
    --top-k 5 \
    --benchmark \
    --dataset-name mean_pooling
```

**Test 3b: CLS+Mean (if computational resources allow)**
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --cache-embeddings "embeddings_clsmean.pt" \
    --pooling-strategy cls+mean \
    --temperature 1.0 \
    --top-k 5 \
    --benchmark \
    --dataset-name clsmean_pooling
```

**Compare:**
- Which pooling gives best NDCG@5?
- Consider speed vs. accuracy tradeoff
- For CAD parts, `mean` often wins

---

#### Phase 4: Optimize Temperature

**Once you've selected model and pooling, tune temperature:**

**Test 4a: Sharper (0.8)**
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --pooling-strategy mean \
    --temperature 0.8 \
    --top-k 5 \
    --benchmark \
    --dataset-name temp_0.8
```

**Test 4b: Softer (1.2)**
```bash
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query "path/to/queries" \
    --image-dir "path/to/pool" \
    --pooling-strategy mean \
    --temperature 1.2 \
    --top-k 5 \
    --benchmark \
    --dataset-name temp_1.2
```

**Compare NDCG@k scores across temperatures**

---

## Recommended Configurations by Use Case

### High Accuracy (Best Results)
```bash
--model-variant large \
--pooling-strategy cls+mean \
--temperature 0.8 \
--top-k 10
```
**Trade-off:** Slower processing, higher memory usage

---

### Balanced (Recommended Default)
```bash
--model-variant base \
--pooling-strategy mean \
--temperature 1.0 \
--top-k 5
```
**Trade-off:** Good balance of speed and accuracy

---

### Fast Processing (Speed Priority)
```bash
--model-variant base \
--pooling-strategy cls \
--temperature 1.0 \
--top-k 5
```
**Trade-off:** Faster but slightly lower accuracy

---

### CAD Parts / Geometric Shapes
```bash
--model-variant base \
--pooling-strategy mean \
--temperature 0.9 \
--top-k 5
```
**Why:** Mean pooling captures geometric features better than CLS token alone

---

### Limited GPU Memory
```bash
--model-variant small \
--pooling-strategy cls \
--temperature 1.0 \
--top-k 5
```

---

## Monitoring Results

### Key Metrics to Track

1. **NDCG@k:** Normalized Discounted Cumulative Gain - primary accuracy metric
   - Higher is better (range: 0.0 to 1.0)
   - Check NDCG@5, NDCG@10 for different top-k values

2. **Retrieval Time:** Total time to process all queries
   - Important for production deployment
   - Compare across different configurations

3. **Model Loading Time:** Time to load the model
   - One-time cost, but important for cold starts

4. **Embedding Extraction Time:** Time to process all images in pool
   - Cached, so only relevant for first run

### Using Benchmark Mode

Enable benchmarking to track all metrics:
```bash
--benchmark \
--benchmark-output "results/comparison" \
--dataset-name "config_name"
```

This generates:
- JSON file with detailed metrics
- CSV summary with key statistics
- CSV detailed with per-query results

Compare CSV files across different configurations to find the best settings.

---

## Common Issues and Solutions

### Issue: Low NDCG Scores
**Solutions:**
1. Try larger model (`--model-variant large`)
2. Switch to `--pooling-strategy mean` or `cls+mean`
3. Check if queries and database images are compatible
4. Verify image quality and preprocessing

### Issue: Results Too Similar / Low Diversity
**Solutions:**
1. Increase temperature (`--temperature 1.2-1.5`)
2. Increase `--top-k` to see more diverse results

### Issue: Slow Processing
**Solutions:**
1. Use `--model-variant small` or `base`
2. Use `--pooling-strategy cls` (fastest)
3. Enable caching with `--cache-embeddings`
4. Reduce batch size in code if GPU memory limited

### Issue: Top Matches Not Relevant
**Solutions:**
1. Try `--pooling-strategy mean` for better geometric matching
2. Verify query images are representative
3. Check if embeddings cache matches current image set
4. Try different temperature (0.8-0.9 for sharper)

---

## Best Practices

1. **Always use caching:** Save computation time on repeated experiments
   ```bash
   --cache-embeddings "embeddings.pt" --cache-paths "paths.txt"
   ```

2. **Test one parameter at a time:** Isolate the effect of each change

3. **Use benchmark mode:** Get quantitative metrics to compare configurations
   ```bash
   --benchmark --benchmark-output "results"
   ```

4. **Start simple:** Begin with defaults, then optimize based on results

5. **Consider your use case:** 
   - Production deployment → prioritize speed
   - Research/Evaluation → prioritize accuracy
   - CAD/Geometric shapes → use `mean` pooling

6. **Document your experiments:** Keep track of which configurations you've tested and their results

---

## Example Complete Tuning Workflow

```bash
# 1. Baseline
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query queries/ --image-dir pool/ \
    --cache-embeddings emb_base.pt --cache-paths paths.txt \
    --model-variant base --pooling-strategy cls --temperature 1.0 \
    --benchmark --benchmark-output results/ --dataset-name baseline

# 2. Try mean pooling
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query queries/ --image-dir pool/ \
    --cache-embeddings emb_mean.pt \
    --model-variant base --pooling-strategy mean --temperature 1.0 \
    --benchmark --benchmark-output results/ --dataset-name mean

# 3. Compare results, if mean is better, try temperature tuning
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query queries/ --image-dir pool/ \
    --cache-embeddings emb_mean.pt \
    --model-variant base --pooling-strategy mean --temperature 0.8 \
    --benchmark --benchmark-output results/ --dataset-name mean_temp08

# 4. If still need better accuracy, try larger model
python retrieve_similar_images_use_pretrained_model_dinov2_centroid.py \
    --query queries/ --image-dir pool/ \
    --cache-embeddings emb_large_mean.pt \
    --model-variant large --pooling-strategy mean --temperature 0.8 \
    --benchmark --benchmark-output results/ --dataset-name large_mean_temp08
```

Compare the CSV files in `results/` to select the best configuration!

---

## Additional Notes

- **Embedding Caching:** Embeddings are cached based on `--cache-embeddings` filename. Different pooling strategies require different cache files (e.g., `emb_cls.pt` vs `emb_mean.pt`).

- **Model Loading:** Models are downloaded from Hugging Face on first use and cached locally.

- **Memory Usage:** 
  - Base model: ~350MB GPU memory
  - Large model: ~1.2GB GPU memory
  - Giant model: ~4GB GPU memory

- **Speed Comparison (approximate, on GPU):**
  - Small: 100 images/sec
  - Base: 80 images/sec
  - Large: 40 images/sec
  - Giant: 20 images/sec

---

## Questions or Issues?

If you encounter problems or need help tuning for your specific use case:
1. Check the benchmark CSV files for quantitative comparisons
2. Verify image paths and formats are correct
3. Ensure sufficient GPU/CPU memory for larger models
4. Review error messages for specific issues


