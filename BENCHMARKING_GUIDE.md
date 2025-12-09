# Benchmarking Guide for Image Retrieval Models

This guide explains how to benchmark and compare different image retrieval models (VAE, CLIP, DINOv2) for scientific evaluation.

## Overview

The benchmarking system provides comprehensive timing and performance metrics suitable for scientific papers, including:

- **Model Loading Time**: Time to load the model into memory
- **Embedding Extraction Time**: Time to extract embeddings for all images in the database
- **Per-Query Processing Time**: Detailed statistics (mean, median, std, percentiles) for each query
- **Throughput Metrics**: Queries per second, images per second
- **Quality Metrics**: NDCG scores with statistical analysis
- **Export Formats**: JSON and CSV for further analysis

## Quick Start

### 1. Run Benchmarks for Each Model

#### DINOv2 Model
```powershell
python retrieve_similar_images_use_pretrained_model_dinov2.py `
    --query "output_data/Test_query/R1" `
    --image-dir "output_data/14_views_10000_mechanical" `
    --top-k 5 `
    --benchmark `
    --dataset-name "R1" `
    --cache-embeddings "embeddings_dino.pt" `
    --cache-paths "image_paths_dino.txt"
```

#### CLIP Model
```powershell
python retrieve_similar_images_use_pretrained_model.py `
    --query "output_data/Test_query/R1" `
    --image-dir "output_data/14_views_10000_mechanical" `
    --top-k 5 `
    --benchmark `
    --dataset-name "R1" `
    --cache-embeddings "embeddings_clip.pt" `
    --cache-paths "image_paths_clip.txt"
```

#### VAE Model
```powershell
python retrieve_similar_images.py `
    --query "output_data/Test_query/R1" `
    --output-dir "vae_retrieval_output" `
    --top-k 5 `
    --benchmark `
    --dataset-name "R1"
```

### 2. Output Files

Each benchmark run creates three files in the `benchmark_results` folder:

- **`benchmark_{dataset}_{model}_{timestamp}.json`**: Complete detailed results
- **`benchmark_{dataset}_{model}_summary_{timestamp}.csv`**: Summary statistics (one row)
- **`benchmark_{dataset}_{model}_detailed_{timestamp}.csv`**: Per-query details

## Metrics Explained

### Timing Metrics

1. **Model Loading Time**: One-time cost to load the model
   - Important for: Cold start scenarios, deployment considerations

2. **Embedding Extraction Time**: Time to process all database images
   - **Embedding Rate**: Images per second (throughput)
   - **Time per Image**: Milliseconds per image
   - Important for: Database indexing time, scalability

3. **Query Processing Time**: Time to process each query
   - **Mean/Median**: Central tendency
   - **Std**: Variability
   - **Percentiles (P25, P75, P95, P99)**: Distribution analysis
   - **Min/Max**: Best/worst case performance
   - Important for: User experience, real-time applications

4. **Throughput**: Queries per second
   - Important for: System capacity planning

### Quality Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain
  - Mean, median, std, min, max
  - Important for: Retrieval accuracy comparison

## Scientific Reporting Best Practices

### 1. Fair Comparison Methodology

**Use the same test conditions:**
- Same query images
- Same database images
- Same hardware (CPU/GPU)
- Same batch sizes (if applicable)
- Same number of queries

**Warm-up runs:**
- Run each model once before timing (to account for JIT compilation, GPU warm-up)
- Or explicitly report cold vs warm performance

**Multiple runs:**
- Run each configuration multiple times (e.g., 3-5 runs)
- Report mean ± standard deviation
- Use statistical tests if comparing models

### 2. Reporting Format

#### Table Format (Recommended)

| Metric | DINOv2 | CLIP | VAE |
|--------|--------|------|-----|
| Model Loading (s) | X.XX | X.XX | X.XX |
| Embedding Extraction (s) | X.XX | X.XX | X.XX |
| Embedding Rate (img/s) | X.XX | X.XX | X.XX |
| Query Time Mean (ms) | X.XX | X.XX | X.XX |
| Query Time Median (ms) | X.XX | X.XX | X.XX |
| Query Time Std (ms) | X.XX | X.XX | X.XX |
| Queries/Second | X.XX | X.XX | X.XX |
| NDCG@5 Mean | X.XXXX | X.XXXX | X.XXXX |

#### Text Format

"On dataset R1 with 1000 database images and 50 queries:
- DINOv2 achieved a mean query time of 45.2 ms (±3.1 ms) with NDCG@5 of 0.8234
- CLIP achieved a mean query time of 38.7 ms (±2.8 ms) with NDCG@5 of 0.7891
- VAE achieved a mean query time of 52.1 ms (±4.2 ms) with NDCG@5 of 0.8012"

### 3. Statistical Analysis

For rigorous comparison, consider:

1. **Paired t-test**: Compare query times between models
2. **Effect size**: Cohen's d to measure practical significance
3. **Confidence intervals**: 95% CI for mean values
4. **Non-parametric tests**: Mann-Whitney U test if data not normally distributed

### 4. What to Report

**Must report:**
- Hardware specifications (CPU, GPU, RAM)
- Software versions (PyTorch, CUDA, etc.)
- Dataset sizes (number of database images, queries)
- Mean and standard deviation of key metrics
- Number of runs/averaging method

**Should report:**
- Median (robust to outliers)
- Percentiles (P95, P99 for worst-case analysis)
- Throughput metrics
- Memory usage (if relevant)

**Nice to have:**
- Speed vs accuracy trade-off curves
- Scalability analysis (performance vs database size)
- Ablation studies (effect of batch size, etc.)

## Example Workflow

### Step 1: Run Benchmarks for All Models

```powershell
# Test dataset R1
python retrieve_similar_images_use_pretrained_model_dinov2.py --query "output_data/Test_query/R1" --image-dir "output_data/14_views_10000_mechanical" --benchmark --dataset-name "R1" --cache-embeddings "embeddings_dino.pt"
python retrieve_similar_images_use_pretrained_model.py --query "output_data/Test_query/R1" --image-dir "output_data/14_views_10000_mechanical" --benchmark --dataset-name "R1" --cache-embeddings "embeddings_clip.pt"
python retrieve_similar_images.py --query "output_data/Test_query/R1" --output-dir "vae_retrieval_output" --benchmark --dataset-name "R1"

# Test dataset R2
python retrieve_similar_images_use_pretrained_model_dinov2.py --query "output_data/Test_query/R2" --image-dir "output_data/14_views_10000_mechanical" --benchmark --dataset-name "R2" --cache-embeddings "embeddings_dino.pt"
python retrieve_similar_images_use_pretrained_model.py --query "output_data/Test_query/R2" --image-dir "output_data/14_views_10000_mechanical" --benchmark --dataset-name "R2" --cache-embeddings "embeddings_clip.pt"
python retrieve_similar_images.py --query "output_data/Test_query/R2" --output-dir "vae_retrieval_output" --benchmark --dataset-name "R2"
```

### Step 2: Analyze Results

Load the CSV files into your analysis tool (Python pandas, R, Excel):

```python
import pandas as pd

# Load summary files
dino_summary = pd.read_csv("benchmark_results/benchmark_R1_dino_summary_*.csv")
clip_summary = pd.read_csv("benchmark_results/benchmark_R1_clip_summary_*.csv")
vae_summary = pd.read_csv("benchmark_results/benchmark_R1_vae_summary_*.csv")

# Compare key metrics
comparison = pd.DataFrame({
    'DINOv2': [dino_summary['query_time_mean_seconds'].values[0] * 1000,
               dino_summary['ndcg_mean'].values[0]],
    'CLIP': [clip_summary['query_time_mean_seconds'].values[0] * 1000,
             clip_summary['ndcg_mean'].values[0]],
    'VAE': [vae_summary['query_time_mean_seconds'].values[0] * 1000,
            vae_summary['ndcg_mean'].values[0]]
}, index=['Query Time (ms)', 'NDCG@5'])

print(comparison)
```

### Step 3: Create Visualizations

Create plots for:
- Box plots of query times
- Bar charts comparing mean metrics
- Speed-accuracy trade-off plots
- Scalability curves

## Common Pitfalls to Avoid

1. **Not accounting for caching**: Always report whether embeddings were cached or computed
2. **Single run**: Always run multiple times and average
3. **Different hardware**: Ensure all models tested on same hardware
4. **Different dataset sizes**: Keep database size constant across models
5. **Ignoring variance**: Report standard deviation, not just mean
6. **Not reporting hardware**: Always specify CPU/GPU model and specifications

## Advanced: Automated Comparison Script

You can create a script to automatically compare all models:

```python
from benchmark_utils import BenchmarkTimer, compare_benchmarks
import json
import glob

# Load all benchmark results
benchmarks = []
for json_file in glob.glob("benchmark_results/benchmark_*_*.json"):
    with open(json_file) as f:
        data = json.load(f)
        timer = BenchmarkTimer(model_name=data['summary']['model_name'],
                              dataset_name=data['summary']['dataset_name'])
        timer.query_times = [q['query_time_seconds'] for q in data['query_details']]
        timer.query_details = data['query_details']
        timer.model_loading_time = data['summary'].get('model_loading_time_seconds')
        timer.embedding_extraction_time = data['summary'].get('embedding_extraction_time_seconds')
        timer.num_images_embedded = data['summary'].get('num_images_embedded', 0)
        timer.retrieval_start = 0  # Approximate
        timer.retrieval_end = sum(timer.query_times)
        benchmarks.append(timer)

# Compare
compare_benchmarks(benchmarks, output_dir="comparison_results")
```

## Questions to Answer in Your Paper

1. **Which model is fastest?** (Query time, throughput)
2. **Which model is most accurate?** (NDCG scores)
3. **What is the speed-accuracy trade-off?** (Plot query time vs NDCG)
4. **How does performance scale?** (Test with different database sizes)
5. **What are the computational requirements?** (Memory, GPU usage)
6. **Is the difference statistically significant?** (Statistical tests)

## References

- Use standard statistical methods (t-tests, effect sizes)
- Report confidence intervals
- Consider multiple evaluation metrics (not just NDCG)
- Document all experimental conditions thoroughly
