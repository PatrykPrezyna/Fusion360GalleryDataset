"""
Benchmarking utilities for comparing retrieval model performance.

This module provides comprehensive timing and performance metrics
for scientific evaluation of different image retrieval models.
"""

import os
import time
import json
import csv
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict


class BenchmarkTimer:
    """
    Comprehensive timing system for benchmarking retrieval models.
    
    Tracks:
    - Model loading time
    - Embedding extraction time (per image and total)
    - Per-query processing time
    - Similarity computation time
    - Total retrieval time
    """
    
    def __init__(self, model_name: str, dataset_name: str = "unknown"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.reset()
    
    def reset(self):
        """Reset all timing measurements."""
        self.model_loading_time = None
        self.embedding_extraction_start = None
        self.embedding_extraction_time = None
        self.num_images_embedded = 0
        self.query_times = []
        self.query_details = []  # List of dicts with detailed per-query info
        self.retrieval_start = None
        self.retrieval_end = None
        
    def start_model_loading(self):
        """Mark the start of model loading."""
        self.model_loading_start = time.time()
    
    def end_model_loading(self):
        """Mark the end of model loading."""
        if hasattr(self, 'model_loading_start'):
            self.model_loading_time = time.time() - self.model_loading_start
    
    def start_embedding_extraction(self):
        """Mark the start of embedding extraction."""
        self.embedding_extraction_start = time.time()
    
    def end_embedding_extraction(self, num_images: int):
        """Mark the end of embedding extraction."""
        if self.embedding_extraction_start is not None:
            self.embedding_extraction_time = time.time() - self.embedding_extraction_start
            self.num_images_embedded = num_images
    
    def start_retrieval(self):
        """Mark the start of retrieval phase."""
        self.retrieval_start = time.time()
    
    def start_query(self, query_idx: int, query_path: str):
        """Start timing a single query."""
        self.current_query_start = time.time()
        self.current_query_idx = query_idx
        self.current_query_path = query_path
    
    def end_query(self, num_results: int, ndcg_score: Optional[float] = None, 
                  num_relevant: Optional[int] = None, 
                  matches_in_topk: Optional[int] = None):
        """End timing a single query and record metrics."""
        if hasattr(self, 'current_query_start'):
            query_time = time.time() - self.current_query_start
            self.query_times.append(query_time)
            
            query_info = {
                'query_idx': self.current_query_idx,
                'query_path': self.current_query_path,
                'query_time_seconds': query_time,
                'num_results': num_results,
                'ndcg_score': ndcg_score,
                'num_relevant': num_relevant,
                'matches_in_topk': matches_in_topk,
            }
            self.query_details.append(query_info)
    
    def end_retrieval(self):
        """Mark the end of retrieval phase."""
        if self.retrieval_start is not None:
            self.retrieval_end = time.time()
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary statistics."""
        summary = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'model_loading_time_seconds': self.model_loading_time,
            'embedding_extraction_time_seconds': self.embedding_extraction_time,
            'num_images_embedded': self.num_images_embedded,
            'num_queries': len(self.query_times),
        }
        
        # Embedding extraction metrics
        if self.embedding_extraction_time and self.num_images_embedded > 0:
            summary['embedding_extraction_rate_images_per_second'] = (
                self.num_images_embedded / self.embedding_extraction_time
            )
            summary['embedding_extraction_time_per_image_seconds'] = (
                self.embedding_extraction_time / self.num_images_embedded
            )
        
        # Query processing metrics
        if len(self.query_times) > 0:
            query_times_array = np.array(self.query_times)
            summary['total_retrieval_time_seconds'] = (
                self.retrieval_end - self.retrieval_start if self.retrieval_end else sum(self.query_times)
            )
            summary['query_time_mean_seconds'] = float(np.mean(query_times_array))
            summary['query_time_median_seconds'] = float(np.median(query_times_array))
            summary['query_time_std_seconds'] = float(np.std(query_times_array))
            summary['query_time_min_seconds'] = float(np.min(query_times_array))
            summary['query_time_max_seconds'] = float(np.max(query_times_array))
            summary['query_time_p25_seconds'] = float(np.percentile(query_times_array, 25))
            summary['query_time_p75_seconds'] = float(np.percentile(query_times_array, 75))
            summary['query_time_p95_seconds'] = float(np.percentile(query_times_array, 95))
            summary['query_time_p99_seconds'] = float(np.percentile(query_times_array, 99))
            
            # Throughput metrics
            total_retrieval_time = summary['total_retrieval_time_seconds']
            summary['queries_per_second'] = len(self.query_times) / total_retrieval_time if total_retrieval_time > 0 else 0
            
            # NDCG statistics if available
            ndcg_scores = [q['ndcg_score'] for q in self.query_details if q['ndcg_score'] is not None]
            if ndcg_scores:
                ndcg_array = np.array(ndcg_scores)
                summary['ndcg_mean'] = float(np.mean(ndcg_array))
                summary['ndcg_median'] = float(np.median(ndcg_array))
                summary['ndcg_std'] = float(np.std(ndcg_array))
                summary['ndcg_min'] = float(np.min(ndcg_array))
                summary['ndcg_max'] = float(np.max(ndcg_array))
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of all metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print(f"BENCHMARK SUMMARY - {self.model_name.upper()}")
        print("="*70)
        
        # Model loading
        if summary['model_loading_time_seconds'] is not None:
            print(f"\nModel Loading:")
            print(f"  Time: {summary['model_loading_time_seconds']:.3f} seconds")
        
        # Embedding extraction
        if summary['embedding_extraction_time_seconds'] is not None:
            print(f"\nEmbedding Extraction:")
            print(f"  Total time: {summary['embedding_extraction_time_seconds']:.3f} seconds")
            print(f"  Images processed: {summary['num_images_embedded']}")
            print(f"  Rate: {summary.get('embedding_extraction_rate_images_per_second', 0):.2f} images/second")
            print(f"  Time per image: {summary.get('embedding_extraction_time_per_image_seconds', 0)*1000:.2f} ms")
        
        # Query processing
        if summary['num_queries'] > 0:
            print(f"\nQuery Processing:")
            print(f"  Total queries: {summary['num_queries']}")
            print(f"  Total retrieval time: {summary['total_retrieval_time_seconds']:.3f} seconds")
            print(f"  Throughput: {summary['queries_per_second']:.2f} queries/second")
            print(f"\n  Per-Query Statistics:")
            print(f"    Mean:   {summary['query_time_mean_seconds']*1000:.2f} ms")
            print(f"    Median: {summary['query_time_median_seconds']*1000:.2f} ms")
            print(f"    Std:    {summary['query_time_std_seconds']*1000:.2f} ms")
            print(f"    Min:    {summary['query_time_min_seconds']*1000:.2f} ms")
            print(f"    Max:    {summary['query_time_max_seconds']*1000:.2f} ms")
            print(f"    P25:    {summary['query_time_p25_seconds']*1000:.2f} ms")
            print(f"    P75:    {summary['query_time_p75_seconds']*1000:.2f} ms")
            print(f"    P95:    {summary['query_time_p95_seconds']*1000:.2f} ms")
            print(f"    P99:    {summary['query_time_p99_seconds']*1000:.2f} ms")
            
            # NDCG if available
            if 'ndcg_mean' in summary:
                print(f"\n  Retrieval Quality (NDCG):")
                print(f"    Mean:   {summary['ndcg_mean']:.4f}")
                print(f"    Median: {summary['ndcg_median']:.4f}")
                print(f"    Std:    {summary['ndcg_std']:.4f}")
                print(f"    Min:    {summary['ndcg_min']:.4f}")
                print(f"    Max:    {summary['ndcg_max']:.4f}")
        
        print("="*70 + "\n")
    
    def save_json(self, filepath: str):
        """Save detailed results to JSON file."""
        output = {
            'summary': self.get_summary(),
            'query_details': self.query_details,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Benchmark results saved to: {filepath}")
    
    def save_csv_summary(self, filepath: str):
        """Save summary statistics to CSV file."""
        summary = self.get_summary()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        
        print(f"Summary CSV saved to: {filepath}")
    
    def save_csv_detailed(self, filepath: str):
        """Save per-query details to CSV file."""
        if not self.query_details:
            print("No query details to save.")
            return
        
        fieldnames = self.query_details[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.query_details)
        
        print(f"Detailed CSV saved to: {filepath}")


def compare_benchmarks(benchmarks: List[BenchmarkTimer], output_dir: str = "benchmark_comparison"):
    """
    Compare multiple benchmark results and generate comparison report.
    
    Args:
        benchmarks: List of BenchmarkTimer objects to compare
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect summaries
    summaries = [b.get_summary() for b in benchmarks]
    
    # Create comparison table
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    
    # Table header
    print(f"\n{'Metric':<40} " + "".join([f"{b.model_name:>15}" for b in benchmarks]))
    print("-" * 100)
    
    # Key metrics to compare
    metrics_to_compare = [
        ('Model Loading (s)', 'model_loading_time_seconds'),
        ('Embedding Extraction (s)', 'embedding_extraction_time_seconds'),
        ('Embedding Rate (img/s)', 'embedding_extraction_rate_images_per_second'),
        ('Total Retrieval Time (s)', 'total_retrieval_time_seconds'),
        ('Queries/Second', 'queries_per_second'),
        ('Query Time Mean (ms)', lambda s: s.get('query_time_mean_seconds', 0) * 1000),
        ('Query Time Median (ms)', lambda s: s.get('query_time_median_seconds', 0) * 1000),
        ('Query Time Std (ms)', lambda s: s.get('query_time_std_seconds', 0) * 1000),
        ('NDCG Mean', 'ndcg_mean'),
        ('NDCG Median', 'ndcg_median'),
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        if callable(metric_key):
            values = [metric_key(s) for s in summaries]
        else:
            values = [s.get(metric_key, 0) for s in summaries]
        
        row = f"{metric_name:<40} "
        for val in values:
            if isinstance(val, float):
                row += f"{val:>15.3f}"
            elif val is None:
                row += f"{'N/A':>15}"
            else:
                row += f"{val:>15}"
        print(row)
    
    print("="*100 + "\n")
    
    # Save comparison JSON
    comparison_path = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump({
            'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': [b.model_name for b in benchmarks],
            'summaries': summaries,
        }, f, indent=2)
    
    print(f"Comparison results saved to: {comparison_path}")
