"""
Script to create a comparison table from JSON benchmark results.
Generates a LaTeX table suitable for scientific papers.
"""

import json
import os
import glob
from pathlib import Path

def extract_model_name(filename):
    """Extract clean model name from filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Clean up the name for display
    name = name.replace("_", " ").replace(".json", "")
    return name

def load_json_data(filepath):
    """Load JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def extract_part_data(data, row_id):
    """Extract data for a specific part (row_id)."""
    # Handle different JSON structures
    if isinstance(data, dict):
        # Find the first key that contains an array
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("row_id") == row_id:
                        return {
                            "ndcg_at_10": item.get("ndcg_at_10", 0.0),
                            "rank": item.get("match_details", {}).get("rank", None),
                            "similarity": item.get("match_details", {}).get("score", None)
                        }
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("row_id") == row_id:
                return {
                    "ndcg_at_10": item.get("ndcg_at_10", 0.0),
                    "rank": item.get("match_details", {}).get("rank", None),
                    "similarity": item.get("match_details", {}).get("score", None)
                }
    return None

def create_comparison_table(json_folder):
    """Create comparison table from all JSON files in folder."""
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    
    if not json_files:
        print(f"No JSON files found in {json_folder}")
        return
    
    # Collect data for all models
    models_data = {}
    
    for json_file in json_files:
        model_name = extract_model_name(json_file)
        data = load_json_data(json_file)
        
        if data is None:
            continue
        
        # Extract data for parts 1-4
        parts_data = {}
        for part_id in [1, 2, 3, 4]:
            part_data = extract_part_data(data, part_id)
            if part_data:
                parts_data[part_id] = part_data
        
        if parts_data:
            models_data[model_name] = parts_data
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE FOR SCIENTIFIC PAPER")
    print("="*80 + "\n")
    
    # LaTeX table header
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Comparison of retrieval performance across different models}")
    print("\\label{tab:model_comparison}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l" + "|c c c" * 4 + "}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{Part 1}} & \\multicolumn{3}{c|}{\\textbf{Part 2}} & \\multicolumn{3}{c|}{\\textbf{Part 3}} & \\multicolumn{3}{c|}{\\textbf{Part 4}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
    print(" & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. \\\\")
    print("\\midrule")
    
    # Table rows
    for model_name in sorted(models_data.keys()):
        parts_data = models_data[model_name]
        row = [model_name]
        
        for part_id in [1, 2, 3, 4]:
            if part_id in parts_data:
                part = parts_data[part_id]
                ndcg = part.get("ndcg_at_10", 0.0)
                rank = part.get("rank", "N/A")
                sim = part.get("similarity", None)
                
                # Format values
                ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
                rank_str = str(rank) if rank is not None else "N/A"
                sim_str = f"{sim:.3f}" if sim is not None else "N/A"
                
                row.extend([ndcg_str, rank_str, sim_str])
            else:
                row.extend(["N/A", "N/A", "N/A"])
        
        # Format row for LaTeX
        row_str = " & ".join(str(x) for x in row) + " \\\\"
        print(row_str)
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table*}")
    
    # Also generate a CSV version
    print("\n" + "="*80)
    print("CSV TABLE")
    print("="*80 + "\n")
    
    # CSV header
    csv_header = ["Model"]
    for part_id in [1, 2, 3, 4]:
        csv_header.extend([f"Part{part_id}_NDCG@10", f"Part{part_id}_Rank", f"Part{part_id}_Similarity"])
    print(",".join(csv_header))
    
    # CSV rows
    for model_name in sorted(models_data.keys()):
        parts_data = models_data[model_name]
        row = [model_name]
        
        for part_id in [1, 2, 3, 4]:
            if part_id in parts_data:
                part = parts_data[part_id]
                ndcg = part.get("ndcg_at_10", 0.0)
                rank = part.get("rank", "N/A")
                sim = part.get("similarity", None)
                
                ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
                rank_str = str(rank) if rank is not None else "N/A"
                sim_str = f"{sim:.3f}" if sim is not None else "N/A"
                
                row.extend([ndcg_str, rank_str, sim_str])
            else:
                row.extend(["N/A", "N/A", "N/A"])
        
        print(",".join(str(x) for x in row))
    
    # Save to files
    output_dir = os.path.dirname(json_folder) if json_folder != "." else "."
    latex_file = os.path.join(output_dir, "comparison_table.tex")
    csv_file = os.path.join(output_dir, "comparison_table.csv")
    
    # Write LaTeX file
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of retrieval performance across different models}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{l" + "|c c c" * 4 + "}\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{Part 1}} & \\multicolumn{3}{c|}{\\textbf{Part 2}} & \\multicolumn{3}{c|}{\\textbf{Part 3}} & \\multicolumn{3}{c|}{\\textbf{Part 4}} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}\n")
        f.write(" & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. & NDCG@10 & Rank & Sim. \\\\\n")
        f.write("\\midrule\n")
        
        for model_name in sorted(models_data.keys()):
            parts_data = models_data[model_name]
            row = [model_name]
            
            for part_id in [1, 2, 3, 4]:
                if part_id in parts_data:
                    part = parts_data[part_id]
                    ndcg = part.get("ndcg_at_10", 0.0)
                    rank = part.get("rank", "N/A")
                    sim = part.get("similarity", None)
                    
                    ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
                    rank_str = str(rank) if rank is not None else "N/A"
                    sim_str = f"{sim:.3f}" if sim is not None else "N/A"
                    
                    row.extend([ndcg_str, rank_str, sim_str])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            
            row_str = " & ".join(str(x) for x in row) + " \\\\\n"
            f.write(row_str)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table*}\n")
    
    # Write CSV file
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(",".join(csv_header) + "\n")
        
        for model_name in sorted(models_data.keys()):
            parts_data = models_data[model_name]
            row = [model_name]
            
            for part_id in [1, 2, 3, 4]:
                if part_id in parts_data:
                    part = parts_data[part_id]
                    ndcg = part.get("ndcg_at_10", 0.0)
                    rank = part.get("rank", "N/A")
                    sim = part.get("similarity", None)
                    
                    ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "N/A"
                    rank_str = str(rank) if rank is not None else "N/A"
                    sim_str = f"{sim:.3f}" if sim is not None else "N/A"
                    
                    row.extend([ndcg_str, rank_str, sim_str])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            
            f.write(",".join(str(x) for x in row) + "\n")
    
    print(f"\n✓ LaTeX table saved to: {latex_file}")
    print(f"✓ CSV table saved to: {csv_file}")
    print(f"\nTotal models processed: {len(models_data)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "output_data/Test_query/final_results"
    
    if not os.path.exists(folder):
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    create_comparison_table(folder)

