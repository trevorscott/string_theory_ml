import os
import argparse
import torch
import numpy as np
from torch_geometric.data import Data
from collections import Counter

# Prevent numpy serialization errors on Mac (Compatible with NumPy 1.x and 2.x)
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except AttributeError:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Convert raw Calabi-Yau geometry into 3D Graph structures.")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the raw harvested universes file (e.g., data/deep_space_10k.pt)")
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help="Path to save the processed graph dataset. Auto-generates if left blank.")
    
    args = parser.parse_args()
    
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        exit(1)

    # 2. Auto-generate output filename based on input
    if args.output is None:
        base_name = os.path.basename(input_file)
        # Translates 'deep_space_10k.pt' -> 'smart_graph_10k.pt'
        name_part = base_name.replace("deep_space_", "").replace(".pt", "")
        
        # Determine the directory of the input file to save alongside it
        input_dir = os.path.dirname(input_file)
        if input_dir == "":
            input_dir = "data" # fallback if run from root without path
            
        output_file = os.path.join(input_dir, f"smart_graph_{name_part}.pt")
    else:
        output_file = args.output

    
    print("🧠 Waking up the Smart Topological Weaver...")
    print(f"📥 Loading raw universes from: {input_file}")
    
    # Load the raw tensors/lists
    raw_data = torch.load(input_file, weights_only=False)
    print(f"📦 Found {len(raw_data)} universes to process.")
    
    # 3. Build Global Y-Vocabulary
    all_keys = set()
    for universe in raw_data:
        all_keys.update(universe["Y_physics"].keys())
    master_keys = sorted(list(all_keys))
    y_dim = len(master_keys)

    graph_dataset = []

    # 4. The Core Graph Weaving Loop
    for idx, universe in enumerate(raw_data):
        simplices = universe["X_simplices"]
        num_simplices = len(simplices)
        
        # --- A. CALCULATE TOPOLOGICAL FEATURES ---
        # Find out how "popular" each vertex is in the entire universe
        all_vertices = [v for simplex in simplices for v in simplex]
        vertex_counts = Counter(all_vertices)
        
        # Determine the Edges and Simplex Degrees first
        edges = []
        simplex_degrees = np.zeros(num_simplices)
        
        for i in range(num_simplices):
            for j in range(num_simplices):
                if i != j:
                    shared = len(set(simplices[i]).intersection(set(simplices[j])))
                    if shared >= 3:
                        edges.append([i, j])
                        simplex_degrees[i] += 1
                        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # Build the Smart Node Features (x)
        node_features = []
        for i, simplex in enumerate(simplices):
            # Feature 1: The degree of the simplex itself
            deg = simplex_degrees[i]
            
            # Features 2-6: The popularity of its 5 vertices (sorted so order doesn't matter!)
            v_pop = sorted([vertex_counts[v] for v in simplex], reverse=True)
            
            # Combine into a 6-number pure topological fingerprint
            feat = [deg] + v_pop
            node_features.append(feat)
            
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # --- B. TARGET PHYSICS (Y) ---
        y_dense = torch.zeros(1, y_dim)
        for key, value in universe["Y_physics"].items():
            if key in master_keys:
                key_idx = master_keys.index(key)
                y_dense[0, key_idx] = float(value)
                
        # --- C. BUILD THE GRAPH ---
        graph = Data(x=x, edge_index=edge_index, y=y_dense)
        graph_dataset.append(graph)
        
        # Dynamic update printing based on dataset size
        update_interval = 1000 if len(raw_data) >= 10000 else max(1, int(len(raw_data) / 10))
        if (idx + 1) % update_interval == 0:
            print(f"   > Engineered {idx + 1} / {len(raw_data)} smart universes...")

    # Ensure the data directory exists before saving
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    torch.save({"graphs": graph_dataset, "y_dim": y_dim}, output_file)
    
    print("\n" + "="*50)
    print(f"✅ SMART GRAPH CONVERSION COMPLETE!")
    print(f"Total Graphs Built: {len(graph_dataset)}")
    print(f"Physics Output Dimension (y_dim): {y_dim}")
    print(f"Saved to: {output_file}")
    print("="*50)