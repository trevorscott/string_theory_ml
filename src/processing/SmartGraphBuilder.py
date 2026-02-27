import os
import torch
import numpy as np
from torch_geometric.data import Data
from collections import Counter


os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

print("🧠 Waking up the Smart Topological Weaver...")
raw_data = torch.load("deep_space_10k.pt", weights_only=False)
# 1. Build Global Y-Vocabulary
all_keys = set()
for universe in raw_data:
    all_keys.update(universe["Y_physics"].keys())
master_keys = sorted(list(all_keys))
y_dim = len(master_keys)

graph_dataset = []

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
    
    if (idx + 1) % 100 == 0:
        print(f"   > Engineered {idx + 1} / {len(raw_data)} smart universes...")

torch.save({"graphs": graph_dataset, "y_dim": y_dim}, "smart_graph_multiverse.pt")
print(f"\n✅ Smart Graph Conversion Complete! Node features are now purely geometric.")