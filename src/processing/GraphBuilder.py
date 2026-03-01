import torch
import numpy as np
from torch_geometric.data import Data
import os

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

print("🕸️  Waking up the Topological Weaver...")
# Load the raw, unpadded dataset
raw_data = torch.load("universal_multiverse_5k.pt", weights_only=False)

# 1. Build Global Y-Vocabulary (Same as before to keep physics consistent)
all_keys = set()
for universe in raw_data:
    all_keys.update(universe["Y_physics"].keys())
master_keys = sorted(list(all_keys))
y_dim = len(master_keys)

print(f"Global Physics Targets: {y_dim} couplings.")

graph_dataset = []

for idx, universe in enumerate(raw_data):
    simplices = universe["X_simplices"]
    num_simplices = len(simplices)
    
    # --- A. NODE FEATURES ---
    # Each node is a simplex. Its "feature" is the 5 vertex numbers that define it.
    x = torch.tensor(simplices, dtype=torch.float32) 
    
    # --- B. EDGES (The Glue) ---
    edges = []
    # Loop through every pair of simplices to see if they touch
    for i in range(num_simplices):
        for j in range(num_simplices):
            if i != j:
                # How many vertices do they share?
                shared = len(set(simplices[i]).intersection(set(simplices[j])))
                # If they share 3 or more vertices, they are physically glued together
                if shared >= 3:
                    edges.append([i, j])
    
    # Format for PyTorch Geometric (shape: [2, num_edges])
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Failsafe for a disconnected universe (very rare in Calabi-Yau)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # --- C. TARGET PHYSICS (Y) ---
    y_dense = torch.zeros(1, y_dim) # Notice the shape is [1, 364]
    for key, value in universe["Y_physics"].items():
        if key in master_keys:
            key_idx = master_keys.index(key)
            y_dense[0, key_idx] = float(value)
            
    # --- D. BUILD THE GRAPH OBJECT ---
    graph = Data(x=x, edge_index=edge_index, y=y_dense)
    graph_dataset.append(graph)
    
    if (idx + 1) % 100 == 0:
        print(f"   > Woven {idx + 1} / {len(raw_data)} universes into graphs...")

# Save the new Graph Multiverse
torch.save({"graphs": graph_dataset, "y_dim": y_dim}, "graph_multiverse.pt")
print(f"\n✅ Graph Conversion Complete! Saved {len(graph_dataset)} 3D Web Universes.")