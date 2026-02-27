from cytools import fetch_polytopes
import torch
import numpy as np

def build_balanced_dataset():
    x_list, y_list = [], []
    # Targeted ranges to cover the 'corners' of the landscape
    targets = [
        {'h11': 1, 'limit': 5000},    # Simple Quintic-like
        {'h11': 10, 'limit': 5000},   # Mid-range
        {'h11': 50, 'limit': 2000},   # Complex
        {'h21': 50, 'limit': 2000}    # Mirror Complex
    ]
    
    for target in targets:
        print(f"🏝️ Harvesting from neighborhood: {target}...")
        g = fetch_polytopes(lattice="N", as_list=False, **target)
        for p in g:
            try:
                h11, h21 = p.h11(lattice="N"), p.h21(lattice="N")
                verts = p.vertices()
                if verts.shape[0] <= 36:
                    # NORMALIZATION: Scale coordinates down
                    padded = np.zeros((36, 4))
                    # Normalizing to approx -1 to 1 range based on KS bounds
                    padded[:verts.shape[0], :] = verts / 10.0 
                    x_list.append(padded)
                    y_list.append([h11, h21])
            except: continue

    X = torch.tensor(np.array(x_list), dtype=torch.float32)
    Y = torch.tensor(np.array(y_list), dtype=torch.float32)
    torch.save((X, Y), "balanced_landscape.pt")
    print(f"✅ Balanced dataset saved with {len(X)} samples.")

build_balanced_dataset()