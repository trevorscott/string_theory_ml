import torch
import numpy as np
from cytools import fetch_polytopes
import os


target_size = 5000
print(f"🚜 Starting the Quota Harvester (Target: {target_size} universes)...")

# Fetch a massive pool of scaffolds to ensure we don't run out
polys = fetch_polytopes(h11=7, limit=200, as_list=True)
print(f"📦 Downloaded {len(polys)} candidate scaffolds.")

universal_dataset = []
failed_math_count = 0

for p_idx, p in enumerate(polys):
    if len(universal_dataset) >= target_size:
        break # We hit our 5,000 quota!
        
    print(f"\nShattering Scaffold #{p_idx} (Hodge: {p.h11(lattice='N')}, {p.h21(lattice='N')})...")
    
    # Try to pull up to 1000 triangulations from this single shape
    t_gen = p.random_triangulations_fast(N=1000)
    
    valid_from_this_shape = 0
    for t in t_gen:
        if len(universal_dataset) >= target_size:
            break
            
        try:
            simplices = [list(s) for s in t.simplices()]
            physics = t.get_cy().intersection_numbers()
            
            universal_dataset.append({
                "scaffold_id": p_idx,
                "X_simplices": simplices,
                "Y_physics": physics
            })
            valid_from_this_shape += 1
            
            # Print an update every 500 universes
            if len(universal_dataset) % 500 == 0:
                print(f"   📈 Progress: {len(universal_dataset)} / {target_size} universes collected...")
                
        except Exception:
            failed_math_count += 1
            continue
            
    print(f"  > Extracted {valid_from_this_shape} universes from Scaffold #{p_idx}.")

# Save the raw dataset
torch.save(universal_dataset, "universal_multiverse_5k.pt")

print("\n" + "="*40)
print(f"✅ REAL HARVEST COMPLETE!")
print(f"Total Universes Saved: {len(universal_dataset)}")
print(f"Total Scaffolds Used: {universal_dataset[-1]['scaffold_id'] + 1}")
print(f"Math Failures Skipped: {failed_math_count}")
print("="*40)