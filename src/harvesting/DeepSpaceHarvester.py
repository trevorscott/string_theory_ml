import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from cytools import fetch_polytopes

# Prevent numpy serialization errors on Mac
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

target_size = 50000
print(f"🌌 Launching MASSIVE Deep Space Harvest (Target: {target_size} universes)...")

# We need ~3,500 scaffolds to hit 50k based on the 16-universe average
polys = fetch_polytopes(h11=10, limit=3500, as_list=True)
print(f"📦 Downloaded {len(polys)} complex candidate scaffolds.")

universal_dataset = []
failed_math_count = 0

for p_idx, p in enumerate(polys):
    if len(universal_dataset) >= target_size:
        break
        
    try:
        h21_val = p.h21(lattice='N')
        print(f"\nShattering Scaffold #{p_idx} (Hodge: 10, {h21_val})...")
    except:
        print(f"\nShattering Scaffold #{p_idx} (Hodge: 10)...")
    
    t_gen = p.random_triangulations_fast(N=3000)
    
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
            
            # Print a satisfying update every 1,000 universes
            if len(universal_dataset) % 1000 == 0:
                print(f"   📈 PROGRESS: {len(universal_dataset)} / {target_size} universes secured...")
                
        except Exception:
            failed_math_count += 1
            continue
            
    print(f"  > Extracted {valid_from_this_shape} universes from Scaffold #{p_idx}.")

torch.save(universal_dataset, "deep_space_50k.pt")

print("\n" + "="*40)
print(f"✅ MASSIVE HARVEST COMPLETE!")
print(f"Total Universes Saved: {len(universal_dataset)}")
print(f"Total Scaffolds Used: {p_idx + 1}")
print(f"Math Failures Skipped: {failed_math_count}")
print("="*40)