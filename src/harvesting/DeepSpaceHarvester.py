import os
import argparse
import torch
import numpy as np
from cytools import fetch_polytopes

# Prevent numpy serialization errors on Mac
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Mine the Calabi-Yau landscape for stable universes.")
    parser.add_argument('-u', '--universes', type=int, default=1000, 
                        help="The exact number of valid universes to harvest (default: 1000)")
    
    args = parser.parse_args()
    target_size = args.universes

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    print(f"🌌 Launching MASSIVE Deep Space Harvest (Target: {target_size} universes)...")

    # Estimate scaffolds needed (assuming a conservative ~10 valid universes per scaffold)
    estimated_scaffolds = max(100, int(target_size / 10) + 500)
    
    polys = fetch_polytopes(h11=10, limit=estimated_scaffolds, as_list=True)
    print(f"📦 Downloaded {len(polys)} complex candidate scaffolds.")

    universal_dataset = []
    failed_math_count = 0
    p_idx = 0

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
                
                # Dynamic satisfying updates (every 1000 for big runs, every 10% for small runs)
                update_interval = 1000 if target_size >= 1000 else max(1, int(target_size / 10))
                if len(universal_dataset) % update_interval == 0:
                    print(f"   📈 PROGRESS: {len(universal_dataset)} / {target_size} universes secured...")
                    
            except Exception:
                failed_math_count += 1
                continue
                
        print(f"  > Extracted {valid_from_this_shape} universes from Scaffold #{p_idx}.")

    # Ensure the data directory exists before trying to save!
    os.makedirs("data", exist_ok=True)
    
    # Save dynamically based on target size
    output_filename = f"data/deep_space_{target_size}.pt"
    torch.save(universal_dataset, output_filename)

    print("\n" + "="*40)
    print(f"✅ MASSIVE HARVEST COMPLETE!")
    print(f"Total Universes Saved: {len(universal_dataset)}")
    print(f"Total Scaffolds Used: {p_idx + 1}")
    print(f"Math Failures Skipped: {failed_math_count}")
    print(f"Saved to: {output_filename}")
    print("="*40)