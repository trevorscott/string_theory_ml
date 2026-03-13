import os
import argparse
import torch
import numpy as np
import random
from cytools import fetch_polytopes

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

# Full range of h11 in the Kreuzer-Skarke database is 1-491
KS_H11_MAX = 491
KS_H11_MIN = 1

def save_checkpoint(dataset, checkpoint_path):
    """Save current dataset to disk as a checkpoint."""
    torch.save(dataset, checkpoint_path)
    print(f"   💾 Checkpoint saved: {len(dataset)} universes → {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harvest CY manifolds where |h11 - h21| = 3 sampled uniformly across the KS landscape."
    )
    parser.add_argument('-u', '--universes', type=int, default=10000,
                        help="Target number of valid universes to harvest (default: 10000)")
    parser.add_argument('--h11_samples', type=int, default=100,
                        help="Number of h11 values to sample across KS range (default: 100)")
    parser.add_argument('--scaffolds_per_h11', type=int, default=200,
                        help="Max polytopes to fetch per h11 value (default: 200)")
    parser.add_argument('--triangulations_per_scaffold', type=int, default=500,
                        help="Max triangulations per polytope (default: 500)")
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help="Save checkpoint every N universes (default: 1000)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducible h11 sampling (default: 42)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint file to resume from")

    args = parser.parse_args()
    target_size = args.universes

    # Reproducible sampling across the full KS landscape
    random.seed(args.seed)
    h11_values = sorted(random.sample(range(KS_H11_MIN, KS_H11_MAX + 1), k=args.h11_samples))

    print(f"🌌 Launching Standard Model Harvest (Target: {target_size} universes)")
    print(f"📐 Constraint: |h11 - h21| = 3 (3 particle generations, χ = ±6)")
    print(f"🔭 Sampling {args.h11_samples} h11 values uniformly across KS range [{KS_H11_MIN}, {KS_H11_MAX}]")
    print(f"📊 h11 sample: {h11_values[:10]}... (seed={args.seed})")

    # Resume from checkpoint if provided
    universal_dataset = []
    checkpoint_path = f"data/standard_model_checkpoint.pt"

    if args.resume and os.path.exists(args.resume):
        print(f"\n♻️  Resuming from checkpoint: {args.resume}")
        universal_dataset = torch.load(args.resume, weights_only=False)
        print(f"   Loaded {len(universal_dataset)} universes from checkpoint.")
    
    failed_math_count = 0
    total_scaffolds_scanned = 0
    total_scaffolds_used = 0
    hodge_distribution = {}  # Track diversity of Hodge pairs harvested

    print(f"\n{'='*50}")

    for h11_val in h11_values:
        if len(universal_dataset) >= target_size:
            break

        print(f"\n--- Scanning h11={h11_val} ({h11_values.index(h11_val)+1}/{len(h11_values)}) ---")

        try:
            polys = fetch_polytopes(
                h11=h11_val,
                lattice="N",
                limit=args.scaffolds_per_h11,
                as_list=True
            )
        except Exception as e:
            print(f"  ⚠️  Could not fetch h11={h11_val}: {e}")
            continue

        if not polys:
            print(f"  ⚠️  No polytopes found for h11={h11_val}, skipping.")
            continue

        print(f"  📦 Got {len(polys)} scaffolds to screen.")

        qualifying = 0
        for p in polys:
            if len(universal_dataset) >= target_size:
                break

            total_scaffolds_scanned += 1

            try:
                h21_val = p.h21(lattice='N')

                if abs(h11_val - h21_val) != 3:
                    continue

                qualifying += 1
                total_scaffolds_used += 1
                euler = 2 * (h11_val - h21_val)
                hodge_key = f"({h11_val},{h21_val})"
                hodge_distribution[hodge_key] = hodge_distribution.get(hodge_key, 0)

                print(f"  ✅ Qualifying: (h11={h11_val}, h21={h21_val}), χ={euler}")

                t_gen = p.random_triangulations_fast(N=args.triangulations_per_scaffold)
                valid_from_this_shape = 0

                for t in t_gen:
                    if len(universal_dataset) >= target_size:
                        break

                    try:
                        simplices = [list(s) for s in t.simplices()]
                        physics = t.get_cy().intersection_numbers()

                        universal_dataset.append({
                            "scaffold_id": total_scaffolds_used,
                            "h11": h11_val,
                            "h21": h21_val,
                            "euler": euler,
                            "X_simplices": simplices,
                            "Y_physics": physics
                        })
                        valid_from_this_shape += 1
                        hodge_distribution[hodge_key] += 1

                        # Checkpoint saving
                        if len(universal_dataset) % args.checkpoint_interval == 0:
                            print(f"   📈 PROGRESS: {len(universal_dataset)} / {target_size} universes secured...")
                            save_checkpoint(universal_dataset, checkpoint_path)

                    except Exception:
                        failed_math_count += 1
                        continue

                print(f"  > Extracted {valid_from_this_shape} universes from (h11={h11_val}, h21={h21_val}).")

            except Exception:
                failed_math_count += 1
                continue

        print(f"  > {qualifying}/{len(polys)} scaffolds qualified for h11={h11_val}.")

    # Final save
    os.makedirs("data", exist_ok=True)
    output_filename = f"data/standard_model_{len(universal_dataset)}.pt"
    torch.save(universal_dataset, output_filename)

    # Clean up checkpoint if we finished successfully
    if len(universal_dataset) >= target_size and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🧹 Checkpoint cleaned up (harvest complete).")

    print("\n" + "="*50)
    print(f"✅ STANDARD MODEL HARVEST COMPLETE!")
    print(f"Total Universes Saved:      {len(universal_dataset)}")
    print(f"Total Scaffolds Scanned:    {total_scaffolds_scanned}")
    print(f"Qualifying Scaffolds Used:  {total_scaffolds_used}")
    print(f"Math Failures Skipped:      {failed_math_count}")
    print(f"Hodge Constraint:           |h11 - h21| = 3 (χ = ±6)")
    print(f"Unique Hodge Pairs Found:   {len(hodge_distribution)}")
    print(f"Hodge Distribution:         {hodge_distribution}")
    print(f"h11 Range Sampled:          {min(h11_values)} - {max(h11_values)}")
    print(f"Saved to:                   {output_filename}")
    print("="*50)