#!/usr/bin/env python3
"""
test_pipeline.py — End-to-end smoke test for String Theory ML.

Run this after setup to verify all three pipeline stages work.
Uses a small batch size (20 universes) to complete quickly.

Usage:
    python test_pipeline.py

Requirements:
    - Conda environment with cytools, pytorch, torch_geometric installed
    - Run from the repo root directory
"""

import os
import sys
import subprocess
import time

DIVIDER = "=" * 60

def run(cmd, description):
    """Run a command and report pass/fail."""
    print(f"\n{DIVIDER}")
    print(f"  {description}")
    print(f"  $ {cmd}")
    print(DIVIDER)
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n  ❌ FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        return False
    else:
        print(f"\n  ✅ PASSED ({elapsed:.1f}s)")
        return True


def check_file(path, description):
    """Verify an output file exists and is non-empty."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✅ {description}: {path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  ❌ {description}: {path} NOT FOUND or empty")
        return False


def main():
    print("\n" + DIVIDER)
    print("  STRING THEORY ML — End-to-End Pipeline Test")
    print(DIVIDER)
    
    # Ensure we're in the repo root
    if not os.path.exists("src/harvesting/DeepSpaceHarvester.py"):
        print("  ❌ Run this script from the repo root directory.")
        sys.exit(1)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    results = []
    
    # ──────────────────────────────────────────────────────────
    # STAGE 1: Harvest
    # ──────────────────────────────────────────────────────────
    print("\n\n" + "━" * 60)
    print("  STAGE 1: HARVESTING")
    print("━" * 60)
    
    # Use a small batch for testing. Adjust -u up for a real run.
    harvest_ok = run(
        "python src/harvesting/DeepSpaceHarvester.py -u 20",
        "Harvesting 20 Standard Model candidate universes"
    )
    results.append(("Harvester", harvest_ok))
    
    # Find the output file (filename includes the count)
    harvest_file = None
    if harvest_ok:
        for f in sorted(os.listdir("data")):
            if f.startswith("standard_model") and f.endswith(".pt"):
                harvest_file = os.path.join("data", f)
        
        if harvest_file is None:
            # Fall back to older naming convention
            for f in sorted(os.listdir("data")):
                if f.startswith("deep_space") and f.endswith(".pt"):
                    harvest_file = os.path.join("data", f)
        
        if harvest_file:
            check_file(harvest_file, "Harvested dataset")
        else:
            print("  ⚠️  No .pt file found in data/. Check harvester output.")
            harvest_ok = False
    
    if not harvest_ok or not harvest_file:
        print("\n  ❌ Harvesting failed. Cannot continue pipeline test.")
        print("     Check that CYTools is installed: conda install -c conda-forge cytools")
        print_summary(results)
        sys.exit(1)
    
    # ──────────────────────────────────────────────────────────
    # STAGE 2: Forward Problem (BottNet Oracle)
    # ──────────────────────────────────────────────────────────
    print("\n\n" + "━" * 60)
    print("  STAGE 2: FORWARD PROBLEM (BottNet Oracle)")
    print("━" * 60)
    
    # 2a: Build graphs
    graph_ok = run(
        f"python src/processing/SmartGraphBuilder.py -i {harvest_file}",
        "Converting raw simplices to graph dataset"
    )
    results.append(("SmartGraphBuilder", graph_ok))
    
    # SmartGraphBuilder prefixes "smart_graph_" to the input filename
    graph_file = None
    if graph_ok:
        basename = os.path.basename(harvest_file)
        expected = os.path.join("data", f"smart_graph_{basename}")
        if os.path.exists(expected):
            graph_file = expected
        else:
            # Fallback: find any smart_graph file
            for f in sorted(os.listdir("data")):
                if f.startswith("smart_graph") and f.endswith(".pt"):
                    graph_file = os.path.join("data", f)
        
        if graph_file:
            check_file(graph_file, "Graph dataset")
        else:
            print("  ⚠️  No smart_graph .pt file found.")
            graph_ok = False
    
    # 2b: Train BottNet (short run for testing)
    if graph_ok and graph_file:
        train_ok = run(
            f"python architectures/v1_cvae/training/TrainGraphModel.py -i {graph_file} -e 10 -b 8",
            "Training BottNet (10 epochs, batch=8)"
        )
        results.append(("BottNet Training", train_ok))
        
        if train_ok:
            check_file("checkpoints/gnn_universe_model.pth", "BottNet checkpoint")
        
        # 2c: Oracle inference
        if train_ok:
            oracle_ok = run(
                f"python architectures/v1_cvae/inference/oracle.py -i {graph_file}",
                "Running Oracle inference"
            )
            results.append(("Oracle Inference", oracle_ok))
    else:
        results.append(("BottNet Training", False))
        results.append(("Oracle Inference", False))
    
    # ──────────────────────────────────────────────────────────
    # STAGE 3: Inverse Problem (Diffusion Dreamer)
    # ──────────────────────────────────────────────────────────
    print("\n\n" + "━" * 60)
    print("  STAGE 3: INVERSE PROBLEM (Diffusion Dreamer)")
    print("━" * 60)
    
    # 3a: Train diffusion (short run)
    diffusion_train_ok = run(
        f"python -m architectures.v2_diffusion.train -f {harvest_file} -e 10 -b 8",
        "Training diffusion model (10 epochs, batch=8)"
    )
    results.append(("Diffusion Training", diffusion_train_ok))
    
    if diffusion_train_ok:
        check_file("checkpoints/v2_diffusion_model.pth", "Diffusion checkpoint")
        
        # 3b: Generate
        gen_ok = run(
            "python -m architectures.v2_diffusion.validate -m checkpoints/v2_diffusion_model.pth -s 5",
            "Generating 5 candidate universes"
        )
        results.append(("Universe Generation", gen_ok))
    else:
        results.append(("Universe Generation", False))
    
    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    print_summary(results)


def print_summary(results):
    print("\n\n" + DIVIDER)
    print("  PIPELINE TEST SUMMARY")
    print(DIVIDER)
    
    all_pass = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("  🎉 All stages passed! Pipeline is working end-to-end.")
    else:
        print("  ⚠️  Some stages failed. Check the output above for details.")
    
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()