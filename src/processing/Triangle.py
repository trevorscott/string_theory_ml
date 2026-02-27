from cytools import fetch_polytopes

print("🔭 Asking cytools to find favorable 3-generation candidates...")

candidates = []

# neighborhood search for h11=1 (simplest) or h11=2 (shatter-able)
for h_val in [1, 2]:
    print(f"Checking h11={h_val} neighborhood...")
    # 'lattice' is required for h11/h21 searches in fetch_polytopes
    res = fetch_polytopes(h11=h_val, lattice="N", favorable=True, limit=50, as_list=True)
    
    for p in res:
        # Use .chi() instead of .euler()
        if abs(p.chi(lattice="N")) == 6:
            candidates.append(p)
    
    if candidates: break

if candidates:
    p = candidates[0]
    print(f"\n✅ Target Found!")
    print(f"Hodge Numbers: ({p.h11(lattice='N')}, {p.h21(lattice='N')})")
    print(f"Euler Characteristic (Chi): {p.chi(lattice='N')}")
    
    # SHATTER: Generate triangulations
    print("\n🌌 Generating 100 'Universes' (Triangulations)...")
    t_gen = p.random_triangulations_fast(N=100)
    
    # For the first triangulation, let's peek at the 'DNA'
    t1 = next(t_gen)
    v1 = t1.get_toric_variety()
    ints = v1.intersection_numbers()
    
    print(f"Sample Intersection Data from Universe #1: {list(ints.items())[:3]}")
else:
    print("\n❌ No candidates found. Use the manual matrix fallback.")