from cytools import fetch_polytopes

print("🧮 Waking up the TOPCOM C++ backend for Polytope #205...")
polys = fetch_polytopes(h11=7, limit=210, as_list=True)
p_real = polys[205]

# Grab a triangulation
t_real = next(p_real.random_triangulations_fast(N=1))

# Let cytools do the heavy algebraic geometry
print("⏳ Calculating true topological intersections (this might take a second)...")
true_physics = t_real.get_cy().intersection_numbers()

print("\n--- GROUND TRUTH (REAL PHYSICS) ---")
# Sort the dictionary by absolute value to find the biggest physical couplings
sorted_couplings = sorted(true_physics.items(), key=lambda item: abs(item[1]), reverse=True)

print("Top 5 Actual Intersection Numbers:")
for key, val in sorted_couplings[:5]:
    print(f"  Coupling {key}: True Value = {val}")