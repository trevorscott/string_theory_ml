import torch
from cytools import fetch_polytopes

print("💽 Locking onto Polytope #5...")
# Fetch the same list and grab the exact scaffold
polys = fetch_polytopes(h11=7, limit=50, as_list=True)
p = polys[5]

print(f"Hodge Numbers: ({p.h11(lattice='N')}, {p.h21(lattice='N')})")

print("\n🌌 Computing ALL possible universes for this scaffold...")
# We use all_triangulations() to get the absolute ground truth for this shape
t_list = p.all_triangulations(as_list=True)
print(f"Total Universes found: {len(t_list)}")

dataset = []

print("\n🧬 Extracting X (Simplices) and Y (Physics)...")
for i, t in enumerate(t_list):
    # X: The Structural DNA (convert simplices to a standard Python list of lists)
    simplices = [list(s) for s in t.simplices()]
    
    # Y: The Physics (Intersection Numbers dictionary)
    physics_dict = t.get_cy().intersection_numbers()
    
    dataset.append({
        "universe_id": i,
        "X_simplices": simplices,
        "Y_physics": physics_dict
    })

# Save the raw extracted dataset to your Mac
torch.save(dataset, "polytope5_multiverse.pt")

print("\n✅ Dataset saved to 'polytope5_multiverse.pt'!")
print(f"Example X shape (Universe 0): {len(dataset[0]['X_simplices'])} simplices")
print(f"Example Y keys (Universe 0): {len(dataset[0]['Y_physics'])} non-zero intersection couplings")