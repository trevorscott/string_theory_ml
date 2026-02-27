from cytools import fetch_polytopes

# Search for polytopes where 2 * (h11 - h21) = -6
# This usually implies h21 - h11 = 3
print("Searching for '3-Generation' candidates (Euler char = -6)...")

# We can't filter by Euler directly in the fetch, so we'll filter the results
g = fetch_polytopes(lattice="N", limit=500)

candidates = []
for p in g:
    h11 = p.h11(lattice="N")
    h21 = p.h21(lattice="N")
    if 2 * (h11 - h21) == -6:
        candidates.append(p)
        if len(candidates) >= 5: break

for i, p in enumerate(candidates):
    print(f"Candidate {i+1} Vertices:\n{p.vertices()}")
    print(f"Hodge: ({p.h11(lattice='N')}, {p.h21(lattice='N')}) | Euler: -6\n")