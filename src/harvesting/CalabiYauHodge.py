import cytools
from cytools import fetch_polytopes
import random

def get_ks_shape():
    print("Fetching a random 4D shape with h21=7...")
    
    # 1. Get the generator
    # We use a limit to avoid a massive stream, and fetch a few to pick from
    g = fetch_polytopes(h21=7, lattice="N", limit=50)
    
    # 2. Turn the generator into a list so we can grab a random one
    poly_list = list(g)
    
    if not poly_list:
        print("No polytopes found.")
        return

  # 3. Pick a random one from our fetched list
    p = random.choice(poly_list)
    
    # 4. Extract data (Explicitly defining the lattice)
    vertices = p.vertices()
    h11 = p.h11(lattice="N")
    h21 = p.h21(lattice="N")
    euler = 2 * (h11 - h21)

    print("-" * 30)
    print(f"Vertices (Lattice N):\n{vertices}")
    print("-" * 30)
    print(f"Hodge Numbers: (h1,1: {h11}, h2,1: {h21})")
    print(f"Euler Characteristic: {euler}")

if __name__ == "__main__":
    get_ks_shape()