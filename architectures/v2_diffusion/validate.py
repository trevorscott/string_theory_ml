import torch
import networkx as nx
import argparse
import numpy as np
from architectures.v2_diffusion.model import DenseDenoisingGNN
from architectures.v2_diffusion.noise_scheduler import DiscreteNoiseScheduler

try:
    import cytools
except ImportError:
    cytools = None


def print_matrix(adj_matrix, title="Manifold Structure"):
    """
    Renders the binary matrix in the terminal using 1s and 0s.
    """
    active_indices = np.where(adj_matrix.sum(axis=1) > 0)[0]
    if len(active_indices) == 0:
        print(f"\n[ {title}: EMPTY ]")
        return
        
    start, end = active_indices[0], active_indices[-1] + 1
    cropped = adj_matrix[start:end, start:end]
    
    print(f"\n--- {title} (Showing {len(cropped)}x{len(cropped)} active nodes) ---")
    for row in cropped:
        line = "".join(["1 " if val > 0.5 else "0 " for val in row])
        print(line)
    print("-" * (len(cropped) * 2))


def get_graph_metrics(adj_matrix):
    """
    Calculates 'Neighborhood' metrics to see if the generated 
    matrix is in the right physical ballpark.
    """
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    num_nodes = adj_matrix.shape[0]
    edges = G.number_of_edges()
    
    density = edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
    is_connected = nx.is_connected(G) if edges > 0 else False
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if len(degrees) > 0 else 0
    
    return {
        "density": density,
        "is_connected": is_connected,
        "avg_degree": avg_degree,
        "num_edges": edges
    }


def generate_universes(model, scheduler, num_samples, num_nodes, device):
    model.eval()
    print(f"🌌 Denoising {num_samples} universes...")
    
    with torch.no_grad():
        # Start with pure 50/50 random binary noise
        graphs = torch.randint(0, 2, (num_samples, num_nodes, num_nodes)).float().to(device)
        
        # FIX 1: Enforce symmetry on the initial noise using max instead of
        # averaging. Averaging + thresholding at 0.5 was killing sparse edges
        # that were only predicted in one direction. Max preserves them.
        graphs = torch.max(graphs, graphs.transpose(1, 2))
        
        for t_val in reversed(range(scheduler.num_timesteps)):
            t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            predicted_clean_logits = model(graphs, t)
            
            # FIX 2: Only apply hard threshold at the FINAL step.
            # Previously 0.75 threshold was applied at every step, making
            # hard binary kills on uncertain edges that could never recover.
            # During intermediate steps, keep soft probabilities so the
            # iterative process has signal to work with.
            predicted_clean_probs = torch.sigmoid(predicted_clean_logits)
            
            if t_val > 0:
                # FIX 3: THE CRITICAL FIX - actually use the iterative structure.
                # Previously the loop was replacing graphs with the clean prediction
                # directly at every step — making 100 iterations identical to 1.
                # Now we re-noise the prediction to timestep t-1, so each step
                # genuinely refines the previous step's output. This is the entire
                # mechanism by which diffusion models work.
                predicted_clean_hard = (predicted_clean_probs > 0.5).float()
                # Symmetrize before re-noising
                predicted_clean_hard = torch.max(
                    predicted_clean_hard, 
                    predicted_clean_hard.transpose(1, 2)
                )
                t_prev = torch.full((num_samples,), t_val - 1, device=device, dtype=torch.long)
                graphs = scheduler.add_noise(predicted_clean_hard, t_prev)
            else:
                # Final step: apply hard threshold to get clean binary output
                graphs = (predicted_clean_probs > 0.5).float()
                # FIX 4: Final symmetry enforcement using max not average
                graphs = torch.max(graphs, graphs.transpose(1, 2))
        
    return graphs.cpu().numpy()


def matrix_to_simplices(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    cliques = list(nx.find_cliques(G))
    return [c for c in cliques if len(c) >= 3]


def validate_physics(simplices):
    if not cytools: return {"valid": False, "error": "cytools missing"}
    if len(simplices) < 4: return {"valid": False, "error": "Insufficient complexity"}

    try:
        complex_shape = cytools.SimplicialComplex(simplices)
        euler = complex_shape.euler_characteristic()
        
        try:
            h11 = complex_shape.h11()
            h21 = complex_shape.h21()
            is_cy = True
        except:
            h11, h21, is_cy = None, None, False

        return {"valid": True, "euler": euler, "is_cy": is_cy, "h11": h11, "h21": h21}
    except Exception as e:
        return {"valid": False, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help="Path to trained weights")
    parser.add_argument('-s', '--samples', type=int, default=20, help="Number of universes to generate")
    parser.add_argument('-n', '--nodes', type=int, default=50, help="Node padding size")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Hidden dimension size — must match training value (default: 256)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print out all generated matrices")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    scheduler = DiscreteNoiseScheduler(num_timesteps=100)

    # FIX 5: hidden_dim is now a CLI argument so it always matches the
    # trained model. Previously it was hardcoded to 128 in validate.py
    # while train.py defaulted to 256 — a silent shape mismatch that
    # would cause a crash or load incorrect weights.
    model = DenseDenoisingGNN(num_nodes=args.nodes, hidden_dim=args.hidden_dim).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    generated_matrices = generate_universes(model, scheduler, args.samples, args.nodes, device)
    
    stats = {"valid_topo": 0, "cy": 0, "total_edges": 0, "connected_count": 0}
    
    print(f"\n{'ID':<4} | {'Edges':<6} | {'Density':<8} | {'Conn?':<6} | {'Physics Result'}")
    print("-" * 65)

    for i, matrix in enumerate(generated_matrices):
        metrics = get_graph_metrics(matrix)
        simplices = matrix_to_simplices(matrix)
        phys = validate_physics(simplices)

        if args.verbose:
            print_matrix(matrix, title=f"Universe #{i} Topology")
        
        stats["total_edges"] += metrics["num_edges"]
        if metrics["is_connected"]: stats["connected_count"] += 1
        
        status = "❌ Failed"
        if phys["valid"]:
            stats["valid_topo"] += 1
            status = f"✅ Topo (Euler:{phys['euler']})"
            if phys["is_cy"]:
                stats["cy"] += 1
                status = f"🔥 CY (h11:{phys['h11']})"

        print(f"{i:<4} | {metrics['num_edges']:<6} | {metrics['density']:<8.2%} | {str(metrics['is_connected']):<6} | {status}")

    print("\n" + "="*40)
    print("📊 FINAL NEIGHBORHOOD REPORT")
    print("-" * 40)
    print(f"Avg Edges per Gen:    {stats['total_edges']/args.samples:.1f}")
    print(f"Connectivity Rate:    {stats['connected_count']/args.samples:.1%}")
    print(f"Valid Topologies:     {stats['valid_topo']}/{args.samples}")
    print(f"Valid Calabi-Yaus:    {stats['cy']}/{args.samples}")
    print("="*40)