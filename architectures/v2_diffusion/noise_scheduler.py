import torch

class DiscreteNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Builds the Markov transition schedule for discrete graph diffusion.
        Instead of Gaussian noise (continuous), we dictate the probability 
        of a structural edge being overwritten by random noise.
        """
        self.num_timesteps = num_timesteps
        
        # The 'beta' schedule: The probability of corruption at each individual step.
        # Starts small (slow degradation) and gets larger near the end.
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Alpha is the probability that the edge STAYS in its current state
        self.alphas = 1.0 - self.betas
        
        # Cumulative product of alphas (alpha_bar)
        # This is the magic that allows us to jump straight to any timestep 't'
        # without having to calculate a loop of t-1, t-2, t-3...
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_adjacency_matrix, t):
        """
        Takes a perfect Calabi-Yau graph and corrupts it directly to timestep 't'.
        """
        # Ensure we keep everything on the same device (CPU, CUDA, or Apple's MPS)
        device = clean_adjacency_matrix.device
        batch_size = clean_adjacency_matrix.shape[0]
        
        # THE FIX: Push the cumulative alpha schedule to the correct hardware 
        # BEFORE we try to index it with our hardware-accelerated 't' tensor.
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Extract the cumulative alpha for the requested timestep(s)
        # Reshape for broadcasting across the [batch, nodes, nodes] matrix
        alpha_bar_t = self.alphas_cumprod[t].view(batch_size, 1, 1)
        
        # 1. Decide which edges KEEP their true Calabi-Yau structure.
        keep_mask = torch.rand_like(clean_adjacency_matrix) < alpha_bar_t
        
        # 2. For the edges that DON'T keep their structure, replace with pure noise
        random_noise = torch.randint(0, 2, size=clean_adjacency_matrix.shape, device=device).float()
        
        # 3. Combine them
        noisy_adjacency_matrix = torch.where(keep_mask, clean_adjacency_matrix, random_noise)
        
        return noisy_adjacency_matrix

# ==========================================
# TEST BLOCK: Watch the universe degrade
# ==========================================
if __name__ == "__main__":
    # Simulate a single microscopic Calabi-Yau adjacency matrix (10 nodes)
    # 1.0 means a topological edge exists. We start with a perfectly connected structure.
    # Moving this to MPS (Apple Silicon) if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    clean_universe = torch.ones((1, 10, 10)).to(device) 
    
    # Initialize the scheduler with 100 timesteps for a quick test
    scheduler = DiscreteNoiseScheduler(num_timesteps=100)
    
    print("--- FORWARD DIFFUSION TEST ---\n")
    print("Timestep 0: Original Perfect Graph (All 1s):\n", clean_universe[0].int())
    
    # Jump straight to timestep 30 (Starting to degrade)
    t_early = torch.tensor([30])
    noisy_graph_early = scheduler.add_noise(clean_universe, t_early)
    print("\nTimestep 30 (Starting to randomly flip bits):\n", noisy_graph_early[0].int())
    
    # Jump to timestep 70 (Heavy degradation)
    t_mid = torch.tensor([70])
    noisy_graph_mid = scheduler.add_noise(clean_universe, t_mid)
    print("\nTimestep 70 (Structure is mostly lost):\n", noisy_graph_mid[0].int())
    
    # Jump to timestep 99 (Completely destroyed)
    t_end = torch.tensor([99])
    destroyed_graph = scheduler.add_noise(clean_universe, t_end)
    print("\nTimestep 99 (Pure random noise - 50/50 split):\n", destroyed_graph[0].int())
