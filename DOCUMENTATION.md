# String Theory ML — Script Reference

Unix-style manual pages for each active pipeline script. All scripts are fully parameterized via `argparse`. Run any script with `--help` for the full option list.

> **Note on deprecated scripts:** `UniverseGenerator.py`, `GenerateUniverse.py`, and `BigHarvester.py` are from the v1 CVAE architecture and are not documented here. They remain in the repo for reference but are superseded by the v2 diffusion architecture.

---

## Stage 1 — Harvesting

---

### DeepSpaceHarvester.py

```
NAME
       DeepSpaceHarvester.py — Calabi-Yau manifold harvester

SYNOPSIS
       python src/harvesting/DeepSpaceHarvester.py [OPTIONS]

DESCRIPTION
       Scans the Kreuzer-Skarke database for reflexive polytopes, generates
       random triangulations via CYTools' C++ backend, and filters for
       Calabi-Yau manifolds that satisfy the Standard Model constraint
       (|h11 - h21| = 3, corresponding to three generations of matter
       particles). For each valid manifold, extracts the simplicial complex
       (structural skeleton) and intersection numbers (prediction target).

       The output is saved as a PyTorch .pt file containing a list of
       dictionaries, one per manifold. Checkpoints periodically so long
       runs can be resumed.

       This filter is a topologically motivated first pass. It will miss
       some valid candidates and include some false ones. See README for
       full discussion of limitations.

OPTIONS
       -u, --universes <int>
              Target number of valid manifolds to harvest.
              (default: 10000)

       --h11_min <int>
              Minimum h11 value to scan in the KS database.
              (default: 1, the KS minimum)

       --h11_max <int>
              Maximum h11 value to scan in the KS database.
              (default: 491, the KS maximum)

       --h11_samples <int>
              Number of h11 values to randomly sample within the specified
              range. If not set, scans the full range exhaustively.
              Use this for broad exploratory runs across the landscape.
              (default: None — uses full range)

       --scaffolds_per_h11 <int>
              Maximum number of polytopes to fetch per h11 value.
              (default: 200)

       --triangulations_per_scaffold <int>
              Maximum number of triangulations to attempt per polytope.
              (default: 500)

       --checkpoint_interval <int>
              Save a checkpoint every N manifolds harvested.
              (default: 1000)

       --seed <int>
              Random seed for reproducible h11 sampling.
              (default: 42)

       --resume <file>
              Path to a checkpoint file to resume a previous run.
              (default: None)

OUTPUT
       data/standard_model_N.pt
              List of dicts, each containing:
                X_simplices  — list of vertex tuples (the structural skeleton)
                Y_physics    — intersection numbers (the prediction target)
                h11          — first Hodge number
                h21          — second Hodge number
                euler        — Euler characteristic

EXAMPLES
       # Quick test harvest
       python src/harvesting/DeepSpaceHarvester.py -u 50

       # Targeted run — physically motivated range, 370k manifolds
       python src/harvesting/DeepSpaceHarvester.py --h11_min 17 --h11_max 53 -u 370000

       # Rare Standard Model candidates (small h11)
       python src/harvesting/DeepSpaceHarvester.py --h11_min 13 --h11_max 16 -u 5000

       # Resume a previous run
       python src/harvesting/DeepSpaceHarvester.py -u 50000 --resume data/standard_model_checkpoint.pt
```

---

## Stage 2 — Processing

---

### SmartGraphBuilder.py

```
NAME
       SmartGraphBuilder.py — Topological graph dataset builder

SYNOPSIS
       python src/processing/SmartGraphBuilder.py -i <input_file> [OPTIONS]

DESCRIPTION
       Takes raw harvester output and translates each manifold's simplicial
       complex into a PyTorch Geometric Data object suitable for GNN training.

       Strips arbitrary vertex identifiers assigned by CYTools during
       triangulation and replaces them with topological node features
       (degree centrality, local clustering coefficient, simplex membership
       counts). This forces BottNet to learn pure structural relationships
       rather than memorizing arbitrary index patterns.

       The output file mirrors the input filename with a "smart_graph_"
       prefix unless overridden with -o.

OPTIONS
       -i, --input <file>
              (Required) Path to the raw harvested manifolds file.
              Example: data/standard_model_50.pt

       -o, --output <file>
              Path to save the processed graph dataset.
              (default: data/smart_graph_<input_filename>)

OUTPUT
       data/smart_graph_standard_model_N.pt
              PyTorch Geometric dataset. Each item contains:
                x           — node feature matrix [num_nodes, num_features]
                edge_index  — sparse adjacency in COO format [2, num_edges]
                y           — intersection numbers (prediction target)

EXAMPLES
       python src/processing/SmartGraphBuilder.py -i data/standard_model_50.pt
       python src/processing/SmartGraphBuilder.py -i data/standard_model_5000.pt -o data/graphs_5k.pt
```

---

## Stage 2 — Training & Inference (Forward Problem)

---

### TrainGraphModel.py

```
NAME
       TrainGraphModel.py — Train BottNet (the Oracle)

SYNOPSIS
       python architectures/v1_cvae/training/TrainGraphModel.py -i <input_file> [OPTIONS]

DESCRIPTION
       Trains BottNet, a Graph Neural Network, to predict intersection numbers
       directly from the structural graph of a Calabi-Yau manifold — bypassing
       the analytically expensive computation that CYTools would otherwise
       require.

       BottNet takes the SmartGraphBuilder output as input. It learns to map
       the graph structure (nodes, edges, topological features) to the
       intersection numbers that characterize the manifold's topology.

       Trained weights are saved to checkpoints/ and can be loaded by oracle.py
       for inference on new manifolds.

OPTIONS
       -i, --input <file>
              (Required) Path to the processed graph dataset.
              Example: data/smart_graph_standard_model_50.pt

       -o, --output <file>
              Path to save the trained model weights.
              (default: checkpoints/gnn_universe_model.pth)

       -e, --epochs <int>
              Number of training epochs.
              (default: 150)

       -b, --batch_size <int>
              Batch size for the dataloader.
              (default: 32)

OUTPUT
       checkpoints/gnn_universe_model.pth
              Trained BottNet weights.

EXAMPLES
       python architectures/v1_cvae/training/TrainGraphModel.py -i data/smart_graph_standard_model_50.pt
       python architectures/v1_cvae/training/TrainGraphModel.py -i data/smart_graph_standard_model_5000.pt -e 200 -b 64
```

---

### oracle.py

```
NAME
       oracle.py — Forward problem inference engine

SYNOPSIS
       python architectures/v1_cvae/inference/oracle.py -i <input_file> [OPTIONS]

DESCRIPTION
       Loads a trained BottNet model and runs inference on a graph dataset,
       printing predicted vs. actual intersection numbers for each manifold.

       The Oracle answers the forward problem: given a geometry, what are
       the intersection numbers? It does this via GNN inference rather than
       analytical computation.

OPTIONS
       -i, --input <file>
              (Required) Path to the graph dataset to run inference on.
              Example: data/smart_graph_standard_model_50.pt

       -m, --model <file>
              Path to the trained BottNet weights.
              (default: checkpoints/gnn_universe_model.pth)

OUTPUT
       Prints predicted vs. actual intersection numbers to stdout for a
       sample of manifolds in the dataset.

EXAMPLES
       python architectures/v1_cvae/inference/oracle.py -i data/smart_graph_standard_model_50.pt
       python architectures/v1_cvae/inference/oracle.py -i data/smart_graph_standard_model_50.pt -m checkpoints/gnn_universe_model.pth
```

---

## Stage 3 — Training & Inference (Inverse Problem)

---

### architectures/v2_diffusion/train.py

```
NAME
       train.py — Train the v2 discrete graph diffusion model (The Dreamer)

SYNOPSIS
       python -m architectures.v2_diffusion.train -f <data_file> [OPTIONS]

DESCRIPTION
       Trains a discrete graph diffusion model on harvested Calabi-Yau
       adjacency matrices. The model learns to denoise binary adjacency
       matrices corrupted with discrete Bernoulli noise, with the goal
       of learning the distribution of valid Calabi-Yau graph structures.

       Takes raw harvester output directly — not SmartGraphBuilder output.
       Internally converts simplicial complexes to padded binary adjacency
       matrices of size num_nodes x num_nodes.

       The pos_weight for the BCE loss is calculated empirically from the
       training data at initialization, accounting for the sparsity of
       real Calabi-Yau adjacency matrices (~6% edge density).

       Training can be interrupted with Ctrl+C — weights are saved on
       interrupt.

       NOTE: The v2 model generates structured adjacency matrices that do
       not pass CYTools validity checks. This is the continuous-discrete
       trap — see README for full discussion. The v3 architecture will
       address this with a native discrete noise model.

OPTIONS
       -f, --file <file>
              (Required) Path to the raw harvested .pt data file.
              Example: data/standard_model_1000.pt

       -e, --epochs <int>
              Number of training epochs.
              (default: 50)

       -b, --batch_size <int>
              Batch size for training.
              (default: 32)

       -n, --nodes <int>
              Matrix padding size — all adjacency matrices are padded to
              this size. Must match the value used in validate.py.
              (default: 50)

       --hidden_dim <int>
              Hidden dimension size for the DenseDenoisingGNN.
              Must match the value used in validate.py.
              (default: 256)

OUTPUT
       checkpoints/v2_diffusion_model.pth
              Trained model weights.

EXAMPLES
       python -m architectures.v2_diffusion.train -f data/standard_model_50.pt
       python -m architectures.v2_diffusion.train -f data/standard_model_5000.pt -e 100 -b 64 --hidden_dim 256
```

---

### architectures/v2_diffusion/validate.py

```
NAME
       validate.py — Generate and validate candidate manifolds (The Dreamer inference)

SYNOPSIS
       python -m architectures.v2_diffusion.validate -m <model_path> [OPTIONS]

DESCRIPTION
       Loads a trained v2 diffusion model and runs the reverse diffusion
       process to generate candidate Calabi-Yau adjacency matrices from
       noise. Each generated matrix is validated against CYTools to check
       whether it corresponds to a valid Calabi-Yau manifold.

       The denoising loop starts from random binary noise and iteratively
       refines it toward a clean adjacency matrix over num_timesteps steps.
       At each intermediate step, the predicted clean graph is re-noised to
       t-1 so the process genuinely iterates rather than collapsing to a
       single forward pass.

       Outputs a summary table with edge count, density, connectivity, and
       CYTools validation result for each generated candidate. Prints a
       final neighborhood report with aggregate statistics.

       With --verbose, also renders each generated adjacency matrix in the
       terminal as a grid of 1s and 0s.

OPTIONS
       -m, --model_path <file>
              (Required) Path to the trained diffusion model weights.
              Example: checkpoints/v2_diffusion_model.pth

       -s, --samples <int>
              Number of candidate matrices to generate.
              (default: 20)

       -n, --nodes <int>
              Node padding size — must match the value used during training.
              (default: 50)

       --hidden_dim <int>
              Hidden dimension size — must match the value used during training.
              (default: 256)

       -v, --verbose
              Print each generated adjacency matrix to the terminal as a
              grid of 1s and 0s, cropped to active nodes.
              (default: False)

OUTPUT
       Stdout table:
              ID | Edges | Density | Connected? | Physics Result
              One row per generated candidate. Physics Result is one of:
                ❌ Failed      — did not pass CYTools validation
                ✅ Topo        — valid topology (Euler characteristic reported)
                🔥 CY          — valid Calabi-Yau (h11 reported)

       Final neighborhood report:
              Avg edges per generated matrix
              Connectivity rate
              Valid topology count
              Valid Calabi-Yau count

EXAMPLES
       python -m architectures.v2_diffusion.validate -m checkpoints/v2_diffusion_model.pth
       python -m architectures.v2_diffusion.validate -m checkpoints/v2_diffusion_model.pth -s 50 --hidden_dim 256 --verbose
```

---

*See [README.md](README.md) for full pipeline context, setup instructions, and project status.*
