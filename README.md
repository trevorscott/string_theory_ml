# 🌌 String Theory ML

*"There are two ways to do great mathematics. The first is to be smarter than everybody else. The second way is to be stupider than everybody else — but persistent." — Raoul Bott*

An end-to-end machine learning pipeline for exploring the string theory landscape — built for engineers and curious minds, not just physicists.

---

## What Is This?

This is an **educational and exploratory** project that bridges theoretical physics and deep learning. It is not a research paper. It's a working pipeline that lets any engineer with a laptop mine the mathematical landscape of string theory, train neural networks on the geometry of hidden dimensions, and explore what it would take to reverse-engineer the shape of our universe.

If you want the full story — why string theory needs ML, what Calabi-Yau manifolds are, and how this connects to the search for a Theory of Everything — read the companion article: **[The First Step to Solving the Universe: AI, Topology, and String Theory](https://medium.com/@marquan03)**.

This repo walks you through three things:

1. **Mining the multiverse** — Harvesting thousands of mathematically valid Calabi-Yau manifolds from the Kreuzer-Skarke database, filtered for Standard Model compatibility
2. **Teaching AI to read geometry** — Training a Graph Neural Network to predict the physical laws (intersection numbers) of a universe from its structural blueprint
3. **Dreaming new universes** — Using a generative diffusion model to attempt the inverse problem: given target physics, generate geometry

---

## The Big Idea (60-Second Version)

String theory says the physics of our universe — particle masses, force strengths, everything — is determined by the shape of 6 extra dimensions curled up at every point in space. These shapes are called **Calabi-Yau manifolds**.

There are an estimated 10^500 possible shapes. Finding the one that matches our universe is a needle-in-a-haystack problem. This pipeline uses modern tools & machine learning to:

- **Filter** the landscape for physically motivated candidates
- **Predict** topological properties (the "DNA" of a shape) using GNNs instead of impossible analytical math
- **Generate** new candidate geometries using diffusion models

---

## The Standard Model Filter

Unlike most prior work that samples the landscape arbitrarily, this pipeline filters for **physically motivated candidates**. Our universe has 3 generations of matter particles (electron/muon/tau, up/charm/top, etc.), which in string theory corresponds to an Euler characteristic of ±6:

```
|h¹·¹ - h²·¹| = 3   →   χ = ±6
```

Scanning the Kreuzer-Skarke database for this constraint yields a small, focused dataset of Standard Model candidate universes — the geometries that could plausibly describe the hidden dimensions of *our* universe.

---

## Prerequisites

- **Git**
- **Conda or Mamba** — Required for CYTools and its C++ algebraic geometry backends
- **Python 3.11** (3.12+ has compatibility issues with CYTools on Apple Silicon)
- **uv** — For remaining Python dependency management

## Setup

```bash
# 1. Clone
git clone https://github.com/trevorscott/string_theory_ml.git
cd string_theory_ml

# 2. Create environment
conda create -n string_theory_ml python=3.11
conda activate string_theory_ml

# 3. Install CYTools (requires Conda — this pulls in C++ backends)
conda install -c conda-forge cytools

# 4. Prevent OpenMP/MKL thread conflicts (important on macOS)
conda install nomkl
conda install pytorch -c pytorch

# 5. Install remaining Python dependencies
uv sync
```

> ⚠️ **macOS Note:** Do *not* use `KMP_DUPLICATE_LIB_OK=True` as a workaround for OpenMP segfaults. The `nomkl` install above is the correct fix — it prevents Intel MKL from silently bottlenecking your CPU threads during training.

---

## Walkthrough

The pipeline has three stages. Each builds on the output of the previous one.

### Stage 1: Mine the Multiverse

The harvester downloads polytope scaffolds from the Kreuzer-Skarke database, generates random triangulations via CYTools' C++ backend, and filters for Calabi-Yau manifolds that satisfy the Standard Model constraint (`|h¹·¹ - h²·¹| = 3`). For each valid manifold, it extracts the simplicial complex (the discrete structural skeleton) and the intersection numbers (the physics).

```bash
# Harvest 50 Standard Model candidate universes (quick test)
python src/harvesting/DeepSpaceHarvester.py -u 50

# Full harvest — this takes a while
python src/harvesting/DeepSpaceHarvester.py -u 5000
```

**Output:** `data/standard_model_N.pt` — a list of dictionaries, each containing:
- `X_simplices` — the simplicial complex (list of vertex tuples forming the structural skeleton)
- `Y_physics` — the intersection numbers (the target physical properties)
- `h11`, `h21`, `euler` — the Hodge numbers and Euler characteristic

> **Note:** Data files are not committed to the repo (they exceed GitHub's size limits). You must run the harvester locally to generate them.

### Stage 2: Train the Oracle (Forward Problem)

*"Given a geometry, what are the physics?"*

This is the forward problem. We train a Graph Neural Network (BottNet) to predict the intersection numbers of a manifold directly from its simplicial geometry — bypassing the analytically impossible math that would otherwise be required.

**Step 2a: Convert raw geometry into graphs**

The `SmartGraphBuilder` takes the raw simplicial complexes from the harvester and translates them into PyTorch Geometric graph objects. It strips arbitrary vertex identifiers and computes topological node features, forcing the model to learn pure spatial relationships rather than memorized indices.

```bash
python src/processing/SmartGraphBuilder.py -i data/standard_model_50.pt
```

**Output:** `data/smart_graph_standard_model_50.pt`

**Step 2b: Train BottNet**

```bash
python architectures/v1_cvae/training/TrainGraphModel.py -i data/smart_graph_standard_model_50.pt -e 150 -b 32
```

**Output:** `checkpoints/gnn_universe_model.pth`

**Step 2c: Run inference**

Pass an unknown manifold to the trained Oracle and ask it to predict the physics:

```bash
python architectures/v1_cvae/inference/oracle.py -i data/smart_graph_standard_model_50.pt
```

This prints predicted vs. actual intersection numbers for a sample of universes in the dataset.

### Stage 3: The Dreamer (Inverse Problem)

*"Given target physics, what is the geometry?"*

This is the hard problem — and the frontier. Instead of asking "what physics does this shape produce?", we ask "what shape produces *these* physics?"

The Dreamer is a diffusion model that learns the distribution of valid Calabi-Yau adjacency matrices and attempts to generate new ones from noise.

> ⚠️ **Important:** The diffusion model takes **raw harvester output** (`standard_model_*.pt`), *not* the SmartGraphBuilder output. It builds its own adjacency matrix representation internally.

**Step 3a: Train the diffusion model**

```bash
python -m architectures.v2_diffusion.train -f data/standard_model_50.pt -e 50 -b 32
```

**Output:** `checkpoints/v2_diffusion_model.pth`

**Step 3b: Generate universes**

```bash
python -m architectures.v2_diffusion.validate -m checkpoints/v2_diffusion_model.pth -s 20
```

This generates candidate adjacency matrices and (if CYTools is available) validates whether they correspond to real Calabi-Yau manifolds.

---

## Pipeline Architecture

The pipeline branches after harvesting — don't mix inputs between the two paths:

```
src/harvesting/DeepSpaceHarvester.py
              │
              ▼
     standard_model_N.pt  (raw harvested manifolds)
              │
    ┌─────────┴─────────────────────────────────┐
    │                                            │
    ▼                                            ▼
src/processing/                     architectures/v2_diffusion/
SmartGraphBuilder.py                train.py
    │                               (builds adjacency matrices
    ▼                                from raw simplices internally)
smart_graph_standard_model_N.pt              │
    │                                        ▼
    ├──→ architectures/v1_cvae/     architectures/v2_diffusion/
    │    training/TrainGraphModel.py validate.py
    │         │                     (generate + validate)
    │         ▼
    └──→ architectures/v1_cvae/
         inference/oracle.py
```

---

## The Continuous vs. Discrete Trap

A key finding from building this pipeline: continuous generative models (VAEs, vanilla Gaussian diffusion) **cannot natively produce valid discrete topology**.

Calabi-Yau adjacency matrices are binary — a structural connection either exists (1) or it doesn't (0). When a continuous neural network outputs a fractional value like 0.87, the mathematical wireframe can't snap together. The simulated universes collapse.

Think of it like trying to build a Lego set out of Play-Doh. The AI wants to output smooth, continuous values, but the underlying structure demands discrete, integer-valued geometry.

This is an active area of research in geometric deep learning. The current diffusion implementation here uses binary noise corruption as a step in the right direction, but state-of-the-art approaches like [DiGress](https://arxiv.org/abs/2209.14734) (discrete denoising diffusion for graphs) and [Cometh](https://openreview.net/forum?id=nuN1mRrrjX) (continuous-time discrete-state graph diffusion) represent the frontier for this class of problem.

---

## What's Next

This repo is the educational foundation. A next-generation pipeline is in development that incorporates:

- **True discrete graph diffusion** — A DiGress/Cometh-style noise model that operates in binary space natively, with a marginal-preserving Markov chain
- **Graph transformer backbone** — Replacing message-passing GNNs with full O(n²) attention, enabling the model to capture global topological properties like "holes" in the manifold
- **Spectral and structural features** — Laplacian eigenvectors and random walk encodings injected at each denoising step
- **Classifier-free guidance** — Conditioning generation on target Hodge numbers without a separate classifier

If you're interested in contributing to the next phase, open an issue or reach out.

---

## Project Status

🟢 **Harvester** — Working. Scans the Kreuzer-Skarke database with Standard Model filtering.
🟢 **SmartGraphBuilder** — Working. Converts raw simplices to PyTorch Geometric graphs.
🟢 **BottNet (Oracle)** — Working. Trains and runs inference on topological prediction.
🟡 **Diffusion (Dreamer)** — Experimental. Trains and generates samples, but generated manifolds do not yet pass CYTools validity checks consistently. This is the continuous-discrete trap in action — and the motivation for the v3 architecture.

---

## Related Work & Further Reading

- [CYTools](https://cy.tools/) — The foundational Calabi-Yau analysis package this pipeline is built on
- [edhirst/P4CY3ML](https://github.com/edhirst/P4CY3ML) — ML on P4 Calabi-Yau threefolds
- [TomasSilva/LearningG2](https://github.com/TomasSilva/LearningG2) — Learning G2 manifolds
- [DiGress](https://arxiv.org/abs/2209.14734) — Discrete denoising diffusion for graph generation (Vignac et al., ICLR 2023)
- [Cometh](https://openreview.net/forum?id=nuN1mRrrjX) — Continuous-time discrete-state graph diffusion (Siraudin et al., TMLR 2025)
- [He et al.](https://arxiv.org/abs/2408.05076) — Distinguishing Calabi-Yau topology using machine learning (2024)
- [Erbin & Finotello](https://link.aps.org/doi/10.1103/PhysRevD.103.126014) — Machine learning for complete intersection Calabi-Yau manifolds
- [The First Step to Solving the Universe](https://medium.com/@marquan03) — Companion article by Trevor Scott

---

## About

Built by [Trevor Scott](https://github.com/trevorscott), inspired by the work of [Raoul Bott](https://en.wikipedia.org/wiki/Raoul_Bott).

*Be persistent.*
