# 🌌 String Theory ML

*"There are two ways to do great mathematics. The first is to be smarter than everybody else. The second way is to be stupider than everybody else — but persistent." — Raoul Bott*

An end-to-end machine learning pipeline for exploring the string theory landscape.

---

## What Is This?

This is an **educational and exploratory** project that bridges theoretical physics and deep learning. It's a working pipeline that lets anyone with a laptop mine the mathematical landscape of string theory, train neural networks on the geometry of our hidden dimensions, and explore what it would take to reverse-engineer the shape of our universe.

This repo walks you through three things:

1. **Mining the landscape** — Harvesting mathematically valid Calabi-Yau manifolds from the Kreuzer-Skarke database, filtered for Standard Model compatibility
2. **Teaching AI to read geometry** — Training a Graph Neural Network to predict the topological properties (intersection numbers) of a manifold from its structural blueprint
3. **Attempting the inverse problem** — Using a generative diffusion model to ask: given target physics, what geometry produces them?

---

## The Big Idea (60-Second Version)

String theory says the physics of our universe — particle masses, force strengths, everything — emerges from the geometry of 6 extra dimensions curled up at every point in space, combined with the configuration of energy fields threaded through them. These hidden dimensions take the shape of mathematical objects called **Calabi-Yau manifolds**. Finding the right geometry is the Step 1 of many to find the final form of our universe. 

There are 10^80 atoms in the universe. A **vacuum configuration** is a potential version of reality — a different combination of geometry and energy fields that produces a self-consistent set of physical laws. String theory has around 10^500 of them. Finding the geometry that matches our universe is a needle-in-a-multiversic-haystack problem of almost incomprehensible scale. This pipeline applies an imperfect filter in an attempt to identify potential candidates. It uses machine learning to:

- **Filter** the landscape for physically motivated candidates using topological constraints
- **Predict** topological invariants (intersection numbers) using GNNs instead of expensive analytical computation
- **Attempt generation** of new candidate geometries using diffusion models — and document exactly where that attempt hits a known frontier problem

---

## The Standard Model Filter

This pipeline applies a topological first-pass filter motivated by a key property of our universe: three generations of matter particles (electron/muon/tau, up/charm/top, etc.).

In the heterotic string framework, the number of matter generations is directly related to the Euler characteristic of the Calabi-Yau manifold:

```
|h¹·¹ - h²·¹| = 3   →   χ = ±6
```

This pipeline uses the Kreuzer-Skarke database, which is built for Type IIB string compactifications. In that context, the Hodge number filter is a physically motivated proxy — not a rigorous derivation. It may miss some valid candidates and include some false ones. But it eliminates the vast majority of geometrically incompatible manifolds and makes the search tractable.

**This filter is a first gate, not a complete physical constraint.** A manifold passing this filter has the right topological capacity for three generations of matter. Whether it produces the correct particle masses, force strengths, gauge group, and cosmological constant requires the full metric — which requires solving the Monge-Ampère equation — a fully nonlinear partial differential equation that becomes computationally ruinous when applied across hundreds of millions of manifolds.

---

## Prerequisites

- **Git**
- **Conda or Mamba** — Required for CYTools and its C++ algebraic geometry backends
- **Python 3.11** (3.12+ has compatibility issues with CYTools on Apple Silicon)

## Setup

```bash
# 1. Clone
git clone https://github.com/trevorscott/string_theory_ml.git
cd string_theory_ml

# 2. Create environment
conda create -n string_theory_ml python=3.11
conda activate string_theory_ml

# 3. Install everything via conda
conda install -c conda-forge cytools numpy scipy matplotlib tqdm
conda install nomkl
conda install pytorch -c pytorch
conda install -c pyg pyg

# 4. Install the repo itself
pip install -e .
```

> ⚠️ **macOS Note:** Do *not* use `KMP_DUPLICATE_LIB_OK=True` as a workaround for OpenMP segfaults. The `nomkl` install above is the correct fix — it prevents Intel MKL from silently bottlenecking your CPU threads during training.

---

## Documentation

If you want to go on a self-guided adventure, check out the docs:

- [DOCUMENTATION.md](DOCUMENTATION.md) — Unix-style manual pages for all pipeline scripts

## Walkthrough

The pipeline has three stages. Each builds on the output of the previous one.

### Stage 1: Mine the Landscape

The harvester downloads polytope scaffolds from the Kreuzer-Skarke database, generates random triangulations via CYTools' C++ backend, and filters for Calabi-Yau manifolds that satisfy the Standard Model constraint (`|h¹·¹ - h²·¹| = 3`). For each valid manifold, it extracts the simplicial complex (the discrete structural skeleton) and the intersection numbers.

```bash
# Harvest 50 Standard Model candidate manifolds (quick test)
python src/harvesting/DeepSpaceHarvester.py -u 50

# Target a specific Hodge number range (recommended for focused datasets)
python src/harvesting/DeepSpaceHarvester.py -u 5000 --h11_min 17 --h11_max 53

# Full harvest across the KS database
python src/harvesting/DeepSpaceHarvester.py -u 5000
```

**Output:** `data/standard_model_N.pt` — a list of dictionaries, each containing:
- `X_simplices` — the simplicial complex (list of vertex tuples forming the structural skeleton)
- `Y_physics` — the intersection numbers (the prediction target)
- `h11`, `h21`, `euler` — the Hodge numbers and Euler characteristic

> **Note:** Data files are not committed to the repo (they exceed GitHub's size limits). You must run the harvester locally to generate them.

### Stage 2: Train the Oracle (Forward Problem)

*"Given a geometry, what are the physics?"*

This is the forward problem. We train a Graph Neural Network (BottNet) to predict the intersection numbers of a manifold directly from its simplicial geometry — bypassing the analytically expensive computation that would otherwise be required.

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

Pass a manifold to the trained Oracle and ask it to predict the intersection numbers:

```bash
python architectures/v1_cvae/inference/oracle.py -i data/smart_graph_standard_model_50.pt
```

This prints predicted vs. actual intersection numbers for a sample of manifolds in the dataset.

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

**Step 3b: Generate and validate candidates**

```bash
python -m architectures.v2_diffusion.validate -m checkpoints/v2_diffusion_model.pth -s 20
```

This generates candidate adjacency matrices and validates whether they correspond to real Calabi-Yau manifolds via CYTools.

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

A key finding from building this pipeline: continuous generative models cannot natively produce valid discrete topology.

Calabi-Yau adjacency matrices are binary — a structural connection either exists (1) or it doesn't (0). When a continuous neural network outputs a fractional value like 0.87, the mathematical wireframe can't snap together properly.

The v2 diffusion model generates structured 50×50 adjacency matrices with genuine internal patterns — not random noise. But when fed into CYTools for validation, none pass. The matrices have the appearance of structure without the underlying geometric validity. A connection that should be 0 is 0.3. A simplex that needs to close cleanly leaves a fractional gap.

Think of it like trying to build a Lego set out of Play-Doh. The model wants to output smooth, continuous values. The underlying structure demands discrete, integer-valued geometry.

This is an active area of research in geometric deep learning. The current diffusion implementation uses binary noise corruption as a step in the right direction, but state-of-the-art approaches like [DiGress](https://arxiv.org/abs/2209.14734) (discrete denoising diffusion for graphs) and [Cometh](https://openreview.net/forum?id=nuN1mRrrjX) (continuous-time discrete-state graph diffusion) represent the frontier for this class of problem. The v2 architecture documents the wall. The v3 architecture will attempt to climb it.

---

## What's Next

This repo is the educational foundation. A next-generation pipeline is in development that incorporates:

- **True discrete graph diffusion** — A DiGress/Cometh-style noise model that operates in binary space natively, with a marginal-preserving Markov chain
- **Graph transformer backbone** — Replacing message-passing GNNs with full O(n²) attention, enabling the model to capture global topological properties
- **Spectral and structural features** — Laplacian eigenvectors and random walk encodings injected at each denoising step
- **Classifier-free guidance** — Conditioning generation on target Hodge numbers without a separate classifier

Full roadmap is here: - [ROADMAP.md](ROADMAP.md) 

If you're interested in contributing to the next phase, open an issue or reach out. 

---

## Project Status

🟢 **Harvester** — Working. Scans the Kreuzer-Skarke database with Standard Model filtering. Supports targeted Hodge number ranges via `--h11_min` / `--h11_max`.  
🟢 **SmartGraphBuilder** — Working. Converts raw simplices to PyTorch Geometric graphs.  
🟢 **BottNet (Oracle)** — Working. Trains and runs inference on intersection number prediction.  
🟡 **Diffusion (Dreamer)** — Experimental. Trains and generates structured candidate matrices, but none pass CYTools validity checks. This is the continuous-discrete trap in action — and the motivation for the v3 architecture.  

---

## Limitations

This pipeline searches one corner of a much larger landscape:

- The Kreuzer-Skarke database covers one construction method for Calabi-Yau manifolds. Other families — Complete Intersection Calabi-Yaus (CICYs), free quotients, non-geometric constructions — are not represented. If our universe's geometry comes from one of those families, this pipeline won't find it.
- The Hodge number filter is a topological proxy, not a complete physical constraint. It has both false positives and false negatives.
- Even a manifold that passes every filter in this pipeline is nowhere near verified. Full verification requires the Ricci-flat metric — which requires solving the Monge-Ampère equation — plus flux compactification, moduli stabilization, and gauge group matching. None of that is in scope here.
- The total number of Calabi-Yau manifolds across all construction methods is unknown and not proven to be finite, though most mathematicians believe it is.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full script reference.

---

## Related Work & Further Reading

- [CYTools](https://cy.tools/) — The foundational Calabi-Yau analysis package this pipeline is built on
- [edhirst/P4CY3ML](https://github.com/edhirst/P4CY3ML) — ML on P4 Calabi-Yau threefolds
- [TomasSilva/LearningG2](https://github.com/TomasSilva/LearningG2) — Learning G2 manifolds
- [DiGress](https://arxiv.org/abs/2209.14734) — Discrete denoising diffusion for graph generation (Vignac et al., ICLR 2023)
- [Cometh](https://openreview.net/forum?id=nuN1mRrrjX) — Continuous-time discrete-state graph diffusion (Siraudin et al., TMLR 2025)
- [He et al.](https://arxiv.org/abs/2408.05076) — Distinguishing Calabi-Yau topology using machine learning (2024)
- [Erbin & Finotello](https://link.aps.org/doi/10.1103/PhysRevD.103.126014) — Machine learning for complete intersection Calabi-Yau manifolds

---

## About

Built by [Trevor Scott](https://github.com/trevorscott), inspired by the work of his grandfather [Raoul Bott](https://en.wikipedia.org/wiki/Raoul_Bott).

*Stupider than everyone else, but, persistent*
