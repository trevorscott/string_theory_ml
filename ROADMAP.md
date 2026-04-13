# 🌌 String Theory ML — Research Roadmap

> *"The second way is to be stupider than everybody else — but persistent."* — Raoul Bott

---

## Current State (April 2026)

### ✅ Done

- End-to-end pipeline: harvest → process → train → infer
- `DeepSpaceHarvester.py`: production harvester with full Standard Model constraint (`|h11-h21|=3`), checkpoint/resume, CLI parameterization
- `DeepSpaceHarvester.py`: added `--h11_min`, `--h11_max`, `--h11_samples` params for targeted Hodge number range harvesting
- `SmartGraphBuilder.py`: topologically invariant graph construction
- `BottNet`: GNN forward problem (intersection number prediction from geometry)
- `v2_diffusion`: discrete graph diffusion architecture with proper iterative denoising loop (5 critical bugs fixed — see commit history)
- `standard_model_7283.pt`: first ML-ready Standard Model candidate dataset (7,283 manifolds across 4 Hodge pairs)
- Empirical documentation of the continuous-discrete trap — model generates structured 50×50 adjacency matrices with genuine internal patterns that fail CYTools validity checks. Not a bug. A documented frontier finding.
- Updated README with honest framing of the Standard Model filter — it is a topologically motivated first-pass proxy, not a rigorous physical constraint. Has both false positives and false negatives.
- `DOCUMENTATION.md`: full Unix-style MAN pages for all active scripts including new v2 diffusion train/validate pages
- Hodge scan across 815,432 manifolds — documented distribution across 208 unique Hodge pairs, identified rare low-h11 candidates (h11=13-16) as high-value validation set
- Identified physically motivated harvest range: h11=17-53 (37 Hodge pairs, all satisfying χ=±6)

### ⚠️ Known gaps

- No train/test split or held-out evaluation
- BottNet not validated against analytically known results
- No literature review completed
- Dataset has no formal data card
- Codebase has significant dead code (superseded scripts)
- The Hodge number filter is a heterotic-motivated proxy applied to a Type IIB database (KS). This conflation is documented in README and DOCUMENTATION.md but not yet resolved architecturally.
- Pipeline covers only the KS corner of the landscape. CICYs, free quotients, G2 manifolds (M-theory) are not represented.

---

## Phase 1 — Clean House

*Before anything else. Research built on messy foundations is hard to defend.*

- **Remove dead scripts** — delete `BigHarvester.py`, `GraphBuilder.py`, `CalabiYauHodge.py`, `sixEulerCharPolytopes.py`, `Triangle.py`, `CalculateUniverses.py`, `createTensors.py`, `CreateTensorDataset.py`, `GlobalTensorizer.py`
- **Keep as archive** — move `GroundTruth.py` to `tools/` — useful for validation
- **Rename dataset** — `deep_space_50k.pt` → `baseline_h11_10_unfiltered.pt` (honest naming)
- **Consolidate architectures** — ensure v2_diffusion is the canonical inverse model; archive CVAE clearly labeled as "documented failure"
- **Update README** — done, with honest filter framing and corrected technical claims
- **Add ROADMAP.md** — this file
- **Add DOCUMENTATION.md** — MAN-style pages for all active scripts

---

## Phase 2 — Scientific Foundation

*Make this defensible as actual research.*

### 2.1 Literature Review

- Yang-Hui He et al. — ML on Calabi-Yau manifolds (read + annotate)
- Halverson, Nelson, Ruehle — GNNs on string vacua
- Constantin, Lukas et al. — topological invariants via ML
- James Gray group — Hodge number prediction
- Erbin & Finotello — ML for complete intersection CY manifolds
- Document: what datasets exist, what constraints were used, what accuracy was achieved, what's novel about this work

### 2.2 Dataset Integrity

- Harvest targeted dataset: h11=17-53 range (37 Hodge pairs × up to 10k manifolds each)
- Harvest rare validation set: h11=13-16 (5,736 total manifolds — use as held-out "does this match our universe" validation after model training)
- Add χ sign filter to harvester (prefer χ=−6 for left-chiral matter)
- Investigate simply connected filter via CYTools `fundamental_group()`
- Add non-zero Yukawa coupling filter (intersection numbers already computed — use them)
- Write formal **data card** for `standard_model_7283.pt`: Hodge distribution, χ distribution, intersection number statistics, how it was generated
- Publish dataset to HuggingFace Datasets or Zenodo with DOI

### 2.3 Train/Test Split & Baselines

- Implement reproducible 80/10/10 split with fixed random seed — commit the split indices
- Baseline 1: random forest on Hodge numbers alone
- Baseline 2: linear regression on intersection number statistics
- Show BottNet beats baselines on held-out test set

---

## Phase 3 — BottNet Validation

*Prove the Oracle is learning geometry, not noise.*

- Identify 5–10 manifolds from published literature with known intersection numbers
- Run BottNet on these and compare predictions to ground truth (use `tools/GroundTruth.py`)
- Benchmark against Constantin et al. accuracy numbers if comparable setup exists
- Document: what BottNet gets right, what it gets wrong, and why

---

## Phase 4 — Dreamer V3 (Discrete Graph Diffusion)

*Clean rebuild. Not a modification of v2 — a fresh architecture.*

The v2 diffusion model is a documented demonstration of the continuous-discrete trap. V3 is a DiGress/Cometh-style rebuild that operates natively in discrete space. Five sequential Claude Code prompts:

- **Prompt 1** — Discrete DiGress-style noise model: Bernoulli edge-flip Markov chain with marginal-preserving stationary distribution and cosine schedule
- **Prompt 2** — Graph transformer denoiser: replace DenseDenoisingGNN with full O(n²) attention backbone, sinusoidal time embedding, edge update MLPs
- **Prompt 3** — Enriched data pipeline: Laplacian eigenvectors, random walk encodings, spectral node features injected at each denoising step
- **Prompt 4** — Classifier-free guidance: conditioning on target Hodge numbers without a separate classifier; 10-20% dropout during training, guidance scale w at generation
- **Prompt 5** — CYTools validation loop: reverse diffusion sampling, post-generation simplex extraction, CYTools validity check, BottNet oracle pass for intersection number prediction on valid candidates

Target metric: >0/50 valid CY manifolds on generation. Baseline is currently 0/50.

---

## Phase 5 — Metric Approximation (cymetric integration)

*Move from topology to geometry. The next real frontier.*

The geometric pipeline currently ends at intersection numbers (topology). The next step is approximating the Ricci-flat metric — the full geometry needed for physical predictions.

`cymetric` (Larfors, Schneider et al.) is an actively maintained Python package that uses ML to numerically approximate Calabi-Yau metrics by training a neural network to satisfy the Monge-Ampère equation. This is the most impactful single addition to the pipeline. It moves the project from topological characterization to geometric computation and makes the repo worth citing in the metric approximation literature.

- Install and validate `cymetric` against known simple manifolds (e.g. quintic threefold)
- Build Stage 4 pipeline script: takes validated manifold from Stage 1 harvester, outputs numerical metric approximation
- Benchmark metric approximation quality against Donaldson's algorithm on simple cases
- Identify computational limits: what h11 range is tractable on Apple Silicon?
- Enrich `standard_model_N.pt` with metric data for tractable candidates
- Document: what the metric tells us that intersection numbers don't

---

## Phase 6 — Flux Compactification

*Connect geometry to the 10^500 vacuum count.*

Each manifold can support an enormous number of distinct flux configurations — different ways to thread energy fields through its topological cycles. This is where most of the 10^500 comes from. CYTools has partial support for this.

- Compute flux superpotential W for candidate manifolds using CYTools
- Check tadpole cancellation condition: N_flux + N_branes = χ(CY)/24
- Enumerate a sample of valid flux configurations per candidate manifold
- Apply additional physical filters: is the flux configuration consistent with Standard Model gauge group?
- Document: how many valid flux configurations does a typical candidate support?
- Add flux configuration data to the dataset pipeline

---

## Phase 7 — Moduli Stabilization

*Lock the shape parameters in place.*

If moduli are unstabilized the physics drifts over time. We don't observe that. Moduli stabilization checks whether a candidate manifold plus flux configuration admits a stable vacuum.

- Compute Kähler potential for candidate manifolds using CYTools
- Check for stable AdS minimum (KKLT step 1) using CYTools stabilization tools
- Flag candidates that admit stable AdS minimum vs those that don't
- Document: what fraction of flux-filtered candidates survive moduli stabilization?
- Note: full de Sitter uplifting (KKLT step 2) is contested physics — document the Swampland controversy honestly, do not attempt to implement

---

## Phase 8 — Brane Configuration & Gauge Group Matching

*Find the candidates that could actually produce Standard Model physics.*

D-brane configurations on the manifold give rise to gauge symmetry. We need SU(3) × SU(2) × U(1). This requires sheaf cohomology computation.

- Compute divisor data and line bundle cohomology for candidate manifolds using CYTools and cohomCalg Python bindings
- Enumerate brane configurations that produce Standard Model gauge group
- Flag candidates that admit at least one valid brane configuration
- Document: what fraction of moduli-stabilized candidates have viable gauge structure?

---

## Phase 9 — Publication

*Turn this into a citable contribution.*

- Write paper draft (target: arXiv, ML4Physics workshop, or Journal of Computational Physics)
  - Section 1: The Standard Model constraint and honest filter framing
  - Section 2: Dataset construction methodology
  - Section 3: BottNet forward problem results
  - Section 4: Discrete diffusion and the continuous-discrete trap
  - Section 5: Metric approximation via cymetric integration
  - Section 6: Limitations, the KS corner problem, and future work
- Email Constantin / Halverson / He group — share repo, ask for feedback
- Add GitHub topics: `calabi-yau`, `string-theory`, `graph-neural-networks`, `geometric-deep-learning`
- Write companion article for Substack

---

## Potential Blockers

Two foundational theoretical problems could affect the validity of this entire research program:

**The de Sitter problem** — string theory's most rigorous mathematical machinery lives in Anti-de Sitter space. Our universe is de Sitter. Either a stable de Sitter vacuum must be constructed within string theory (contested — the Swampland conjecture argues this may be impossible), or a precise mathematical bridge between AdS and dS must be found. Without this, the landscape of 10^500 vacua may not contain our universe at all.

**The AdS/dS correspondence** — AdS/CFT gives us powerful tools for computing in Anti-de Sitter space. A de Sitter equivalent — dS/CFT — does not yet exist in any rigorous form. Building it, or proving the two are related by an analytic continuation of time, is one of the deepest open problems in theoretical physics.

If either of these is resolved, the engineering chain becomes clear. If the Swampland conjecture is proven correct, string theory cannot describe our universe and the search moves to the next framework. Either outcome is progress.

---

## The End Game (Beyond This Repo)

**Theoretical prerequisites (unsolved)**

This pipeline is the beginning of a much longer arc. The full chain from topology to verified physics:

1. **Topological filter (this repo)** — Hodge number constraint reduces the landscape to physically motivated candidates. Fast, imperfect, necessary.
2. **Metric approximation (Phase 5)** — ML-based solvers approximate the Ricci-flat metric for shortlisted candidates. Moves from topology to geometry.
3. **Flux + moduli + branes (Phases 6-8)** — Full vacuum specification. Each candidate manifold gets a flux configuration, stabilized moduli, and brane arrangement. Narrows candidates to those with viable Standard Model physics.
4. **Exact metric (quantum computing)** — Fault-tolerant quantum computers solve the Monge-Ampère equation exactly for the surviving candidates, computing definitive particle masses and coupling constants.
5. **Verification** — Predicted masses and constants matched against measured values. If a candidate matches, we have found the geometry of our universe.

If we're right about our universe's Calabi-Yau manifold, Step 5 produces exact predictions for the electron mass, quark masses, and gravitational constant from pure geometry. That's the Theory of Everything.

Two honest caveats: string theory may be wrong, and the de Sitter Swampland Conjecture may mean stable de Sitter vacua don't exist in string theory at all. Either outcome advances understanding. The search is worth doing regardless.

*Stupider than everybody else, but, persistent.*