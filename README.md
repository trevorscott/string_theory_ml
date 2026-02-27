# 🌌 String Theory ML: Mapping the Calabi-Yau Landscape

*“There are two ways to do great mathematics. The first is to be smarter than everybody else. The second way is to be stupider than everybody else — but persistent.” — Raoul Bott*

## Overview
This repository is a local, end-to-end Machine Learning pipeline designed to explore the structural DNA of the universe. 

In string theory, the physics of our 4D reality—from the mass of an electron to the strength of gravity—is dictated by the geometry of 6 extra, invisible dimensions curled up into Calabi-Yau manifolds. With an estimated $10^{500}$ possible shapes, finding the one that matches our universe is fundamentally a Big Data problem.

This project bridges theoretical physics and deep learning. It automates the harvesting of multi-dimensional geometric scaffolds, translates their topology into Graph Neural Networks (GNNs), and deploys Generative AI to reverse-engineer universes based on targeted physical laws.

## Pipeline Architecture
The codebase is structured into a four-stage experimental pipeline:

### 1. ⛏️ Harvesting (`src/harvesting/`)
* **Objective:** Mine the string theory landscape for mathematically stable universes.
* **Mechanism:** Uses combinatorial C++ backends to shatter complex $h^{1,1}$ scaffolds, extracting the precise matrices of simplices that form the structural boundaries of Calabi-Yau spaces. 
* **Key Scripts:** `DeepSpaceHarvester.py`, `BigHarvester.py`

### 2. 🧬 Processing (`src/processing/`)
* **Objective:** Translate discrete geometry into machine-readable mathematical formats.
* **Mechanism:** Strips arbitrary vertex identifiers and maps the continuous manifolds into 3D Graph structures and Tensor datasets, ensuring the AI learns pure spatial relationships rather than memorizing noise.
* **Key Scripts:** `SmartGraphBuilder.py`, `GlobalTensorizer.py`

### 3. 🧠 Training (`src/models/` & `src/training/`)
* **Objective:** Teach AI the inductive bias of string theory geometry.
* **Mechanism:** Trains Graph Neural Networks (GNNs) to organically deduce topological intersection numbers (the capacity for physical laws) directly from structural blueprints, bypassing impossible analytical calculations.
* **Key Scripts:** `BottNet.py`, `TrainGraphModel.py`, `UniverseGenerator.py`

### 4. ⚡ Inference & God Mode (`src/inference/`)
* **Objective:** Solve the "Inverse Problem" of string theory.
* **Mechanism:** Deploys a Conditional Variational Autoencoder (CVAE). Instead of blindly searching $10^{500}$ shapes, this generative engine takes target physical laws as an input, samples a continuous probability cloud, and hallucinates the rigid geometric matrix of a novel universe.
* **Key Scripts:** `GenerateUniverse.py`, `oracle.py`

## The Math and the Mission
To find the master equation of reality, we must separate **Topology** (the squishy, invariant capacity for physics) from the **Metric** (the continuous, rigid geometry required to calculate exact particle masses). Finding the metric requires solving the non-linear Monge-Ampère equation—a task impossible for modern supercomputers to do blindly.

This pipeline acts as the ultimate filter. By using AI to master the topology and throw away the wrong universes, we are building the target list for the fault-tolerant quantum computers of the future. 

## Setup & Execution

*Note: Due to GitHub file size limits, the `data/` (.pt) and `checkpoints/` (.pth) directories are ignored. You must run the Harvesters locally to generate the raw universe datasets.*

**1. Initialize the Environment**
This project requires `cytools` (best installed via conda/mamba) and PyTorch. If you are using `uv` for dependency management (as configured in the `pyproject.toml` and `uv.lock`), sync your environment:

```bash
conda activate cytools
uv sync
```
**2. Harvest the Multiverse**
Mine the $h^{1,1}=10$ landscape for stable Calabi-Yau manifolds.
(Note: Mac users might need use the KMP_DUPLICATE_LIB_OK=True override to prevent OpenMP library conflicts during heavy combinatorial math).

```bash
KMP_DUPLICATE_LIB_OK=True python src/harvesting/DeepSpaceHarvester.py
```

**3. Process the Topology into Graphs**
Strip the raw geometry of arbitrary identifiers and weave it into 3D Graph structures for the neural network.
```bash 
KMP_DUPLICATE_LIB_OK=True python src/processing/SmartGraphBuilder.py
```

**4. Train the Models**
Train the God Mode CVAE (Conditional Variational Autoencoder) to map the continuous probability cloud of the universes.
```bash
KMP_DUPLICATE_LIB_OK=True python src/training/UniverseGenerator.py
```

**5. God Mode (Inference)**
Command the trained CVAE to hallucinate a brand new universe by passing it arbitrary physical laws.
```bash
KMP_DUPLICATE_LIB_OK=True python src/inference/GenerateUniverse.py
```