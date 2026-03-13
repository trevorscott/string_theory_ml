# 🌌 String Theory ML: Mapping the Calabi-Yau Landscape

*“There are two ways to do great mathematics. The first is to be smarter than everybody else. The second way is to be stupider than everybody else — but persistent.” — Raoul Bott*

Or...

Be stupider & just use AI.

## Overview
This repository is a local, end-to-end Machine Learning pipeline designed to explore the structural DNA of the universe. 

In string theory, the physics of our 4D reality—from the mass of an electron to the strength of gravity—is dictated by the geometry of 6 extra, invisible dimensions curled up into Calabi-Yau manifolds. With an estimated $10^{500}$ possible shapes, finding the one that matches our universe is fundamentally a Big Data problem.

This project bridges theoretical physics and deep learning. It automates the harvesting of multi-dimensional geometric scaffolds from the Kreuzer-Skarke database and translates their topology into Graph Neural Networks (GNNs). 

The repository is structured as a **"living history" monorepo**, tracking the evolution from standard continuous generative models (V1) to state-of-the-art discrete denoising architectures (V2).

## Pipeline Architecture
The codebase is structured into four experimental stages:

### 1. ⛏️ Harvesting (`src/harvesting/`)
* **Objective:** Mine the string theory landscape for mathematically stable universes.
* **Mechanism:** Uses combinatorial C++ backends to shatter complex $h^{1,1}$ scaffolds, extracting the precise matrices of simplices that form the structural boundaries of Calabi-Yau spaces. Put simply, the $h^{1,1}$ scaffold represents the overarching blueprint and complexity of the extra dimensions, while the "simplices" are the fundamental, indivisible building blocks (like high-dimensional triangles or Lego bricks) that connect together to build the final shape. 

### 2. 🧬 Processing (`src/processing/`)
* **Objective:** Translate discrete geometry into machine-readable mathematical formats.
* **Mechanism:** Strips arbitrary vertex identifiers and maps the continuous manifolds into 3D Graph structures and Tensor datasets, ensuring the AI learns pure spatial relationships rather than memorizing noise.

### 3. 🧠 V1: The Continuous Trap (`architectures/v1_cvae/`)
* **Objective:** Solve the "Inverse Problem" using a Conditional Variational Autoencoder (CVAE).
* **The Architecture:** Takes target physical laws as input, samples a continuous probability cloud, and hallucinates a geometric matrix.
* **The Wall:** Standard VAEs output continuous probabilities (e.g., an edge exists at a 0.87 probability). But Calabi-Yau manifolds are perfectly rigid, discrete topological structures. Attempting to round these probabilities to 0 or 1 destroys the gradients and collapses the physics math. V1 remains in the repo as a foundational stepping stone.

### 4. ⚡ V2: Discrete Graph Diffusion (`architectures/v2_diffusion/`)
* **Objective:** The State-of-the-Art solution for discrete manifold generation.
* **The Architecture:** Abandons continuous Gaussian noise. Instead, it uses a Markov Transition Matrix to violently corrupt perfect Calabi-Yau matrices into binary noise, bit by bit. A `DenseGCNConv` Graph Neural Network is then trained to natively reverse the entropy. 
* **The Result:** The model organically learns the inductive bias of string theory geometry, outputting mathematically pure, rigid matrices natively on Apple Silicon (MPS) or CUDA.

## The Math and the Mission
To find the master equation of reality, we must separate **Topology** (the squishy, invariant capacity for physics) from the **Metric** (the continuous, rigid geometry required to calculate exact particle masses). Finding the metric requires solving the non-linear Monge-Ampère equation—a task impossible for modern supercomputers to do blindly.

This pipeline acts as the ultimate filter. By using AI to master the topology and throw away the wrong universes, we are building the target list for the fault-tolerant quantum computers of the future. 

---

## Prerequisites
Before igniting the pipeline, ensure your system has the following installed:
* **Git**: To clone the repository.
* **Conda / Mamba** (or **uv**): For strict environment and dependency management.
* **Python 3.10+**
* **cytools**: The foundational backend for Calabi-Yau algebraic geometry.
* **PyTorch & PyTorch Geometric**: The engines for the Graph Neural Networks.

## Setup & Execution

*Note: Due to GitHub file size limits, the `data/` (.pt) and `checkpoints/` (.pth) directories are ignored. You must run the Harvesters locally to generate the raw universe datasets.*

**1. Clone the Repository**
```bash
git clone https://github.com/trevorscott/string_theory_ml.git
cd string_theory_ml
```

**2. Initialize the Environment**
Because `string_theory_ml` relies heavily on `cytools` and its underlying C++ algebraic geometry libraries, a Conda environment is required as the base. 

**⚠️ Important macOS / OpenMP Note:** To prevent OpenMP thread contention and segmentation faults between PyTorch and Conda's default Intel Math Kernel Library (`mkl`), we explicitly use `nomkl`. Do *not* use the `KMP_DUPLICATE_LIB_OK=True` workaround, as it will silently bottleneck your CPU threads during training.

```bash
# 1. Create and activate the base conda environment
conda create -n string_theory_ml python=3.10
conda activate string_theory_ml

# 2. Install CYTools (Conda is required here)
conda install -c conda-forge cytools 

# 3. Force Conda to drop Intel MKL dependencies, then cleanly install PyTorch
conda install nomkl
conda install pytorch torchvision torchaudio -c pytorch

# 4. Sync the remaining Python dependencies using uv
uv sync
```

**3. Harvest the Multiverse**
Mine the $h^{1,1}=10$ landscape for mathematically stable Calabi-Yau manifolds. 

```text
NAME
       DeepSpaceHarvester.py - Automated string theory landscape miner

SYNOPSIS
       python src/harvesting/DeepSpaceHarvester.py [OPTIONS]

DESCRIPTION
       Downloads complex topological scaffolds and calculates random triangulations 
       to find mathematically stable universes. Automatically bypasses non-viable 
       geometries and saves the exact structural matrices to the data/ directory.

OPTIONS
       -u, --universes <int>
              The exact number of valid universes to harvest. (default: 1000)
```
**Example:**
```bash
python src/harvesting/DeepSpaceHarvester.py -u 10000
```

**4. Process the Topology into Graphs (Required for V1)**
Strip the raw geometry of arbitrary identifiers and weave it into 3D Graph structures.

```text
NAME
       SmartGraphBuilder.py - Topological graph dataset processor

SYNOPSIS
       python src/processing/SmartGraphBuilder.py -i <input_file> [OPTIONS]
```
**Example:**
```bash 
python src/processing/SmartGraphBuilder.py -i data/deep_space_1000.pt
```

---

### 🚀 Training V2: Discrete Graph Diffusion (State-of-the-Art)
Bypass the continuous rounding trap and train the native discrete denoising architecture.

```text
NAME
       train.py - Discrete Graph Diffusion Engine

SYNOPSIS
       python -m architectures.v2_diffusion.train -f <input_file> [OPTIONS]

DESCRIPTION
       Loads raw harvested universes and dynamically translates them into padded 
       adjacency matrices. Corrupts the structures using a Markov Transition Matrix 
       and trains a DenseGCNConv Graph Neural Network to reverse the entropy, 
       learning the rigid rules of topological physics. Natively supports MPS/CUDA.

OPTIONS
       -f, --file <str>
              (Required) Path to the raw harvested .pt dataset.

       -e, --epochs <int>
              (Optional) Number of training epochs. (default: 50)

       -b, --batch_size <int>
              (Optional) Batch size for the dataloader. (default: 32)
```
**Example:**
```bash
python -m architectures.v2_diffusion.train -f data/deep_space_1000.pt -e 50
```

---

### 🏛️ V1 Legacy: The Continuous CVAE 
*(For historical reference on the continuous-to-discrete gradient trap)*

**Train the Predictive Oracle**
```bash
python architectures/v1_cvae/training/TrainGraphModel.py -i data/smart_graph_1000.pt -e 150
```

**Consult the Oracle (Forward Inference)**
```bash
python architectures/v1_cvae/inference/oracle.py -i data/smart_graph_50.pt
```

**Train the God Mode CVAE**
```bash
python architectures/v1_cvae/training/UniverseGenerator.py -i data/deep_space_1000.pt
```

**God Mode (Reverse Inference)**
```bash
python architectures/v1_cvae/inference/GenerateUniverse.py -p 24.0 -12.0 8.0
```