# 🌌 String Theory ML: Mapping the Calabi-Yau Landscape

*“There are two ways to do great mathematics. The first is to be smarter than everybody else. The second way is to be stupider than everybody else — but persistent.” — Raoul Bott*

Or...

Be stupider & just use AI.

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

---

## Prerequisites
Before igniting the pipeline, ensure your system has the following installed:
* **Git**: To clone the repository.
* **Conda / Mamba** (or **uv**): For strict environment and dependency management.
* **Python 3.10+**
* **cytools**: The foundational backend for Calabi-Yau algebraic geometry.
* **PyTorch**: The engine for the Graph Neural Networks and Generative models.

## Setup & Execution

*Note: Due to GitHub file size limits, the `data/` (.pt) and `checkpoints/` (.pth) directories are ignored. You must run the Harvesters locally to generate the raw universe datasets.*

**1. Clone the Repository**
To pull this project onto your local machine, open your terminal and run the following command. You can replace `string_theory_test` with whatever you want to name your local folder:
```bash
git clone [https://github.com/trevorscott/string_theory_ml.git](https://github.com/trevorscott/string_theory_ml.git) string_theory_test
cd string_theory_test
```

**2. Initialize the Environment**
This project requires `cytools` (best installed via conda/mamba) and PyTorch. If you are using `uv` for dependency management (as configured in the `pyproject.toml` and `uv.lock`), sync your environment:

```bash
conda activate cytools
uv sync
```

**3. Harvest the Multiverse**
Mine the $h^{1,1}=10$ landscape for mathematically stable Calabi-Yau manifolds. 

```text
NAME
       DeepSpaceHarvester.py - Automated string theory landscape miner

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/harvesting/DeepSpaceHarvester.py [OPTIONS]

DESCRIPTION
       Downloads complex topological scaffolds and calculates random triangulations 
       to find mathematically stable universes. Automatically bypasses non-viable 
       geometries and saves the exact structural matrices to the data/ directory.
       
       Note: Mac users must prepend KMP_DUPLICATE_LIB_OK=True to prevent OpenMP 
       library collisions during heavy combinatorial math.

OPTIONS
       -u, --universes <int>
              The exact number of valid universes to harvest. The script will 
              dynamically calculate the required number of scaffolds to shatter 
              in order to meet this quota.
              (default: 1000)
```
**Example:**
```bash
KMP_DUPLICATE_LIB_OK=True python src/harvesting/DeepSpaceHarvester.py
```

**4. Process the Topology into Graphs**
Strip the raw geometry of arbitrary identifiers and weave it into 3D Graph structures for the neural network.

```text
NAME
       SmartGraphBuilder.py - Topological graph dataset processor

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/processing/SmartGraphBuilder.py -i <input_file> [OPTIONS]

DESCRIPTION
       Loads raw geometric simplices from the harvester and translates them into 
       undirected mathematical graphs. This strips away arbitrary vertex identifiers 
       and forces the AI to learn pure spatial and topological relationships.

       Note: Mac users must prepend KMP_DUPLICATE_LIB_OK=True to prevent OpenMP 
       library collisions.

OPTIONS
       -i, --input <file>
              (Required) Path to the raw harvested universes file.
              Example: data/deep_space_50000.pt

       -o, --output <file>
              (Optional) Path to save the processed graph dataset. If omitted, 
              the script will automatically mirror the input filename 
              (e.g., data/smart_graph_50000.pt).
```
**Example:**
```bash 
KMP_DUPLICATE_LIB_OK=True python src/processing/SmartGraphBuilder.py -i data/deep_space_1000.pt
```

**5. Train the Oracle (The Forward Problem)**
Train a Graph Neural Network (GNN) to organically deduce the topological intersection numbers (the capacity for physical laws) directly from the discrete mathematical graph of the manifold.

```text
NAME
       TrainGraphModel.py - Train the Predictive Physics Oracle

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/training/TrainGraphModel.py -i <input_file> [OPTIONS]

DESCRIPTION
       Loads the processed 3D Graph datasets and trains a neural network to predict 
       the continuous physical properties of the universe based purely on its 
       geometric fingerprint. This shatters the accuracy floor by learning the 
       inductive bias of string theory.

OPTIONS
       -i, --input <file>
              (Required) Path to the processed graph dataset.
              Example: data/smart_graph_50000.pt

       -o, --output <file>
              (Optional) Path to save the trained Oracle weights.
              (default: checkpoints/gnn_universe_model.pth)

       -e, --epochs <int>
              (Optional) Number of training epochs. (default: 100)
```
**Example:**
```bash
KMP_DUPLICATE_LIB_OK=True python src/training/TrainGraphModel.py -i data/smart_graph_1000.pt -e 150
```

**6. Consult the Oracle (Inference)**
Pass an unknown structural universe to the trained GNN and ask it to predict the laws of physics that would emerge from that geometry.

```text
NAME
       oracle.py - Forward problem inference engine

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/inference/oracle.py -i <input_graph> [OPTIONS]

DESCRIPTION
       Loads the trained Oracle model and a target geometric graph. The AI performs 
       a forward pass, analyzing the topological structure to output the predicted 
       intersection numbers (physical constraints) of that specific manifold without 
       solving the analytical math.

OPTIONS
       -i, --input <file>
              (Required) Path to the single universe graph or dataset to analyze.

       -m, --model <file>
              (Optional) Path to the trained Oracle model weights. 
              (default: checkpoints/gnn_universe_model.pth)
```
**Example:**
```bash
KMP_DUPLICATE_LIB_OK=True python src/inference/oracle.py -i data/smart_graph_50.pt
```

**7. Train the God Mode CVAE**
Train the God Mode CVAE (Conditional Variational Autoencoder) to map the continuous probability cloud of the universe landscape.

```text
NAME
       UniverseGenerator.py - Train the Generative Universe Model

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/training/UniverseGenerator.py -i <input_file> [OPTIONS]

DESCRIPTION
       Loads the raw harvested dataset and trains a Conditional Variational Autoencoder. 
       The model learns to encode geometric blueprints and target physical laws into a 
       latent probability cloud, enabling the reverse-engineering of novel universes.

OPTIONS
       -i, --input <file>
              (Required) Path to the raw harvested dataset. 
              Example: data/deep_space_50000.pt

       -o, --output <file>
              (Optional) Path to save the trained model weights. 
              (default: checkpoints/universe_generator.pth)

       -e, --epochs <int>
              (Optional) Number of training epochs. (default: 50)

       -b, --batch_size <int>
              (Optional) Batch size for the dataloader. (default: 64)
```
**Example:**
```bash
KMP_DUPLICATE_LIB_OK=True python src/training/UniverseGenerator.py -i data/deep_space_1000.pt
```

**8. God Mode (Inference)**
Command the trained CVAE to hallucinate the geometric blueprint of a brand new universe by passing it arbitrary physical laws.

```text
NAME
       GenerateUniverse.py - The Inverse Problem solver

SYNOPSIS
       KMP_DUPLICATE_LIB_OK=True python src/inference/GenerateUniverse.py [OPTIONS]

DESCRIPTION
       Loads the trained Generative God-Mode model and a random latent vector, 
       then conditions the mathematical probability cloud with user-defined 
       physical parameters. The AI will attempt to reverse-engineer and output 
       the structural matrix (simplices) of a Calabi-Yau manifold that 
       satisfies those exact laws.

OPTIONS
       -m, --model <file>
              (Optional) Path to the trained model weights. 
              (default: checkpoints/universe_generator.pth)

       -p, --physics <float> [<float> ...]
              (Optional) A list of target physical properties you want your 
              custom universe to possess (mapped to intersection numbers).
              (default: 24.0 -12.0 8.0)
```
**Example:**
```bash
KMP_DUPLICATE_LIB_OK=True python src/inference/GenerateUniverse.py
```
