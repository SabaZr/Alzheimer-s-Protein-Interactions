# 🧠 Alzheimer’s Protein–Protein Interaction Prediction

**[📄 View Preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.11.647919v1)**   
**📝 Title:** _Graph-Based Modeling of Alzheimer's Protein Interactions via Spiking Neural, Hyperdimensional Encoding, and Scalable Ray-Based Learning_  
**👤 Author:** Saba Zare

This project introduces a scalable and biologically inspired framework to predict missing or novel protein–protein interactions (PPIs) related to **Alzheimer’s disease**. It integrates graph modeling, hyperdimensional encoding, spiking neural networks (SNNs), and evolutionary feature selection for accurate and efficient link prediction.

---

## 🔬 Overview

- **Graph Construction**
  - Built from curated PPI databases (e.g., BioGRID, UniProt, KEGG).
  - Enriched using sequence, structural, and functional similarity.
  - Similarity scores are encoded as weighted edges or node attributes.

- **Feature Selection**
  - A graph-aware **Genetic Algorithm (via DEAP)** filters biologically relevant features.
  - Reduces dimensionality and enhances GNN performance.

- **Embedding & Learning**
  - Proteins are embedded using:
    - **Hyperdimensional Computing (HDC)**
    - **Spiking Neural Networks (LIF neurons)**
    - **Graph Neural Networks (GNNs)**

- **Link Prediction**
  - Combines:
    - **LAVA** (cosine similarity metric)
    - **Annoy** (Approximate Nearest Neighbors)
  - Enables scalable and threshold-tuned PPI inference.

- **Distributed Computing**
  - Powered by **Ray** for parallel embedding generation, prediction, and evaluation.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `alzhimerintel1.py` | Main execution script |
| `GNN.py` | Graph Neural Network implementation |
| `HDC.py` | Hyperdimensional encoding module |
| `lif.py` | Leaky Integrate-and-Fire SNN simulation |
| `fusion_link_prediction.py` | Hybrid link prediction logic |
| `evaluate.py` | Evaluation metrics and graph analytics |
| `similarity.py` | Computation of biological similarities |

---

## 🧬 Datasets Used

| Dataset | Description | Link |
|--------|-------------|------|
| **BioGRID Alzheimer’s Project** | Curated PPIs for Alzheimer's disease | [Download](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.244/BIOGRID-PROJECT-alzheimers_disease_project-4.4.244.zip) |
| **UniProtKB/Swiss-Prot** | Annotated protein database | [Visit](https://www.uniprot.org/uniprotkb) |
| **KEGG Pathway hsa05010** | Alzheimer’s disease pathway | [Visit](https://www.kegg.jp/entry/hsa05010) |
| **RCSB PDB** | Protein 3D structural data | [Visit](https://www.rcsb.org/docs/general-help/organization-of-3d-structures-in-the-protein-data-bank) |

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- NetworkX
- Scikit-learn
- DEAP
- Ray
- Annoy
- NumPy, Pandas, Matplotlib

  ## 📣 Citation
  If you use this work, please cite the preprint:
@article{zare2025alzheimers,
  author = {Saba Zare},
  title = {Graph-Based Modeling of Alzheimer's Protein Interactions via Spiking Neural, Hyperdimensional Encoding, and Scalable Ray-Based Learning},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.04.11.647919}
}
