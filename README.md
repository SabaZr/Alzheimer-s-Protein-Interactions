# Alzheimer’s Protein–Protein Interaction Prediction

This project presents a scalable framework for predicting missing or novel protein–protein interactions (PPIs) associated with Alzheimer’s disease. It integrates graph-based modeling, biologically inspired learning, and high-dimensional encoding techniques to enhance prediction performance.

## 🔬 Overview

- **Graph Construction & Enrichment**
  - Based on known PPI interactions (e.g., from BioGRID/STRING).
  - Augmented using sequence, structural, and functional similarities.
  - Similarities are incorporated either as weighted edges or node features.

- **Feature Selection with Genetic Algorithm (DEAP)**
  - A graph-aware genetic algorithm is used to select informative features.
  - The refined feature set improves downstream GNN performance and preserves biological relevance.

- **Embedding Generation**
  - Protein nodes are encoded using a combination of:
    - **Hyperdimensional Computing (HDC)**
    - **Spiking Neural Networks (SNNs)**
  - Fused embeddings capture diverse biological representations.

- **Link Prediction**
  - Combines cosine similarity (LAVA) and approximate nearest neighbor search (**Annoy**) for scalable PPI prediction.
  - Threshold-tuned predictions help control graph density and filter noise.

- **Distributed Processing**
  - Leverages **Ray** for parallel and scalable computations across stages such as embedding generation, prediction, and evaluation.

## 🧠 Key Components

| Module                  | Description |
|-------------------------|-------------|
| `alzhimerintel1.py`     | Main script for running the pipeline |
| `GNN.py`                | Implements GNN architecture |
| `HDC.py`                | Encodes protein data using HDC |
| `lif.py`                | Simulates Leaky Integrate-and-Fire neuron models |
| `fusion_link_prediction.py` | Combines LAVA and similarity-based link predictions |
| `evaluate.py`           | Evaluation metrics and performance tracking |
| `similarity.py`         | Computes structural, sequence, and functional similarities |

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- NetworkX
- Scikit-learn
- DEAP
- Ray
- Annoy
- NumPy, Pandas, Matplotlib

  🧬 Datasets Used
The following publicly available datasets were used in this project:

BioGRID Alzheimer’s Project
Release 4.4.244 (2024)
🔗 [Download Dataset](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.244/BIOGRID-PROJECT-alzheimers_disease_project-4.4.244.zip)
Source: The BioGRID Project

UniProtKB/Swiss-Prot
Manually curated protein data (2024)
🔗 [UniProt Website](https://www.uniprot.org/uniprotkb)
Source: The UniProt Consortium

KEGG Alzheimer’s Pathway
hsa05010 pathway (2024)
🔗 [https://www.kegg.jp/entry/hsa05010](https://www.kegg.jp/entry/hsa05010)
Source: Kanehisa Laboratories

RCSB Protein Data Bank (PDB)
3D structural data for protein models
🔗 [https://www.rcsb.org/docs/general-help/organization-of-3d-structures-in-the-protein-data-bank](https://www.rcsb.org/docs/general-help/organization-of-3d-structures-in-the-protein-data-bank)

## 📈 Output
Ranked list of predicted missing PPIs with cosine similarity scores.

Validation against known databases (BioGRID, STRING, HPA).

Visualizations of refined graph structure and prediction confidence

## 📜 License

This project is for academic and research purposes only.

---

Feel free to cite this repository in your research if you find it useful.
