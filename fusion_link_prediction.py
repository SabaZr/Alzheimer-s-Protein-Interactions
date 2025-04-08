import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import pandas as pd

# -------------------------------
# Fusion and Link Prediction Functions
# -------------------------------
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Feature Fusion and Link Prediction
# -------------------------------

def fuse_all_features(spike_embeddings, topo_features, group_features, gnn_embeddings):
    # Flatten spike_embeddings if needed
    if spike_embeddings.ndim > 2:
        spike_embeddings = spike_embeddings.reshape(spike_embeddings.shape[0], -1)
    # Combine all features
    fused_embeddings = np.concatenate([spike_embeddings, topo_features, group_features, gnn_embeddings], axis=1)
    print(f"Fused embeddings shape: {fused_embeddings.shape}")
    return fused_embeddings


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
########################################################################################
def create_similarity_graph(node_list, feature_matrix, sim_threshold=0.5):

    # محاسبه ماتریس شباهت (cosine similarity)
    sim_matrix = cosine_similarity(feature_matrix)
    np.fill_diagonal(sim_matrix, 0)
    
    G_similarity = nx.Graph()
    G_similarity.add_nodes_from(node_list)
    
    # افزودن لبه‌های وزنی برای پروتئین‌هایی که شباهتشان از آستانه تعیین شده بیشتر است.
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            if sim_matrix[i, j] > sim_threshold:
                G_similarity.add_edge(node_list[i], node_list[j], weight=sim_matrix[i, j])
    
    return G_similarity


from annoy import AnnoyIndex

def create_similarity_graph_approx(node_list, feature_matrix, sim_threshold=0.5, n_trees=10, top_k=10):
    """
    Creates a similarity graph using Annoy for Approximate Nearest Neighbors and early thresholding.
    
    Parameters:
        node_list (List): List of node identifiers
        feature_matrix (np.ndarray): Embeddings for each node
        sim_threshold (float): Minimum cosine similarity to add an edge
        n_trees (int): Number of trees for Annoy index
        top_k (int): Number of nearest neighbors to consider for each node
        
    Returns:
        networkx.Graph: Similarity graph
    """
    dim = feature_matrix.shape[1]
    annoy_index = AnnoyIndex(dim, 'angular')  # 'angular' is for cosine similarity

    # Build the index
    for i in range(len(node_list)):
        annoy_index.add_item(i, feature_matrix[i])
    annoy_index.build(n_trees)

    G = nx.Graph()

    for i in range(len(node_list)):
        G.add_node(node_list[i])
        neighbors = annoy_index.get_nns_by_item(i, top_k + 1, include_distances=True)

        for j, dist in zip(*neighbors):
            if i == j:
                continue
            # Convert angular distance to cosine similarity
            sim = 1 - (dist ** 2) / 2
            if sim >= sim_threshold:
                G.add_edge(node_list[i], node_list[j], weight=sim)

    return G



def combined_link_prediction(fused_embeddings, node_list, G_ppi, lava_threshold=0.4, sim_threshold=0.5,
                               alpha=0.5, beta=0.5, top_k=15, max_candidates=500):
    """
   
    max_candidates limits the total number of candidate pairs to process.
    """
    if len(node_list) != fused_embeddings.shape[0]:
        print(f" Dimension mismatch: node_list has {len(node_list)}, but embeddings have {fused_embeddings.shape[0]}.")
        return []

    # 1. LAVA Predictions (apply filtering if necessary)
    lava_links = perform_lava_link_prediction(fused_embeddings, node_list, G_ppi, threshold=lava_threshold)
    # Optionally, only keep top X lava_links by score:
    lava_links = sorted(lava_links, key=lambda x: x[2], reverse=True)[:max_candidates]

    # 2. Similarity-based Links using Annoy
    G_similarity = create_similarity_graph_approx(node_list, fused_embeddings, sim_threshold=sim_threshold, top_k=top_k)
    similarity_links = [(u, v, G_similarity[u][v]['weight']) for u, v in G_similarity.edges() if not G_ppi.has_edge(u, v)]
    similarity_links = sorted(similarity_links, key=lambda x: x[2], reverse=True)[:max_candidates]

    # 3. Combine LAVA + Similarity
    combined_links = {}
    for u, v, score in lava_links:
        combined_links[(u, v)] = alpha * score
    for u, v, score in similarity_links:
        if (u, v) in combined_links:
            combined_links[(u, v)] += beta * score
        else:
            combined_links[(u, v)] = beta * score

    # 4. Sort and Return
    sorted_links = sorted(combined_links.items(), key=lambda x: x[1], reverse=True)
    if sorted_links:
        print("Top predicted combined missing PPIs:")
        for idx, ((u, v), score) in enumerate(sorted_links[:10], start=1):
            print(f"  {idx}. {u} - {v} | Combined Score: {score:.3f}")
    else:
        print(" No predicted missing PPIs found. Consider adjusting thresholds or top_k.")
    return sorted_links

#################################################################################################
def perform_lava_link_prediction(embeddings, node_list, G_ppi, threshold=0.4):
    """
    Predicts missing links in the full graph using cosine similarity.
    """
    if len(node_list) != embeddings.shape[0]:
        print(f" Dimension mismatch: node_list has {len(node_list)}, but embeddings have {embeddings.shape[0]}.")
        return []

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    print(" Cosine similarity matrix summary:")
    print(f" Min: {np.min(sim_matrix):.3f} | Max: {np.max(sim_matrix):.3f} | Mean: {np.mean(sim_matrix):.3f}")

    # Extract upper triangle indices
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    sim_scores = sim_matrix[triu_indices]
    
    # Filter based on threshold
    valid_indices = np.argwhere(sim_scores > threshold).flatten()
    idx_i, idx_j = triu_indices[0][valid_indices], triu_indices[1][valid_indices]

    predicted_links = []
    for i, j in zip(idx_i, idx_j):
        if i >= len(node_list) or j >= len(node_list):  # Prevent IndexError
            continue

        u, v = node_list[i], node_list[j]
        if not G_ppi.has_edge(u, v):  # Ensure it's a missing edge
            predicted_links.append((u, v, sim_matrix[i, j]))

    # Sort by similarity score in descending order
    predicted_links.sort(key=lambda x: x[2], reverse=True)

    # Display top results
    if predicted_links:
        print(" Top 10 predicted missing PPIs:")
        for idx, (u, v, score) in enumerate(predicted_links[:10], start=1):
            print(f"  {idx}. {u} - {v} | Similarity: {score:.3f}")
    else:
        print(" No predicted missing PPIs found. Try lowering the similarity threshold.")

    return predicted_links
# -------------------------------
# Data Preparation for Subgraph Extraction
# -------------------------------

def prepare_data_for_graph(G_ppi, aggregated_node_features, df):
    """
    Prepares node features and ensures alignment with the full graph (G_ppi).
    
    Parameters:
      - G_ppi: a NetworkX graph containing unique proteins.
      - aggregated_node_features: a feature matrix with shape (num_unique_proteins, feature_dim)
      - df: a DataFrame that contains at least a 'protein' column with unique protein names.
    
    Returns:
      - final_node_features: a NumPy array with node features (after grouping if needed)
      - node_list: a list of protein names
      - G_ppi: the unchanged input graph
    """
    # Determine the protein identifier column.
    if "protein" in df.columns:
        protein_col = "protein"
    else:
        raise KeyError("No valid protein column found in the dataframe.")
    
    # Here we assume df already has one row per unique protein (e.g., 100 rows)
    if aggregated_node_features.shape[0] != len(df):
        raise ValueError(
            f"Mismatch: aggregated_node_features has {aggregated_node_features.shape[0]} rows, "
            f"but df has {len(df)} rows."
        )
    
    # Build a DataFrame using the aggregated features and protein names
    df_features = pd.DataFrame(aggregated_node_features, index=df.index)
    df_features[protein_col] = df[protein_col].values

    # Group by protein (this should not change the count if df already has unique proteins)
    grouped = df_features.groupby(protein_col).mean()
    final_node_features = grouped.values  # shape: (num_unique_proteins, feature_dim)
    node_list = grouped.index.tolist()      # list of protein names
    
    print(f"Graph contains {G_ppi.number_of_nodes()} nodes and {G_ppi.number_of_edges()} edges.")
    return final_node_features, node_list, G_ppi




