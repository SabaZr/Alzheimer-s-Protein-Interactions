from GNN import convert_nx_to_pyg, train_gnn
from similarity import parse_sequence_similarity, parse_functional_similarity, parse_structural_similarity
from lif import run_spiking_network
from fusion_link_prediction import fuse_all_features, create_similarity_graph, create_similarity_graph_approx, combined_link_prediction, perform_lava_link_prediction, prepare_data_for_graph
from HDC import hdc_encode_features

import time
import os
import re
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import shap
import ray

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# DEAP imports
from deap import base, creator, tools, algorithms
from networkx.algorithms.community import greedy_modularity_communities


# Paths for similarity data files
sequence_similarity_path = r"C:\Users\pc\Desktop\ALZ\uniprot_sprot.dat.gz"
kgml_file = r"C:\Users\pc\Desktop\ALZ\kgml.txt"
structural_similarity_path = r"C:\Users\pc\Desktop\ALZ\1hh3.pdb1.gz"
# Call similarity functions
parse_sequence_similarity(sequence_similarity_path)
parse_functional_similarity(kgml_file)
parse_structural_similarity(structural_similarity_path)

def add_similarity_edges(G, seq_sim, func_sim, struct_sim, threshold=0.7):
    """
    For every pair of nodes, compute the average similarity (using sequence, functional, and structural scores).
    If the average exceeds the threshold and no edge exists, add a weighted edge.
    """
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            avg_sim = ((seq_sim[i, 0] + seq_sim[j, 0]) +
                       (func_sim[i, 0] + func_sim[j, 0]) +
                       (struct_sim[i, 0] + struct_sim[j, 0])) / 6.0
            if avg_sim > threshold and not G.has_edge(nodes[i], nodes[j]):
                G.add_edge(nodes[i], nodes[j], weight=avg_sim)
    return G

def enhance_node_features(node_features, seq_sim, func_sim, struct_sim):
    """
    Concatenate original node features with similarity score vectors.
    """
    return np.concatenate([node_features, seq_sim, func_sim, struct_sim], axis=1)

# Global best parameters for the RF model used in GA fitness evaluation
best_params_global = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.8
}
# Bonus weight used in GA fitness evaluation.
bonus_weight = 0.05

# -------------------------------
# PPI DATA LOADING & PROCESSING FUNCTIONS
# -------------------------------
def compute_betweenness(g):
    return np.mean(g.betweenness())

def compute_closeness(g):
    return np.mean(g.closeness())

def compute_cluster(g):
    return g.transitivity_undirected()

def compute_eigen(g):
    return np.mean(g.eigenvector_centrality())
def load_ppi_data(file_path):
    """Loads PPI data from the specified .tab3 file into a DataFrame."""
    print(f"Loading PPI data from {file_path}...")
    return pd.read_csv(file_path, sep='\t', low_memory=False)

def filter_alzheimer_ppi(ppi_data, alz_proteins):
    """Filters the PPI data for human-only interactions (9606) and Alzheimer’s-related proteins."""
    required_cols = ["Official Symbol Interactor A", "Official Symbol Interactor B", 
                     "Organism ID Interactor A", "Organism ID Interactor B"]
    if not all(col in ppi_data.columns for col in required_cols):
        raise ValueError("Missing required columns in the dataset.")
    ppi_filtered = ppi_data[
        (ppi_data["Organism ID Interactor A"] == 9606) & 
        (ppi_data["Organism ID Interactor B"] == 9606) &
        ((ppi_data["Official Symbol Interactor A"].isin(alz_proteins)) |
         (ppi_data["Official Symbol Interactor B"].isin(alz_proteins)))
    ]
    return ppi_filtered

def build_ppi_graph(ppi_filtered):
    """Creates a NetworkX graph from the filtered PPI data."""
    G = nx.from_pandas_edgelist(
        ppi_filtered,
        source="Official Symbol Interactor A",
        target="Official Symbol Interactor B"
    )
    return G
# -------------------------------
# Helper Normalization Functions and Alias Mapping
# -------------------------------
alias_mapping = {
    "APP": "APP",        
    "MAPT": "TAU",       
    "PSEN1": "PSEN1",    
    "PSEN2": "PSEN2",    
    "APOE": "APOE",      
    "CLU": "CLU",        
    "TREM2": "TREM2",    
    "GRN": "GRN",        
    "BIN1": "BIN1",      
    "CD33": "SIGLEC3",   
}

def normalize_name(name):
    norm = str(name).strip().upper()
    norm = re.sub(r'[^A-Z0-9]', '', norm)
    return norm
def normalize_name_with_alias(name):
    norm = normalize_name(name)
    return alias_mapping.get(norm, norm)
# -------------------------------
# Hyperdimensional Computing Function
# -------------------------------
def hdc_encode_features(X, hd_dim=1024):
    n_samples, n_features = X.shape
    feature_hypervectors = np.random.choice([-1, 1], size=(n_features, hd_dim)).astype(np.float32)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    normalized_X = (X - min_vals) / (max_vals - min_vals + 1e-8)
    encoded_samples = np.sign(normalized_X @ feature_hypervectors).astype(np.float32)
    return encoded_samples, feature_hypervectors
# -------------------------------
# Preprocessing and Feature Filtering Function
# -------------------------------
@ray.remote
def preprocess_data(X, y, num_var_threshold=0.001, cat_var_threshold=0.001,  
                    use_mi=False, use_svd=False, svd_components=None,
                    high_dim_threshold=500, apply_light_filtering=True, 
                    pre_filter_percent=0.5, cat_cardinality_threshold=100):
    # Separate numeric and categorical columns.
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Reduce high-cardinality categorical columns.
    X_cat = X[categorical_features].copy()
    for col in categorical_features:
        unique_vals = X_cat[col].nunique()
        if unique_vals > cat_cardinality_threshold:
            # Keep only the top frequent categories; replace the rest with "other".
            top_categories = X_cat[col].value_counts().nlargest(cat_cardinality_threshold).index
            X_cat[col] = X_cat[col].apply(lambda x: x if x in top_categories else "other")
            print(f"Column '{col}' reduced from {unique_vals} to {X_cat[col].nunique()} unique values.")
    # Update original dataframe with reduced categorical values.
    X.update(X_cat)    
    
    # Build numeric and categorical pipelines.
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=num_var_threshold)),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)
    print(f"🔹 After preprocessing, data shape: {X_processed.shape}")

    # Apply SVD reduction if requested.
    if use_svd and svd_components is not None:
        print("🔹 Applying SVD reduction...")
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=svd_components, n_iter=5, random_state=42)
        X_processed = svd.fit_transform(X_processed)
        print(f"🔹 After SVD, data shape: {X_processed.shape}")

    # Apply smart pre-filtering for high-dimensional data.
    if apply_light_filtering and X_processed.shape[1] > high_dim_threshold:
        print(f"⚠️ High-dimensional data detected ({X_processed.shape[1]} features). Applying smart pre-filtering.")
        n_features_total = X_processed.shape[1]
        n_keep = max(int(n_features_total * pre_filter_percent), 10)
        
        # Convert to dense if needed.
        X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        variances = np.var(X_dense, axis=0)
        top_variance_indices = np.argsort(variances)[-n_keep:]
        
        if y is not None:
            if pd.api.types.is_numeric_dtype(y):
                mi_scores = mutual_info_regression(X_dense, y)
            else:
                mi_scores = mutual_info_classif(X_dense, y)
            top_mi_indices = np.argsort(mi_scores)[-n_keep:]
            selected_indices = np.unique(np.concatenate([top_variance_indices, top_mi_indices]))
        else:
            selected_indices = top_variance_indices

        if len(selected_indices) < max(10, n_features_total * 0.05):
            selected_indices = np.argsort(variances)[-max(10, int(n_features_total * 0.05)):]
        
        X_processed = X_dense[:, selected_indices]
        print(f"🔹 After smart pre-filtering, data shape: {X_processed.shape} (Retained {len(selected_indices)} features)")
    
    return X_processed, np.array(y)

# -------------------------------
# GA-Based Feature Selection Functions (Graph-Aware Operators)
# -------------------------------
# Create DEAP types.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def graph_aware_individual(G, n_features, mapping):
    """Custom initialization: favor features with high centrality."""
    # Using a list comprehension to build the individual.
    return creator.Individual([1 if random.random() < centrality_dict.get(i, 0.3) else 0 for i in range(n_features)])

def graph_aware_crossover(G, ind1, ind2, mapping):
    """Graph-aware crossover: swap clusters between parents."""
    reverse_mapping = {node: i for i, node in mapping.items()}
    communities = list(greedy_modularity_communities(G))
    for comm in communities:
        valid_indices = [reverse_mapping[node] for node in comm if node in reverse_mapping and reverse_mapping[node] < len(ind1)]
        if valid_indices and random.random() < 0.5:
            for idx in valid_indices:
                ind1[idx], ind2[idx] = ind2[idx], ind1[idx]
    return ind1, ind2

def graph_aware_mutation(G, individual, indpb, mapping):
    """Graph-aware mutation: lower mutation chance for high-centrality features."""
    for i in range(len(individual)):
        cent = centrality_dict.get(i, 0.3)
        effective_indpb = indpb * (1 - cent)
        if random.random() < effective_indpb:
            individual[i] = 1 - individual[i]
    return individual,

def connectivity_bonus(individual, G, mapping):
    """Fitness bonus: reward individuals that maintain graph connectivity."""
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected_indices:
        return 0
    sub_nodes = [mapping[i] for i in selected_indices if i in mapping]
    subG = G.subgraph(sub_nodes)
    components = nx.number_connected_components(subG)
    bonus = 1 / components if components > 0 else 0
    centrality_sum = sum(centrality_dict.get(i, 0.3) for i in selected_indices)
    return bonus * (centrality_sum / len(selected_indices))

eval_cache = {}
bonus_weight = 0.05   # Weight for connectivity bonus.
penalty_weight = 0.1  # Penalty for fraction of features selected.

def evalClassifier(individual, G, mapping):
    """Improved fitness evaluation: classifier performance plus connectivity bonus minus penalty for selecting too many features."""
    key = tuple(individual)
    if key in eval_cache:
        return eval_cache[key]
    
    mask = np.array(individual, dtype=bool)
    n_selected = mask.sum()
    total_features = len(individual)
    
    if n_selected == 0:
        fitness_val = (0.0,)
        eval_cache[key] = fitness_val
        return fitness_val
    
    X_selected = X_ga[:, mask]
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=50,  # Smaller forest for GA evaluation.
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42
    )
    try:
        from joblib import parallel_backend
        with parallel_backend('loky', n_jobs=-1):
            score = cross_val_score(clf, X_selected, y_ga, cv=cv, scoring='accuracy').mean()
    except Exception:
        fitness_val = (0.0,)
        eval_cache[key] = fitness_val
        return fitness_val
    
    bonus = connectivity_bonus(individual, G, mapping)
    feature_penalty = (n_selected / total_features)
    fitness_value = score + (bonus_weight * bonus) - (penalty_weight * feature_penalty)
    fitness_val = (fitness_value,)
    eval_cache[key] = fitness_val
    return fitness_val


# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 1. Load and Preprocess Alzheimer’s Data from PPI File.
        tab3_file_path = r"C:\Users\pc\Desktop\BIOGRID-PROJECT-alzheimers_disease_project-4.4.243\BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-4.4.243.tab3.txt"
        ppi_data_full = load_ppi_data(tab3_file_path)
        
        # Since the dataset has one class, we assign the same label to all samples.
        if "class" not in ppi_data_full.columns:
            ppi_data_full["class"] = 1
        y_all = ppi_data_full["class"]
        X_all = ppi_data_full.drop(columns=["class"])
        dataset_name = "BioGRID Alzheimer PPI Data (Feature Selection)"
        print(f"Dataset: {dataset_name}")
        print(f"Original data shape: {X_all.shape}")
        print(f"Number of classes: {len(np.unique(y_all))}")
        
        X_tune, y_tune = ray.get(preprocess_data.remote(
            X_all, y_all, 
            num_var_threshold=0.0001, 
            cat_var_threshold=0.0001, 
            use_mi=False, 
            use_svd=True,
            svd_components=500
        ))
        print(f"After preprocessing, data shape: {X_tune.shape}")
        
        if X_tune.shape[1] > 500:
            print("Warning: Preprocessing did not reduce dimensions as expected. Applying manual SVD reduction.")
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=500)
            X_tune = svd.fit_transform(X_tune)
            print(f"Data shape after manual SVD: {X_tune.shape}")
        print(f"Final preprocessed data shape: {X_tune.shape}")
        
        # 2. Filter the PPI data for Alzheimer-related interactions.
        alz_proteins = [
            "APP", "PSEN1", "PSEN2", "APOE", "MAPT", "TREM2", "CLU", "CR1",
            "BIN1", "CD33", "ABCA7", "SORL1", "PICALM"
        ]
        ppi_filtered = filter_alzheimer_ppi(ppi_data_full, alz_proteins)
        print(f"Filtered Alzheimer-related PPI interactions: {ppi_filtered.shape[0]}")
        filtered_csv = "alzheimer_ppi_filtered.csv"
        ppi_filtered.to_csv(filtered_csv, index=False)
        print(f"Filtered PPI data saved as '{filtered_csv}'.")
        
        # 3. Build the PPI Graph and Compute Global Metrics.
        G_ppi = build_ppi_graph(ppi_filtered)
        print(f"PPI Graph constructed with {G_ppi.number_of_nodes()} nodes and {G_ppi.number_of_edges()} edges.")
        avg_degree = np.mean(list(dict(G_ppi.degree()).values()))
        print(f"Average degree in PPI network: {avg_degree:.2f}")
        
        import igraph as ig
        edge_list = list(G_ppi.edges())
        g = ig.Graph.TupleList(edge_list, directed=False)
        with ProcessPoolExecutor() as executor:
            future_betw = executor.submit(compute_betweenness, g)
            future_close = executor.submit(compute_closeness, g)
            future_cluster = executor.submit(compute_cluster, g)
            future_eigen = executor.submit(compute_eigen, g)
            avg_betweenness = future_betw.result()
            avg_closeness = future_close.result()
            avg_clustering = future_cluster.result()
            avg_eigen = future_eigen.result()
        print(f"Global PPI metrics: Betweenness={avg_betweenness:.4f}, Closeness={avg_closeness:.4f}, "
              f"Clustering={avg_clustering:.4f}, Eigenvector={avg_eigen:.4f}")
        
        # 4. GA-based Feature Selection using DEAP with Graph-Aware Operators.
        global X_ga, y_ga, centrality_dict
        X_ga = X_tune
        y_ga = y_tune
        G_enriched = G_ppi  # Use built graph.
        G_renorm = G_enriched
        n_features = min(X_ga.shape[1], G_renorm.number_of_nodes())
        centrality_dict = nx.eigenvector_centrality_numpy(G_renorm)
        feature_mapping = {i: node for i, node in enumerate(list(G_renorm.nodes())) if i < n_features}
        
        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox.register("individual", partial(graph_aware_individual, G_renorm, n_features, feature_mapping))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", lambda ind1, ind2: graph_aware_crossover(G_renorm, ind1, ind2, feature_mapping))
        toolbox.register("mutate", lambda ind: graph_aware_mutation(G_renorm, ind, indpb=0.05, mapping=feature_mapping))
        toolbox.register("evaluate", lambda ind: evalClassifier(ind, G_renorm, feature_mapping))
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop_size = 1
        n_generations = 1
        population = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        print("\nStarting GA-based feature selection with graph-aware operators (revised)...")
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations,
            stats=stats, halloffame=hof, verbose=True
        )
        
        best_individual = hof[0]
        best_mask = np.array(best_individual, dtype=bool)
        best_score = evalClassifier(best_individual, G_renorm, feature_mapping)[0]
        print(f"Best GA fitness (accuracy + bonus): {best_score:.4f}")
        print(f"Number of selected features by GA: {best_mask.sum()} out of {n_features}")
        if best_mask.sum() == 0:
            print("GA did not select any features. Using all graph-selected features.")
            X_final = X_ga
            final_selected_indices = np.arange(n_features)
        else:
            X_final = X_ga[:, best_mask]
            final_selected_indices = np.arange(n_features)[best_mask]
        print(f"Final dataset shape after GA-based selection: {X_final.shape}")
        
        # 5. Model Training and Evaluation (RandomForest).
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        # Since there is only one class, we create a random split to simulate an independent evaluation.
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_tune, test_size=0.3, random_state=42)
        final_model = RandomForestClassifier(
            n_estimators=best_params_global['n_estimators'],
            max_depth=best_params_global['max_depth'],
            min_samples_split=best_params_global['min_samples_split'],
            min_samples_leaf=best_params_global['min_samples_leaf'],
            max_features=best_params_global['max_features'],
            random_state=42
        )
        final_model.fit(X_train, y_train)
        train_pred = final_model.predict(X_train)
        test_pred = final_model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"\nFinal RF model trained on {X_final.shape[1]} features:")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        
        from joblib import parallel_backend as jb_parallel_backend
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        with jb_parallel_backend("loky", n_jobs=-1):
            cv_scores = cross_val_score(final_model, X_final, y_tune, cv=cv, scoring='accuracy')
        print(f"\nRF cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        np.savetxt('selected_features_ga.txt', np.where(best_mask)[0], fmt='%d')
        print("Selected feature indices saved to 'selected_features_ga.txt'")
        output_directory = os.path.join(os.path.expanduser("~"), "Desktop", "bio-inspired", "data")
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, "dataset_ga.csv")
        y_tune = np.ravel(y_tune)
        pd.DataFrame(X_final).assign(label=y_tune).to_csv(output_path, index=False)
        print(f"Dataset with GA-selected features saved to: {output_path}")
        
        # 6. SHAP, Hyperdimensional Encoding, and Integration with Detailed PPI Metrics.
        import shap
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_final)
        abs_shap = np.mean([np.abs(shp) for shp in shap_values], axis=0) if isinstance(shap_values, list) else np.abs(shap_values)
        shap_importance = np.mean(abs_shap, axis=0)
        shap_dict = {orig_idx: importance for orig_idx, importance in zip(final_selected_indices, shap_importance)}
        for node in G_renorm.nodes():
            orig_idx = G_renorm.nodes[node].get('orig_index', None)
            if orig_idx is not None and orig_idx in shap_dict:
                G_renorm.nodes[node]['shap_importance'] = shap_dict[orig_idx]
        selected_nodes = [feature_mapping[i] for i, bit in enumerate(best_individual) if bit == 1]
        G_ga = G_renorm.subgraph(selected_nodes)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        pos1 = nx.spring_layout(G_renorm, seed=42, k=1.5)
        nx.draw_networkx(G_renorm, pos1, node_color='skyblue', node_size=500, with_labels=True)
        plt.title("Graph before GA-based Selection")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        pos2 = nx.spring_layout(G_ga, seed=42, k=1.5)
        nx.draw_networkx(G_ga, pos2, node_color='salmon', node_size=500, with_labels=True)
        plt.title("Graph after GA-based Selection")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        print("\nApplying hyperdimensional encoding on GA-selected features...")
        hd_encoded_samples, feature_hvs = hdc_encode_features(X_final, hd_dim=4096)
        print(f"Hyperdimensional encoded samples shape: {hd_encoded_samples.shape}")
        ppi_metrics = np.array([avg_degree, avg_betweenness, avg_closeness, avg_clustering, avg_eigen])
        ppi_feature = np.tile(ppi_metrics, (hd_encoded_samples.shape[0], 1))
        integrated_features = np.concatenate([hd_encoded_samples, ppi_feature], axis=1)
        print(f"Integrated feature matrix shape (HDC + PPI metrics): {integrated_features.shape}")
        
        # 7. Similarity Integration (Graph Enrichment and Feature Enhancement)
        print("\n--- Similarity Integration ---")
        # Define the number of nodes from final features
        num_nodes = X_final.shape[0]
        # Generate dummy similarity scores (replace with real similarity measures if available)
        dummy_seq_sim = np.random.rand(num_nodes, 1).astype(np.float32)
        dummy_func_sim = np.random.rand(num_nodes, 1).astype(np.float32)
        dummy_struct_sim = np.random.rand(num_nodes, 1).astype(np.float32)
        # Use the main graph G_ppi as the full graph
        G_full = G_ppi.copy()  # Corrected: replacing undefined 'G' with 'G_ppi'
        # Approach 1: Graph Enrichment – add new edges based on average similarity
        G_enriched_similarity = add_similarity_edges(G_full.copy(), dummy_seq_sim, dummy_func_sim, dummy_struct_sim, threshold=0.7)
        print(f"[DEBUG] Enriched Graph - Nodes: {G_enriched_similarity.number_of_nodes()}, Edges: {G_enriched_similarity.number_of_edges()}")
        # Approach 2: Feature Enhancement – concatenate similarity scores to GA-selected features
        enhanced_node_features = enhance_node_features(X_final, dummy_seq_sim, dummy_func_sim, dummy_struct_sim)
        print(f"[DEBUG] Enhanced node features shape: {enhanced_node_features.shape}")
        # If df is not defined, use ppi_data_full (or define df appropriately)
        # For example, if you intended to use ppi_data_full:
        df_protein_interactions = ppi_data_full[['Official Symbol Interactor A']].copy()
        df_protein_interactions = df_protein_interactions.rename(columns={'Official Symbol Interactor A': 'protein'})
        # If a protein appears multiple times, compute its average feature vector.
        # X_final must be converted into a dataframe aligned with ppi_data_full rows.
        df_features = pd.DataFrame(X_final)
        df_features['protein'] = df_protein_interactions['protein'].values
        # Now group by protein and take the mean.
        df_aggregated = df_features.groupby('protein').mean()
        # Now df_aggregated has one row per unique protein.
        aggregated_features = df_aggregated.values
        df_proteins = df_aggregated.reset_index()[['protein']]
        node_feats_enhanced, node_list_enhanced, G_full_enhanced = prepare_data_for_graph(G_full, aggregated_features, df_proteins)
        # Convert enriched graph to PyG format using enhanced node features
        graph_data_enriched = convert_nx_to_pyg(G_enriched_similarity, node_feats_enhanced)
        print(f"[DEBUG] Converted graph_data_enriched with {graph_data_enriched.num_nodes} nodes")
        # Train the GNN on enriched graph
        gnn_embeddings_enriched = train_gnn(graph_data_enriched)
        print(f"[DEBUG] GNN Embeddings Shape: {gnn_embeddings_enriched.shape}")  
        aggregated_node_features = node_feats_enhanced
        spike_embeddings = run_spiking_network(aggregated_node_features, num_steps=20, batch_size=50)  
        print(f"[DEBUG] Spike embeddings shape: {spike_embeddings.shape}")
        # Define topology features and group features as dummy arrays with 50 features each.
        topo_features = np.zeros((spike_embeddings.shape[0], 50), dtype=np.float32)
        group_features = np.zeros((spike_embeddings.shape[0], 50), dtype=np.float32)
        # Fuse features for downstream tasks
        fused_embeddings = fuse_all_features(spike_embeddings, topo_features, group_features, gnn_embeddings_enriched)
        print(f"[DEBUG] Final Fused Embeddings Shape: {fused_embeddings.shape}")
        
        
        # 8. 3D Plotly Visualization of Fused Embeddings
        import plotly.express as px
        from sklearn.decomposition import PCA

        print("\n--- 3D Visualization ---")
        # Apply PCA to reduce the fused embeddings to 3 components.
        pca = PCA(n_components=3)
        fused_3d = pca.fit_transform(fused_embeddings)

        # Create dummy labels for visualization.
        # Here, we simply use the indices of the samples as labels.
        dummy_labels = np.arange(fused_3d.shape[0])

        # Plot the 3D scatter plot. Replace dummy_labels with actual labels if available.
        fig = px.scatter_3d(
            x=fused_3d[:, 0],
            y=fused_3d[:, 1],
            z=fused_3d[:, 2],
            color=dummy_labels.astype(str),  # Using dummy labels for coloring.
            title="3D Visualization of Fused Embeddings (PCA-reduced)",
            labels={"x": "PC1", "y": "PC2", "z": "PC3"}
        )
        fig.update_traces(marker=dict(size=4))
        fig.show()

        # Create similarity-based graph
        #G_similarity = create_similarity_graph(node_list_enhanced, fused_embeddings, sim_threshold=0.5)
        
        #with annoy
        G_similarity = create_similarity_graph_approx(node_list_enhanced, fused_embeddings, sim_threshold=0.5, top_k=15)
        
        # Combine predictions from LAVA and similarity-based methods
        combined_predictions = combined_link_prediction(
            
            fused_embeddings, node_list_enhanced, G_ppi,
            lava_threshold=0.4, sim_threshold=0.5,
            alpha=0.5, beta=0.5, top_k=15, max_candidates=500
        )

        
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        ray.shutdown()
        print("Ray has been shut down.")

if __name__ == "__main__":
    main()










