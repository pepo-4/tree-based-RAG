import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from umap import UMAP
from raptor.config import Config

def find_optimal_clusters_gmm(
    embeddings: np.ndarray,
    max_k: int = 15,
    random_state: int = 42
) -> Tuple[int, GaussianMixture]:
    """
    Finds the optimal number of clusters k for GMM using Bayesian Information Criterion (BIC).
    Identical to cell 6 of tree_creation.ipynb.
    """
    best_bic = np.inf
    best_k = 1
    best_gmm = None
    
    max_k = min(max_k, len(embeddings))
    if max_k < 2:
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=random_state)
        gmm.fit(embeddings)
        return 1, gmm
        
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=random_state,
            init_params='kmeans'
        )
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm
            
    return best_k, best_gmm

def reduce_dimensions_pca(
    embeddings: np.ndarray,
    n_components: float = Config.PCA_VARIANCE_RATIO,
    random_state: int = 42
) -> np.ndarray:
    """
    Applies PCA retaining specified cumulative variance ratio (default 90% = 0.90).
    Identical to cell 4 of tree_creation.ipynb.
    """
    if len(embeddings) < 2:
        return embeddings
        
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings)

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int = Config.GLOBAL_UMAP_DIM,
    n_neighbors: Optional[int] = None,
    metric: str = "euclidean",
    random_state: int = 42
) -> np.ndarray:
    """
    Global UMAP dimensionality reduction.
    Auto-scales n_neighbors to sqrt(N - 1) if not specified.
    Identical to cell 5 of tree_creation.ipynb.
    """
    if len(embeddings) <= max(dim + 1, 3):
        return embeddings
        
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
        
    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    dim = max(1, min(dim, len(embeddings) - 2))
    
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=random_state
    )
    return umap_model.fit_transform(embeddings)

def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int = Config.LOCAL_UMAP_DIM,
    num_neighbors: int = Config.LOCAL_UMAP_NEIGHBORS,
    metric: str = "euclidean",
    random_state: int = 42
) -> np.ndarray:
    """
    Local UMAP dimensionality reduction.
    Uses fixed num_neighbors=10 as specified in RAPTOR paper & notebook cell 8.
    """
    if len(embeddings) <= max(dim + 1, 3):
        return embeddings
        
    n_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
    dim = max(1, min(dim, len(embeddings) - 2))
    
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=random_state
    )
    return umap_model.fit_transform(embeddings)

def perform_soft_clustering(
    documents: List[Any],
    embeddings: np.ndarray,
    cumulative_threshold: float = Config.CUMULATIVE_PROBABILITY_THRESHOLD,
    level: int = 1
) -> Tuple[Dict[int, Dict[int, List[Any]]], Dict[str, Any]]:
    """
    Executes the exact RAPTOR soft clustering workflow, respecting the notebook's two different pipelines:
    - Pipeline A (Level 1): Exact logic from cells 4-11 (fixed n_neighbors=10 for local, euclidean, etc.)
    - Pipeline B (Level > 1): Exact logic from cell 18 `genera_riassunti_raptor` (dynamic PCA, min_dist=0.0, cosine, init='random', skip local UMAP for N<=3).
    """
    if len(embeddings) == 0:
        return {}, {}
        
    if len(embeddings) < 2:
        return {0: {0: list(documents)}}, {"global_k": 1, "local_clusters_count": 1}
        
    N = len(embeddings)
    
    if level == 1:
        # ============================================================
        # PIPELINE A (Cells 4-11) for Level 1
        # ============================================================
        X_pca = reduce_dimensions_pca(embeddings, n_components=Config.PCA_VARIANCE_RATIO)
        
        # Global UMAP (Cell 5)
        n_neighbors_global = int((N - 1) ** 0.5)
        n_neighbors_global = max(2, min(n_neighbors_global, N - 1)) # Safety bound
        dim_global = min(10, max(1, N - 2))
        X_umap_global = UMAP(
            n_neighbors=n_neighbors_global,
            n_components=dim_global,
            metric="euclidean",
            random_state=42
        ).fit_transform(X_pca)
        
        # Global GMM (Cell 6)
        max_global_k = min(80, len(X_umap_global))
        optimal_global_k, global_gmm = find_optimal_clusters_gmm(X_umap_global, max_k=max_global_k)
        global_probabilities = global_gmm.predict_proba(X_umap_global)
        
        local_cluster_data = {}
        for global_cluster_id in range(optimal_global_k):
            indices = [
                i for i, probs in enumerate(global_probabilities)
                if global_cluster_id in np.argsort(probs)[::-1][
                    :np.searchsorted(np.cumsum(np.sort(probs)[::-1]), cumulative_threshold) + 1
                ]
            ]
            if len(indices) < 2:
                continue
                
            X_local = X_pca[indices]
            
            # Local UMAP (Cell 7 & 8)
            n_neighbors_local = min(10, max(2, len(X_local) - 1))
            dim_local = min(10, max(1, len(X_local) - 2))
            X_umap_local = UMAP(
                n_neighbors=n_neighbors_local,
                n_components=dim_local,
                metric="euclidean",
                random_state=42
            ).fit_transform(X_local)
            
            local_cluster_data[global_cluster_id] = {
                "indices": indices, "X_pca": X_local, "X_umap": X_umap_local
            }
            
        local_topics = {}
        for global_cluster_id, data in local_cluster_data.items():
            X_umap_local = data["X_umap"]
            max_local_k = max(1, int(len(X_umap_local) / 2))
            optimal_local_k, local_gmm = find_optimal_clusters_gmm(X_umap_local, max_k=max_local_k)
            local_probabilities = local_gmm.predict_proba(X_umap_local)
            
            local_topics[global_cluster_id] = {
                "indices": data["indices"], "n_topics": optimal_local_k, "probabilities": local_probabilities
            }
            
    else:
        # ============================================================
        # PIPELINE B (Cell 18) for Level > 1 (genera_riassunti_raptor)
        # ============================================================
        # PCA
        pca_comp = 0.90 if N > 10 else min(N - 1, embeddings.shape[1])
        pca = PCA(n_components=pca_comp, random_state=42)
        X_pca = pca.fit_transform(embeddings)
        
        # Global UMAP
        n_neighbors_global = min(10, max(2, int((len(X_pca) - 1) ** 0.5) + 1))
        umap_dim_global = min(10, max(1, len(X_pca) - 1))
        X_umap_global = UMAP(
            n_neighbors=n_neighbors_global,
            n_components=umap_dim_global,
            metric="euclidean",
            random_state=42
        ).fit_transform(X_pca)
        
        # Global GMM
        optimal_global_k, global_gmm = find_optimal_clusters_gmm(X_umap_global, max_k=min(80, max(2, N // 2)))
        global_probabilities = global_gmm.predict_proba(X_umap_global)
        
        local_cluster_data = {}
        for global_cluster_id in range(optimal_global_k):
            indices = [
                i for i, probs in enumerate(global_probabilities)
                if global_cluster_id in np.argsort(probs)[::-1][
                    :np.searchsorted(np.cumsum(np.sort(probs)[::-1]), cumulative_threshold) + 1
                ]
            ]
            if len(indices) < 2:
                continue
                
            X_local = X_pca[indices]
            n_samples_local = len(X_local)
            
            # Local UMAP - dynamic limits & skip if <= 3
            if n_samples_local <= 3:
                X_umap_local = X_local
            else:
                n_neighbors_local = min(10, max(2, n_samples_local - 1))
                umap_dim_local = min(10, max(1, n_samples_local - 2))
                X_umap_local = UMAP(
                    n_neighbors=n_neighbors_local,
                    n_components=umap_dim_local,
                    min_dist=0.0,
                    metric="cosine",
                    init="random",
                    random_state=42
                ).fit_transform(X_local)
                
            local_cluster_data[global_cluster_id] = {
                "indices": indices, "X_pca": X_local, "X_umap": X_umap_local
            }
            
        local_topics = {}
        for global_cluster_id, data in local_cluster_data.items():
            X_umap_local = data["X_umap"]
            n_samples_local = len(X_umap_local)
            max_local_k = max(2, n_samples_local // 2)
            optimal_local_k, local_gmm = find_optimal_clusters_gmm(X_umap_local, max_k=max_local_k)
            local_probabilities = local_gmm.predict_proba(X_umap_local)
            
            local_topics[global_cluster_id] = {
                "indices": data["indices"], "n_topics": optimal_local_k, "probabilities": local_probabilities
            }

    # Soft assignment to local topics (identical between pipelines A and B)
    local_clusters_chunks = {}
    for global_cluster_id, data in local_topics.items():
        probs_array = data["probabilities"]
        indices_orig = data["indices"]
        n_topics = data["n_topics"]
        
        topic_dict = {i: [] for i in range(n_topics)}
        for i, probs in enumerate(probs_array):
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative = np.cumsum(sorted_probs)
            n_topics_to_take = np.searchsorted(cumulative, cumulative_threshold) + 1
            selected_topics = sorted_indices[:n_topics_to_take]
            
            original_doc_idx = indices_orig[i]
            document = documents[original_doc_idx]
            
            for topic in selected_topics:
                topic_dict[topic].append(document)
                
        local_clusters_chunks[global_cluster_id] = topic_dict
        
    stats = {
        "global_k": optimal_global_k,
        "local_clusters_count": len(local_cluster_data)
    }
    
    return local_clusters_chunks, stats
