"""
Leiden Community Detection Clustering Engine for Fed-LCCF.

Computes item-space Leiden community partitions and community centroids:
1. Normalized pairwise item-item cosine similarity matrix S (1682 x 1682, diagonal zeroed).
2. k-NN undirected graph construction (k=20, positive symmetric weights W_ij = max(S_ij, S_ji)).
3. Modularity-maximizing Leiden community detection (igraph + leidenalg).
4. Validation ensuring complete assignment of all items to communities.
5. Community centroid calculation mu_c in R^32.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import igraph as ig
import leidenalg
import numpy as np

logger = logging.getLogger("fedlccf.leiden_engine")

DEFAULT_K_NEIGHBORS = 20
DEFAULT_NUM_ITEMS = 1682
DEFAULT_LATENT_DIM = 32
DEFAULT_RANDOM_SEED = 42


class LeidenClusteringEngine:
    """
    Item latent space clustering engine utilizing Leiden community detection.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors to connect per item in the k-NN graph (default: 20).
    seed : int
        Random seed for deterministic Leiden partition optimization (default: 42).
    num_items : int
        Number of items in catalog (default: 1682).
    latent_dim : int
        Latent dimension of item factor vectors (default: 32).
    """

    def __init__(
        self,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        seed: int = DEFAULT_RANDOM_SEED,
        num_items: int = DEFAULT_NUM_ITEMS,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> None:
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive; got {k_neighbors}")
        if num_items <= 0:
            raise ValueError(f"num_items must be positive; got {num_items}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")

        self.k_neighbors = int(k_neighbors)
        self.seed = int(seed)
        self.num_items = int(num_items)
        self.latent_dim = int(latent_dim)

    # --------------------------------------------------------------------------
    # Cosine Similarity Calculation
    # --------------------------------------------------------------------------

    def compute_cosine_similarity(self, item_factors: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix S between all item factor vectors.

        Formula:
            S_ij = (v_i . v_j^T) / (||v_i||_2 * ||v_j||_2)
            S_ii = 0.0 (self-similarity zeroed to prevent self-loops)

        Parameters
        ----------
        item_factors : np.ndarray
            Matrix of shape (num_items, latent_dim).

        Returns
        -------
        np.ndarray
            Pairwise cosine similarity matrix S of shape (num_items, num_items) with
            values in [-1.0, 1.0] and diagonal set to 0.0.
        """
        V = np.asarray(item_factors, dtype=np.float32)
        if V.shape != (self.num_items, self.latent_dim):
            raise ValueError(
                f"Invalid item_factors shape {V.shape}; expected ({self.num_items}, {self.latent_dim})"
            )

        # L2 row normalization with numerical stabilization
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        # Avoid division by zero for unobserved/zero vectors
        norms = np.where(norms < 1e-9, 1e-9, norms)
        V_norm = V / norms

        # Pairwise dot products on unit vectors give cosine similarities
        S = np.dot(V_norm, V_norm.T)

        # Zero out diagonal to avoid self-loops in k-NN graph
        np.fill_diagonal(S, 0.0)

        # Clip numerical precision artifacts to strictly [-1.0, 1.0]
        np.clip(S, -1.0, 1.0, out=S)

        return S

    # --------------------------------------------------------------------------
    # k-NN Graph Construction
    # --------------------------------------------------------------------------

    def build_knn_graph(
        self,
        similarity_matrix: np.ndarray,
    ) -> Tuple[ig.Graph, int]:
        """
        Construct a symmetric weighted k-NN graph from the cosine similarity matrix.

        Graph Construction Rules:
        - For each item i, find top-k items j != i with S_ij > 0.
        - Add undirected edge between i and j if j in k-NN(i) or i in k-NN(j).
        - Edge weight W_ij = max(S_ij, S_ji) > 0.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Cosine similarity matrix S of shape (num_items, num_items).

        Returns
        -------
        Tuple[ig.Graph, int]
            - graph: igraph.Graph instance with num_items vertices and 'weight' edge attribute.
            - edge_count: Total number of undirected edges.
        """
        S = np.asarray(similarity_matrix, dtype=np.float32)
        if S.shape != (self.num_items, self.num_items):
            raise ValueError(
                f"Invalid similarity_matrix shape {S.shape}; expected ({self.num_items}, {self.num_items})"
            )

        k = min(self.k_neighbors, self.num_items - 1)
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        seen_edges: Set[Tuple[int, int]] = set()

        for i in range(self.num_items):
            row = S[i]
            # Identify top-k candidate neighbors using partial sorting
            top_candidates = np.argpartition(row, -k)[-k:]
            # Sort top-k descending
            top_candidates = top_candidates[np.argsort(row[top_candidates])[::-1]]

            for j in top_candidates:
                # Require strictly positive similarity
                if row[j] > 0.0 and i != j:
                    u = min(i, int(j))
                    v = max(i, int(j))
                    edge = (u, v)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        weight = float(max(S[u, v], S[v, u]))
                        edges.append(edge)
                        weights.append(weight)

        # Construct igraph instance with all items guaranteed present
        graph = ig.Graph(n=self.num_items, edges=edges, directed=False)
        graph.es["weight"] = weights

        logger.debug(
            "Constructed k-NN graph: %d vertices, %d edges (k=%d)",
            graph.vcount(),
            graph.ecount(),
            self.k_neighbors,
        )

        return graph, len(edges)

    # --------------------------------------------------------------------------
    # Community Centroid Calculation
    # --------------------------------------------------------------------------

    def compute_centroids(
        self,
        item_factors: np.ndarray,
        community_map: Dict[int, int],
    ) -> Dict[int, np.ndarray]:
        """
        Calculate the community centroid vector mu_c for each detected community.

        Formula:
            mu_c = (1 / |C_c|) * sum_{j in C_c} v_j in R^d

        Parameters
        ----------
        item_factors : np.ndarray
            Matrix of item latent factors of shape (num_items, latent_dim).
        community_map : Dict[int, int]
            Mapping from item_id to community_id {i: c}.

        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping community_id -> centroid vector (dtype=float32, shape=(latent_dim,)).
        """
        V = np.asarray(item_factors, dtype=np.float32)
        centroids: Dict[int, np.ndarray] = {}

        # Group items by community
        community_items: Dict[int, List[int]] = {}
        for item_id, comm_id in community_map.items():
            community_items.setdefault(comm_id, []).append(item_id)

        for comm_id, items in community_items.items():
            if len(items) == 0:
                continue
            item_vectors = V[items]
            mu_c = np.mean(item_vectors, axis=0).astype(np.float32)
            centroids[comm_id] = mu_c

        return centroids

    # --------------------------------------------------------------------------
    # Complete Leiden Clustering Execution
    # --------------------------------------------------------------------------

    def detect_communities(
        self,
        item_factors: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute full item-space Leiden community detection pipeline.

        Steps:
        1. Compute cosine similarity matrix S (1682 x 1682).
        2. Build k-NN graph (k=20, positive symmetric weights).
        3. Run ModularityVertexPartition Leiden community detection.
        4. Validate partition: all items assigned, non-empty partition.
        5. Compute community centroids mu_c.

        Parameters
        ----------
        item_factors : np.ndarray
            Current aligned item factors matrix of shape (num_items, latent_dim).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'community_map': Dict[int, int] mapping item_id -> community_id
            - 'community_map_str': Dict[str, int] string-keyed mapping for JSON
            - 'centroids': Dict[int, np.ndarray] mapping community_id -> mu_c
            - 'centroids_json': Dict[str, List[float]] serialized centroids for JSON
            - 'num_communities': int total count of detected communities
            - 'modularity': float modularity score of the partition
            - 'community_sizes': Dict[int, int] size of each community
        """
        V = np.asarray(item_factors, dtype=np.float32)
        if V.shape != (self.num_items, self.latent_dim):
            raise ValueError(
                f"Invalid item_factors shape {V.shape}; expected ({self.num_items}, {self.latent_dim})"
            )

        # Step 1: Pairwise item cosine similarity
        S = self.compute_cosine_similarity(V)

        # Step 2: k-NN graph
        graph, num_edges = self.build_knn_graph(S)

        # Step 3: Leiden community detection with ModularityVertexPartition
        partition = leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=self.seed,
        )

        membership = list(partition.membership)

        # Step 4: Partition validation
        if len(membership) != self.num_items:
            raise ValueError(
                f"Partition length mismatch: got {len(membership)}, expected {self.num_items}"
            )

        unique_communities = sorted(list(set(membership)))
        num_communities = len(unique_communities)

        if num_communities < 1:
            raise ValueError("Leiden detection produced 0 communities.")

        # Build community mapping
        community_map: Dict[int, int] = {i: int(comm) for i, comm in enumerate(membership)}

        # Community sizes
        community_sizes: Dict[int, int] = {}
        for comm in membership:
            community_sizes[comm] = community_sizes.get(comm, 0) + 1

        # Step 5: Compute centroids
        centroids = self.compute_centroids(V, community_map)

        modularity_score = float(partition.modularity)

        logger.info(
            "Leiden partition complete: %d communities across %d items (Modularity: %.4f)",
            num_communities,
            self.num_items,
            modularity_score,
        )

        return {
            "community_map": community_map,
            "community_map_str": {str(k): v for k, v in community_map.items()},
            "centroids": centroids,
            "centroids_json": {str(k): v.tolist() for k, v in centroids.items()},
            "num_communities": num_communities,
            "modularity": modularity_score,
            "community_sizes": community_sizes,
        }
