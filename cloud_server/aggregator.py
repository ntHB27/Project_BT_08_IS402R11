"""
Federated Aggregator for Fed-LCCF Cloud Server.

Implements the complete 3-phase federated aggregation lifecycle across 20 communication rounds:
1. Phase 1 (Rounds 1–5): Count-weighted Vanilla FedAvg:
     V_i^(t) = (sum_{k=1}^K n_{k,i} * V_{k,i}) / (sum_{k=1}^K n_{k,i})
   For unobserved items across all clients (N_i == 0): V_i^(t) = V_i^(t-1).

2. Phase 2 (Post-Round 5 Transition):
   Triggered immediately upon completion of Round 5 aggregation:
   Executes item-space Leiden community detection on V^(5) via LeidenClusteringEngine,
   extracts community map C(i) and community centroids mu_c, and transitions to Phase 3.

3. Phase 3 (Rounds 6–20): Community Shrinkage Aggregation:
   Applies shrinkage with parameter gamma = 10.0:
     v_i^(t) = [N_i / (N_i + gamma)] * v_i_raw + [gamma / (N_i + gamma)] * mu_{C(i)}
   For unobserved items (N_i == 0): v_i^(t) = mu_{C(i)}.
   Dynamically updates community centroids mu_c after each round.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cloud_server.leiden_engine import (
    DEFAULT_K_NEIGHBORS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_ITEMS,
    DEFAULT_RANDOM_SEED,
    LeidenClusteringEngine,
)

logger = logging.getLogger("fedlccf.aggregator")

DEFAULT_GAMMA = 10.0
PHASE_WARMUP = 1            # Rounds 1-5 (Vanilla FedAvg)
PHASE_LEIDEN_TRANSITION = 2 # Post-Round 5 (Leiden Clustering)
PHASE_SHRINKAGE = 3         # Rounds 6-20 (Community Shrinkage)


class FederatedAggregator:
    """
    Core federated aggregation engine managing factors, phases, and community shrinkage.

    Parameters
    ----------
    num_items : int
        Number of items in catalog (default: 1682).
    latent_dim : int
        Latent embedding dimension d (default: 32).
    gamma : float
        Shrinkage parameter gamma for Phase 3 (default: 10.0).
    random_state : int
        Seed for reproducible seed factors and clustering (default: 42).
    leiden_k : int
        k-NN parameter for Leiden clustering (default: 20).
    """

    def __init__(
        self,
        num_items: int = DEFAULT_NUM_ITEMS,
        latent_dim: int = DEFAULT_LATENT_DIM,
        gamma: float = DEFAULT_GAMMA,
        random_state: int = DEFAULT_RANDOM_SEED,
        leiden_k: int = DEFAULT_K_NEIGHBORS,
    ) -> None:
        if num_items <= 0:
            raise ValueError(f"num_items must be positive; got {num_items}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative; got {gamma}")

        self.num_items = int(num_items)
        self.latent_dim = int(latent_dim)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
        self.leiden_k = int(leiden_k)

        # Thread synchronization lock
        self._lock = threading.RLock()

        # Clustering engine
        self.leiden_engine = LeidenClusteringEngine(
            k_neighbors=self.leiden_k,
            seed=self.random_state,
            num_items=self.num_items,
            latent_dim=self.latent_dim,
        )

        # Seed global factors for Round 0: reproducible Gaussian N(0, 1/sqrt(d))
        rng = np.random.RandomState(self.random_state)
        scale = 1.0 / np.sqrt(self.latent_dim)
        self._initial_seed_factors = rng.normal(0.0, scale, (self.num_items, self.latent_dim)).astype(np.float32)

        # Active state
        self._global_factors = self._initial_seed_factors.copy()
        self._current_round = 0
        self._current_phase = PHASE_WARMUP

        # History: round_num -> np.ndarray
        self._factors_history: Dict[int, np.ndarray] = {0: self._initial_seed_factors.copy()}

        # Community artifacts generated after Round 5
        self._community_map: Optional[Dict[int, int]] = None
        self._centroids: Optional[Dict[int, np.ndarray]] = None
        self._num_communities: Optional[int] = None
        self._clustering_metadata: Optional[Dict[str, Any]] = None

    # --------------------------------------------------------------------------
    # Factor Retrieval & State Inspection
    # --------------------------------------------------------------------------

    @property
    def current_round(self) -> int:
        """Current highest aggregated round."""
        with self._lock:
            return self._current_round

    @property
    def current_phase(self) -> int:
        """Current protocol phase (1: Warmup, 2: Leiden Transition, 3: Shrinkage)."""
        with self._lock:
            return self._current_phase

    @property
    def community_map(self) -> Optional[Dict[int, int]]:
        """Community map mapping item_id -> community_id, or None if not yet computed."""
        with self._lock:
            return self._community_map

    @property
    def centroids(self) -> Optional[Dict[int, np.ndarray]]:
        """Community centroids mapping community_id -> mu_c, or None if not yet computed."""
        with self._lock:
            return self._centroids

    @property
    def num_communities(self) -> Optional[int]:
        """Total number of detected Leiden communities."""
        with self._lock:
            return self._num_communities

    def get_global_factors(self, round_num: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a copy of global aggregated item factors for round_num (or current round).

        Parameters
        ----------
        round_num : Optional[int]
            Round index, or None for current factors.

        Returns
        -------
        Optional[np.ndarray]
            Copy of factors matrix of shape (num_items, latent_dim), or None if round not found.
        """
        with self._lock:
            if round_num is None:
                return self._global_factors.copy()
            target = int(round_num)
            factors = self._factors_history.get(target)
            return factors.copy() if factors is not None else None

    def has_round(self, round_num: int) -> bool:
        """Check whether factors exist for round_num."""
        with self._lock:
            return int(round_num) in self._factors_history

    def get_phase_for_round(self, round_num: int) -> int:
        """
        Determine canonical phase identifier for round_num.

        - Round 0: 1 (Warmup seed)
        - Rounds 1-5: 1 (Vanilla FedAvg)
        - Post-Round 5 / Round 5 output: 2 (Leiden Community artifacts available)
        - Rounds 6-20: 3 (Community Shrinkage)
        """
        with self._lock:
            r = int(round_num)
            if r <= 4:
                return PHASE_WARMUP
            elif r == 5:
                # If clustering has run for round 5, return 2 (Leiden Transition complete)
                return PHASE_LEIDEN_TRANSITION if self._community_map is not None else PHASE_WARMUP
            else:
                return PHASE_SHRINKAGE

    def get_factors_payload_dict(self, round_num: int) -> Optional[Dict[str, Any]]:
        """
        Build the dictionary structure for HTTP GET /api/v1/fl/get_global_factors/{round}.

        Returns
        -------
        Optional[Dict[str, Any]]
            Payload containing global factors and community metadata, or None if round unavailable.
        """
        with self._lock:
            r = int(round_num)
            if r not in self._factors_history:
                return None

            factors = self._factors_history[r]
            v_list = factors.tolist()
            phase = self.get_phase_for_round(r)

            payload: Dict[str, Any] = {
                "round": r,
                "phase": phase,
                "global_factors": v_list,
                "factors": v_list,
            }

            # Include community information once generated (Round 5 onwards)
            if self._community_map is not None and self._centroids is not None and r >= 5:
                payload["community_map"] = {str(k): int(v) for k, v in self._community_map.items()}
                payload["centroids"] = {str(k): v.tolist() for k, v in self._centroids.items()}
                payload["num_communities"] = self._num_communities

            return payload

    # --------------------------------------------------------------------------
    # Aggregation Execution
    # --------------------------------------------------------------------------

    def _parse_client_upload(
        self,
        client_id: int,
        upload_data: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract item factors and interaction counts from a client upload object.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - factors: np.ndarray of shape (num_items, latent_dim), dtype=float32
            - counts: np.ndarray of shape (num_items,), dtype=float32
        """
        # Handle dict or Pydantic model
        if hasattr(upload_data, "model_dump"):
            data = upload_data.model_dump()
        elif hasattr(upload_data, "dict"):
            data = upload_data.dict()
        elif isinstance(upload_data, dict):
            data = upload_data
        else:
            raise TypeError(f"Unsupported upload data type: {type(upload_data)}")

        # Extract factors with alias fallback
        raw_factors = data.get("item_factors")
        if raw_factors is None:
            raw_factors = data.get("factors")
        if raw_factors is None:
            raise KeyError(f"Client {client_id} upload missing 'item_factors' or 'factors'")

        # Extract counts with alias fallback
        raw_counts = data.get("item_counts")
        if raw_counts is None:
            raw_counts = data.get("counts")
        if raw_counts is None:
            raise KeyError(f"Client {client_id} upload missing 'item_counts' or 'counts'")

        factors = np.asarray(raw_factors, dtype=np.float32)
        counts = np.asarray(raw_counts, dtype=np.float32)

        if factors.shape != (self.num_items, self.latent_dim):
            raise ValueError(
                f"Client {client_id} invalid factor shape {factors.shape}; expected ({self.num_items}, {self.latent_dim})"
            )
        if counts.shape != (self.num_items,):
            raise ValueError(
                f"Client {client_id} invalid counts shape {counts.shape}; expected ({self.num_items},)"
            )

        return factors, counts

    def aggregate_round(
        self,
        round_num: int,
        client_uploads: Dict[int, Any],
    ) -> np.ndarray:
        """
        Aggregate client item factor uploads for round_num and update server state.

        Parameters
        ----------
        round_num : int
            Communication round being aggregated (1..20).
        client_uploads : Dict[int, Any]
            Dictionary of client uploads {client_id: upload_payload}.

        Returns
        -------
        np.ndarray
            New aggregated global item factors V^(t) of shape (num_items, latent_dim).
        """
        with self._lock:
            r = int(round_num)
            if len(client_uploads) == 0:
                raise ValueError(f"Cannot aggregate round {r} with 0 client uploads")

            # Parse factors and counts from all participating clients
            factors_list: List[np.ndarray] = []
            counts_list: List[np.ndarray] = []

            for cid in sorted(client_uploads.keys()):
                f, c = self._parse_client_upload(cid, client_uploads[cid])
                factors_list.append(f)
                counts_list.append(c)

            K = len(factors_list)
            all_factors = np.stack(factors_list, axis=0)  # Shape: (K, num_items, latent_dim)
            all_counts = np.stack(counts_list, axis=0)    # Shape: (K, num_items)

            # Total interactions per item across all clients
            total_counts = np.sum(all_counts, axis=0)      # Shape: (num_items,)
            observed_mask = total_counts > 0.0

            # Previous round factors
            prev_factors = self._global_factors.copy()

            # Compute empirical weighted average: sum(n_{k,i} * V_{k,i}) / sum(n_{k,i})
            # Weighted sum: shape (num_items, latent_dim)
            weighted_sum = np.sum(all_counts[:, :, None] * all_factors, axis=0)

            raw_empirical_mean = np.zeros_like(prev_factors)
            raw_empirical_mean[observed_mask] = weighted_sum[observed_mask] / total_counts[observed_mask, None]

            # ----------------------------------------------------------------------
            # Phase 1: Warmup Vanilla FedAvg (Rounds 1-5)
            # ----------------------------------------------------------------------
            if r <= 5:
                self._current_phase = PHASE_WARMUP
                new_factors = np.zeros_like(prev_factors)
                # For observed items: count-weighted average
                new_factors[observed_mask] = raw_empirical_mean[observed_mask]
                # For unobserved items across all clients: retain previous round factors
                new_factors[~observed_mask] = prev_factors[~observed_mask]

                self._global_factors = new_factors.astype(np.float32)

                # ------------------------------------------------------------------
                # Phase 2 Transition Post-Round 5: Leiden Clustering
                # ------------------------------------------------------------------
                if r == 5:
                    logger.info("Triggering Phase 2 Leiden Community Detection on Round 5 factors...")
                    clustering_res = self.leiden_engine.detect_communities(self._global_factors)
                    self._community_map = clustering_res["community_map"]
                    self._centroids = clustering_res["centroids"]
                    self._num_communities = clustering_res["num_communities"]
                    self._clustering_metadata = clustering_res
                    self._current_phase = PHASE_LEIDEN_TRANSITION
                    logger.info(
                        "Phase 2 Complete: Discovered %d item communities. Ready for Phase 3 shrinkage.",
                        self._num_communities,
                    )

                # ATOMIC COMMIT: Publish to history only AFTER community artifacts are fully generated!
                self._factors_history[r] = self._global_factors.copy()
                self._current_round = r

                logger.info(
                    "Aggregated Round %d [%s] across %d clients. Observed items: %d/%d",
                    r,
                    "Phase 2 Leiden" if r == 5 else "Phase 1 FedAvg",
                    K,
                    int(np.sum(observed_mask)),
                    self.num_items,
                )

                return self._global_factors.copy()

            # ----------------------------------------------------------------------
            # Phase 3: Community Shrinkage Aggregation (Rounds 6-20)
            # ----------------------------------------------------------------------
            else:
                self._current_phase = PHASE_SHRINKAGE

                # Fallback if clustering was not executed previously
                if self._community_map is None or self._centroids is None:
                    logger.warning("Phase 3 reached without Leiden clustering; executing now...")
                    clustering_res = self.leiden_engine.detect_communities(prev_factors)
                    self._community_map = clustering_res["community_map"]
                    self._centroids = clustering_res["centroids"]
                    self._num_communities = clustering_res["num_communities"]
                    self._clustering_metadata = clustering_res

                # Build item centroid matrix from current centroids
                centroid_matrix = np.zeros((self.num_items, self.latent_dim), dtype=np.float32)
                for i in range(self.num_items):
                    comm = self._community_map.get(i, 0)
                    centroid_matrix[i] = self._centroids[comm]

                # Vectorized Community Shrinkage calculation:
                # v_i = [N_i / (N_i + gamma)] * v_i_raw + [gamma / (N_i + gamma)] * mu_{C(i)}
                # If N_i == 0: v_i = mu_{C(i)}
                N_i = total_counts[:, None]  # Shape: (num_items, 1)
                denom = N_i + self.gamma
                valid_mask = denom > 0.0
                weights_raw = np.zeros_like(N_i)
                weights_centroid = np.zeros_like(N_i)
                weights_raw[valid_mask] = N_i[valid_mask] / denom[valid_mask]
                weights_centroid[valid_mask] = self.gamma / denom[valid_mask]

                # For unobserved items, raw_empirical_mean is replaced with centroid to avoid zeros
                item_raw_vectors = np.where(observed_mask[:, None], raw_empirical_mean, centroid_matrix)

                new_factors = weights_raw * item_raw_vectors + weights_centroid * centroid_matrix
                if not np.all(valid_mask):
                    unobserved_global = ~valid_mask.squeeze()
                    new_factors[unobserved_global] = prev_factors[unobserved_global]

                self._global_factors = new_factors.astype(np.float32)

                # Dynamic centroid update for round t BEFORE publishing history
                self._centroids = self.leiden_engine.compute_centroids(
                    self._global_factors,
                    self._community_map,
                )

                # ATOMIC COMMIT: Publish to history only AFTER updated centroids are ready!
                self._factors_history[r] = self._global_factors.copy()
                self._current_round = r

                logger.info(
                    "Aggregated Round %d [Phase 3 Shrinkage gamma=%.1f] across %d clients. Updated %d centroids.",
                    r,
                    self.gamma,
                    K,
                    len(self._centroids),
                )

                return self._global_factors.copy()

    # --------------------------------------------------------------------------
    # Reset & Testing Helpers
    # --------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset aggregator to initial Round 0 seed state."""
        with self._lock:
            self._global_factors = self._initial_seed_factors.copy()
            self._current_round = 0
            self._current_phase = PHASE_WARMUP
            self._factors_history = {0: self._initial_seed_factors.copy()}
            self._community_map = None
            self._centroids = None
            self._num_communities = None
            self._clustering_metadata = None
