"""
Local Matrix Factorization (MF) Subsystem for Fed-LCCF Clients.

Implements on-device Matrix Factorization (R ≈ U · V^T) with:
- Latent factor dimensionality d=32.
- Pointwise Stochastic Gradient Descent (SGD) with L2 regularization.
- Client-specific adaptive regularization (lambda_k) based on partition density:
    Client 1 (dense 45%): lambda_1 = 0.01
    Client 2 (30%):       lambda_2 = 0.02
    Client 3 (18%):       lambda_3 = 0.05
    Client 4 (sparse 7%): lambda_4 = 0.10
- Strict User Factor Privacy: user factors U are stored exclusively on-device
  and NEVER exported or transmitted across the network.
- Phase 1 Unobserved Items Initialization: unobserved items (n_{k,i} == 0)
  are initialized to the mean vector of observed items.
- Phase 3 Leiden Centroid Anchoring: unobserved items are anchored to their
  Leiden community centroid mu_{C(i)}, and observed items incur an anchoring
  loss penalty lambda_anchor * ||v_i - mu_{C(i)}||^2 (lambda_anchor = 0.02).
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

DEFAULT_NUM_USERS = 943
DEFAULT_NUM_ITEMS = 1682
DEFAULT_LATENT_DIM = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LAMBDA_ANCHOR = 0.02

# Adaptive regularization mapping per client density
ADAPTIVE_LAMBDA_MAP: Dict[int, float] = {
    1: 0.01,  # Client 1: 45% ratings (dense)
    2: 0.02,  # Client 2: 30% ratings
    3: 0.05,  # Client 3: 18% ratings
    4: 0.10,  # Client 4: 7% ratings (sparse, cold-start heavy)
}


class MatrixFactorization:
    """
    On-device Matrix Factorization model (R ≈ U · V^T).

    Parameters
    ----------
    num_users : int
        Number of users in global catalog (default: 943).
    num_items : int
        Number of items in global catalog (default: 1682).
    latent_dim : int
        Latent embedding dimension d (default: 32).
    learning_rate : float
        SGD learning rate eta (default: 0.01).
    reg_u : Optional[float]
        L2 regularization parameter for user latent factors.
        If None, automatically inferred from client_id adaptive mapping or defaults to 0.02.
    reg_v : Optional[float]
        L2 regularization parameter for item latent factors.
        If None, automatically inferred from client_id adaptive mapping or defaults to 0.02.
    client_id : Optional[int]
        Client identifier (1..4) used for adaptive lambda_k lookup.
    lambda_anchor : float
        Centroid anchoring regularization penalty parameter for Phase 3 (default: 0.02).
    random_state : Optional[int]
        Random seed for reproducibility.
    init_scale : float
        Standard deviation for Gaussian initialization N(0, init_scale^2) (default: 0.01).
    """

    def __init__(
        self,
        num_users: int = DEFAULT_NUM_USERS,
        num_items: int = DEFAULT_NUM_ITEMS,
        latent_dim: int = DEFAULT_LATENT_DIM,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        reg_u: Optional[float] = None,
        reg_v: Optional[float] = None,
        client_id: Optional[int] = None,
        lambda_anchor: float = DEFAULT_LAMBDA_ANCHOR,
        random_state: Optional[int] = None,
        init_scale: float = 0.01,
    ) -> None:
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.client_id = client_id
        self.lambda_anchor = float(lambda_anchor)
        self.random_state = random_state
        self.init_scale = float(init_scale)

        # Resolve adaptive regularization parameters
        default_reg = 0.02
        if client_id is not None and client_id in ADAPTIVE_LAMBDA_MAP:
            default_reg = ADAPTIVE_LAMBDA_MAP[client_id]

        self.reg_u = float(reg_u) if reg_u is not None else default_reg
        self.reg_v = float(reg_v) if reg_v is not None else default_reg

        self.rng = np.random.default_rng(random_state)

        # Internal private factor matrices
        # U is strictly private: never exported, serialized in upload payload, or transmitted.
        self._user_factors: np.ndarray = self.rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.num_users, self.latent_dim),
        ).astype(np.float32)

        # V is public to the federated aggregation protocol
        self._item_factors: np.ndarray = self.rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.num_items, self.latent_dim),
        ).astype(np.float32)

        # Item interaction counts n_{k, i}
        self._interaction_counts: np.ndarray = np.zeros(self.num_items, dtype=np.uint32)

        # Phase 3 Leiden Community structures
        self._community_map: Optional[Dict[int, int]] = None
        self._community_centroids: Optional[Dict[int, np.ndarray]] = None
        self._use_community_anchoring: bool = False
        self._anchoring_targets: np.ndarray = np.zeros((self.num_items, self.latent_dim), dtype=np.float32)
        self._has_anchor: np.ndarray = np.zeros(self.num_items, dtype=bool)

        # Training telemetry
        self._num_samples: int = 0
        self._latest_loss: float = 0.0
        self._loss_history: List[float] = []

    # --------------------------------------------------------------------------
    # Dimensions & Properties
    # --------------------------------------------------------------------------

    @property
    def user_factors_shape(self) -> Tuple[int, int]:
        """Return the dimensions of the private user factor matrix U."""
        return self._user_factors.shape

    @property
    def item_factors_shape(self) -> Tuple[int, int]:
        """Return the dimensions of the item factor matrix V."""
        return self._item_factors.shape

    @property
    def latest_train_loss(self) -> float:
        """Return the most recent training MSE loss."""
        return self._latest_loss

    @property
    def loss_history(self) -> List[float]:
        """Return the recorded history of training epoch losses."""
        return list(self._loss_history)

    def get_user_factors_for_testing(self) -> np.ndarray:
        """
        Return a copy of the private user factors matrix U.

        NOTE: This accessor exists STRICTLY for local white-box unit testing
        and local persistence. It is never invoked by communication handlers
        or export methods.
        """
        return self._user_factors.copy()

    def get_item_factors(self) -> np.ndarray:
        """Return a copy of the item factor matrix V."""
        return self._item_factors.copy()

    def set_item_factors(self, item_factors: np.ndarray) -> None:
        """
        Update the local item factor matrix V with global aggregated factors.

        Parameters
        ----------
        item_factors : np.ndarray
            Item factors array of shape (num_items, latent_dim).
        """
        factors = np.asarray(item_factors, dtype=np.float32)
        if factors.shape != (self.num_items, self.latent_dim):
            raise ValueError(
                f"Invalid item factors shape {factors.shape}; expected ({self.num_items}, {self.latent_dim})"
            )
        self._item_factors = factors.copy()

    def set_interaction_counts(self, counts: np.ndarray) -> None:
        """
        Set item interaction count vector n_k.

        Parameters
        ----------
        counts : np.ndarray
            1D array of shape (num_items,) indicating ratings frequency per item.
        """
        c = np.asarray(counts, dtype=np.uint32)
        if c.shape != (self.num_items,):
            raise ValueError(f"Invalid interaction counts shape {c.shape}; expected ({self.num_items},)")
        self._interaction_counts = c.copy()

    def get_interaction_counts(self) -> np.ndarray:
        """Return a copy of the interaction counts vector n_k."""
        return self._interaction_counts.copy()

    # --------------------------------------------------------------------------
    # Phase 1: Unobserved Item Initialization
    # --------------------------------------------------------------------------

    def initialize_unobserved_items_mean(
        self,
        observed_indices: Optional[Iterable[int]] = None,
        missing_indices: Optional[Iterable[int]] = None,
        interaction_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize unobserved items (n_{k,i} == 0) to the observed items mean vector.

        Formula:
            v_i = (1 / |I_obs|) * sum_{j in I_obs} v_j, for all i in I_unobs.

        Parameters
        ----------
        observed_indices : Optional[Iterable[int]]
            Indices of items observed in local training set.
        missing_indices : Optional[Iterable[int]]
            Indices of items unobserved in local training set.
        interaction_counts : Optional[np.ndarray]
            Array of shape (num_items,) with interaction counts.

        Returns
        -------
        np.ndarray
            The computed mean vector of observed items.
        """
        if interaction_counts is not None:
            self.set_interaction_counts(interaction_counts)

        if observed_indices is not None and missing_indices is not None:
            obs = np.asarray(list(observed_indices), dtype=np.int32)
            miss = np.asarray(list(missing_indices), dtype=np.int32)
        else:
            obs = np.where(self._interaction_counts > 0)[0]
            miss = np.where(self._interaction_counts == 0)[0]

        if len(obs) == 0:
            # Fallback: compute mean across all items if none observed
            mean_vec = np.mean(self._item_factors, axis=0)
        else:
            mean_vec = np.mean(self._item_factors[obs], axis=0)

        if len(miss) > 0:
            self._item_factors[miss] = mean_vec.copy()

        return mean_vec

    # --------------------------------------------------------------------------
    # Phase 3: Leiden Community Anchoring
    # --------------------------------------------------------------------------

    def set_community_info(
        self,
        community_map: Dict[Union[int, str], int],
        centroids: Dict[Union[int, str], Union[np.ndarray, List[float]]],
        activate_anchoring: bool = True,
    ) -> None:
        """
        Register Leiden community partition and centroids for Phase 3 anchoring.

        Parameters
        ----------
        community_map : Dict[Union[int, str], int]
            Mapping from item_id to community_id {i: C(i)}.
        centroids : Dict[Union[int, str], Union[np.ndarray, List[float]]]
            Mapping from community_id to centroid vector mu_c.
        activate_anchoring : bool
            Whether to activate centroid anchoring during local SGD training.
        """
        self._community_map = {int(k): int(v) for k, v in community_map.items()}
        self._community_centroids = {
            int(k): np.asarray(v, dtype=np.float32) for k, v in centroids.items()
        }
        self._use_community_anchoring = activate_anchoring

        # Precompute dense lookup table for fast SGD execution without dictionary overhead
        self._anchoring_targets.fill(0.0)
        self._has_anchor.fill(False)

        for i in range(self.num_items):
            if i in self._community_map:
                c = self._community_map[i]
                if c in self._community_centroids:
                    self._anchoring_targets[i] = self._community_centroids[c]
                    self._has_anchor[i] = True

    def anchor_unobserved_items_to_centroids(
        self,
        missing_indices: Optional[Iterable[int]] = None,
        interaction_counts: Optional[np.ndarray] = None,
    ) -> int:
        """
        Anchor unobserved items (n_{k,i} == 0) to their Leiden community centroid mu_{C(i)}.

        Parameters
        ----------
        missing_indices : Optional[Iterable[int]]
            Indices of items unobserved by this client.
        interaction_counts : Optional[np.ndarray]
            Counts array if updating counts.

        Returns
        -------
        int
            Number of unobserved items successfully anchored.
        """
        if interaction_counts is not None:
            self.set_interaction_counts(interaction_counts)

        if missing_indices is not None:
            miss = np.asarray(list(missing_indices), dtype=np.int32)
        else:
            miss = np.where(self._interaction_counts == 0)[0]

        anchored_count = 0
        for i in miss:
            if 0 <= i < self.num_items and self._has_anchor[i]:
                self._item_factors[i] = self._anchoring_targets[i].copy()
                anchored_count += 1

        return anchored_count

    # --------------------------------------------------------------------------
    # Model Training: Pointwise SGD
    # --------------------------------------------------------------------------

    def fit(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 5,
        shuffle: bool = True,
        interaction_counts: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Train local Matrix Factorization on observed client ratings using pointwise SGD.

        Pointwise updates:
            e_{ui} = r_{ui} - u_u * v_i^T
            u_u <- u_u + eta * (e_{ui} * v_i - lambda_U * u_u)
            v_i <- v_i + eta * (e_{ui} * u_u - lambda_V * v_i [- lambda_anchor * (v_i - mu_{C(i)})])

        Parameters
        ----------
        users : np.ndarray
            1D array of user IDs (int32).
        items : np.ndarray
            1D array of item IDs (int32).
        ratings : np.ndarray
            1D array of ratings (float32).
        epochs : int
            Number of local training epochs (default: 5).
        shuffle : bool
            Whether to shuffle training samples each epoch (default: True).
        interaction_counts : Optional[np.ndarray]
            Array of shape (num_items,) with interaction frequencies.

        Returns
        -------
        List[float]
            List of training MSE losses recorded after each epoch.
        """
        u_arr = np.asarray(users, dtype=np.int32)
        i_arr = np.asarray(items, dtype=np.int32)
        r_arr = np.asarray(ratings, dtype=np.float32)

        n_samples = len(r_arr)
        if n_samples == 0:
            return []

        self._num_samples = n_samples

        if interaction_counts is not None:
            self.set_interaction_counts(interaction_counts)
        elif self._interaction_counts.sum() == 0:
            # Auto-compute counts from training items
            counts = np.zeros(self.num_items, dtype=np.uint32)
            np.add.at(counts, i_arr, 1)
            self._interaction_counts = counts

        # Local variable references for execution speed
        U = self._user_factors
        V = self._item_factors
        lr = np.float32(self.learning_rate)
        reg_u = np.float32(self.reg_u)
        reg_v = np.float32(self.reg_v)
        l_anchor = np.float32(self.lambda_anchor)
        use_anchor = self._use_community_anchoring
        has_anchor = self._has_anchor
        anchoring_targets = self._anchoring_targets

        epoch_losses: List[float] = []

        for _ in range(epochs):
            if shuffle:
                perm = self.rng.permutation(n_samples)
                cur_users = u_arr[perm]
                cur_items = i_arr[perm]
                cur_ratings = r_arr[perm]
            else:
                cur_users = u_arr
                cur_items = i_arr
                cur_ratings = r_arr

            # Inner Pointwise SGD loop
            if use_anchor:
                for idx in range(n_samples):
                    u = cur_users[idx]
                    i = cur_items[idx]
                    r = cur_ratings[idx]

                    u_vec = U[u]
                    v_vec = V[i]

                    # Prediction error
                    err = r - np.dot(u_vec, v_vec)

                    # Compute gradients before mutating vectors
                    grad_u = err * v_vec - reg_u * u_vec
                    if has_anchor[i]:
                        grad_v = err * u_vec - reg_v * v_vec - l_anchor * (v_vec - anchoring_targets[i])
                    else:
                        grad_v = err * u_vec - reg_v * v_vec

                    # In-place SGD step
                    u_vec += lr * grad_u
                    v_vec += lr * grad_v

                # Maintain unobserved items strictly pinned to community centroids
                self.anchor_unobserved_items_to_centroids()

            else:
                for idx in range(n_samples):
                    u = cur_users[idx]
                    i = cur_items[idx]
                    r = cur_ratings[idx]

                    u_vec = U[u]
                    v_vec = V[i]

                    # Prediction error
                    err = r - np.dot(u_vec, v_vec)

                    # Compute gradients
                    grad_u = err * v_vec - reg_u * u_vec
                    grad_v = err * u_vec - reg_v * v_vec

                    # In-place SGD step
                    u_vec += lr * grad_u
                    v_vec += lr * grad_v

            # Post-epoch MSE loss computation (vectorized)
            preds = np.sum(U[u_arr] * V[i_arr], axis=1)
            epoch_loss = float(np.mean((r_arr - preds) ** 2))
            epoch_losses.append(epoch_loss)

        self._latest_loss = epoch_losses[-1] if epoch_losses else 0.0
        self._loss_history.extend(epoch_losses)

        return epoch_losses

    # --------------------------------------------------------------------------
    # Prediction & Evaluation
    # --------------------------------------------------------------------------

    def predict(
        self,
        user_id: int,
        item_id: int,
        clip_range: Optional[Tuple[float, float]] = (1.0, 5.0),
    ) -> float:
        """
        Predict rating for a single user-item pair.

        Parameters
        ----------
        user_id : int
            0-based user index in [0, num_users - 1].
        item_id : int
            0-based item index in [0, num_items - 1].
        clip_range : Optional[Tuple[float, float]]
            Optional bounds (min, max) to clip prediction.

        Returns
        -------
        float
            Predicted rating.
        """
        dot = float(np.dot(self._user_factors[user_id], self._item_factors[item_id]))
        if clip_range is not None:
            dot = max(clip_range[0], min(clip_range[1], dot))
        return dot

    def predict_batch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        clip_range: Optional[Tuple[float, float]] = (1.0, 5.0),
    ) -> np.ndarray:
        """
        Vectorized prediction for arrays of user and item indices.

        Parameters
        ----------
        users : np.ndarray
            1D array of user indices.
        items : np.ndarray
            1D array of item indices.
        clip_range : Optional[Tuple[float, float]]
            Optional bounds (min, max) to clip predictions.

        Returns
        -------
        np.ndarray
            1D array of predicted ratings.
        """
        u_arr = np.asarray(users, dtype=np.int32)
        i_arr = np.asarray(items, dtype=np.int32)
        preds = np.sum(self._user_factors[u_arr] * self._item_factors[i_arr], axis=1)
        if clip_range is not None:
            np.clip(preds, clip_range[0], clip_range[1], out=preds)
        return preds

    def evaluate_rmse(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        clip_range: Optional[Tuple[float, float]] = (1.0, 5.0),
    ) -> float:
        """Calculate Root Mean Squared Error (RMSE)."""
        preds = self.predict_batch(users, items, clip_range=clip_range)
        r_arr = np.asarray(ratings, dtype=np.float32)
        return float(np.sqrt(np.mean((r_arr - preds) ** 2)))

    def evaluate_mae(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        clip_range: Optional[Tuple[float, float]] = (1.0, 5.0),
    ) -> float:
        """Calculate Mean Absolute Error (MAE)."""
        preds = self.predict_batch(users, items, clip_range=clip_range)
        r_arr = np.asarray(ratings, dtype=np.float32)
        return float(np.mean(np.abs(r_arr - preds)))

    def compute_loss(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        """Calculate unclipped Mean Squared Error (MSE) loss."""
        preds = self.predict_batch(users, items, clip_range=None)
        r_arr = np.asarray(ratings, dtype=np.float32)
        return float(np.mean((r_arr - preds) ** 2))

    # --------------------------------------------------------------------------
    # Privacy Enforcement & Federated Upload Payload
    # --------------------------------------------------------------------------

    def get_upload_payload(self) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Extract the privacy-preserving client upload payload.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int, float]
            - V_k: Item factors matrix of shape (num_items, latent_dim).
            - n_k: Item interaction count vector of shape (num_items,).
            - num_samples: Number of training samples.
            - train_loss: Most recent training loss.

        PRIVACY GUARANTEE:
        The private user latent factor matrix U is strictly excluded from this payload.
        """
        return (
            self._item_factors.copy(),
            self._interaction_counts.copy(),
            int(self._num_samples),
            float(self._latest_loss),
        )

    def get_upload_payload_dict(
        self,
        round_num: int,
        client_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Construct the serialized JSON payload for HTTP POST /api/v1/fl/upload_factors.

        Parameters
        ----------
        round_num : int
            Current communication round number.
        client_id : Optional[int]
            Client identifier (defaults to self.client_id or 1).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing item factors, counts, and metadata.
            Includes both canonical keys ('item_factors', 'item_counts')
            and standard aliases ('factors', 'counts') for server compatibility.
        """
        cid = client_id if client_id is not None else (self.client_id if self.client_id is not None else 1)
        v_list = self._item_factors.tolist()
        n_list = [int(c) for c in self._interaction_counts.tolist()]

        payload = {
            "client_id": int(cid),
            "round": int(round_num),
            "item_factors": v_list,
            "factors": v_list,
            "item_counts": n_list,
            "counts": n_list,
            "num_samples": int(self._num_samples),
            "local_loss": float(self._latest_loss),
        }

        # Assert privacy invariant
        assert "user_factors" not in payload
        assert "_user_factors" not in payload
        assert "U" not in payload

        return payload

    # --------------------------------------------------------------------------
    # Local Persistence (Client-Side Storage)
    # --------------------------------------------------------------------------

    def save_user_factors(self, filepath: Union[str, Path]) -> None:
        """Save private user factors U to local client disk storage."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self._user_factors)

    def load_user_factors(self, filepath: Union[str, Path]) -> None:
        """Load private user factors U from local client disk storage."""
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"User factors file not found: {path}")
        loaded = np.load(str(path))
        if loaded.shape != (self.num_users, self.latent_dim):
            raise ValueError(
                f"Invalid user factors shape {loaded.shape}; expected ({self.num_users}, {self.latent_dim})"
            )
        self._user_factors = loaded.astype(np.float32).copy()

    def __repr__(self) -> str:
        return (
            f"MatrixFactorization(users={self.num_users}, items={self.num_items}, "
            f"d={self.latent_dim}, client_id={self.client_id}, reg_u={self.reg_u}, "
            f"reg_v={self.reg_v}, anchored={self._use_community_anchoring})"
        )
