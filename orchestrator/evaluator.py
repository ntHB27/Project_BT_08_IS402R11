"""
Evaluation Subsystem for Fed-LCCF Recommendation System.

Provides comprehensive, reproducible test set evaluation on MovieLens 100K:
- Loads the 20,000-rating stratified test set from test.db.
- Partitions test ratings by client ownership (T_k = {(u, i, r) in T | u in U_k}).
- Computes Global RMSE and MAE:
    r_hat_{u,i} = clip(u_u^T v_i, 1.0, 5.0)
    RMSE = sqrt( (1 / |T|) sum_{(u,i,r) in T} (r - r_hat_{u,i})^2 )
    MAE  = (1 / |T|) sum_{(u,i,r) in T} |r - r_hat_{u,i}|
- Computes per-client RMSE and MAE for each client partition (Clients 1 to 4).
- Handles unobserved items gracefully by falling back to client or global mean rating
  when factor vectors are uninitialized or unavailable.
"""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Default dataset dimensions
DEFAULT_NUM_USERS = 943
DEFAULT_NUM_ITEMS = 1682
DEFAULT_CLIP_RANGE = (1.0, 5.0)


class Evaluator:
    """
    Evaluation engine for benchmark assessment against test.db.

    Parameters
    ----------
    test_db_path : Optional[Union[str, Path]]
        Path to test.db SQLite database. If None, resolves to data/test.db.
    data_dir : Optional[Union[str, Path]]
        Path to data/ directory containing client_1.db .. client_4.db.
        If None, inferred from test_db_path parent.
    clip_range : Optional[Tuple[float, float]]
        Bounds to clip predicted ratings (default: (1.0, 5.0)).
    """

    def __init__(
        self,
        test_db_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        clip_range: Optional[Tuple[float, float]] = DEFAULT_CLIP_RANGE,
    ) -> None:
        self.clip_range = clip_range

        # Resolve paths
        prototype_dir = Path(__file__).resolve().parent.parent
        default_data_dir = prototype_dir / "data"

        if test_db_path is not None:
            self.test_db_path = Path(test_db_path).resolve()
        else:
            self.test_db_path = (default_data_dir / "test.db").resolve()

        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = self.test_db_path.parent

        if not self.test_db_path.exists():
            raise FileNotFoundError(f"Test database not found at: {self.test_db_path}")

        # Load full test set
        self.test_users, self.test_items, self.test_ratings = self._load_test_data()
        self.num_test_samples = len(self.test_ratings)

        # Discover user partitions and client training statistics
        self.client_users: Dict[int, Set[int]] = {}
        self.client_test_indices: Dict[int, np.ndarray] = {}
        self.client_mean_ratings: Dict[int, float] = {}
        self.client_observed_items: Dict[int, Set[int]] = {}
        self.global_train_mean: float = 3.52986  # ML-100K train mean baseline fallback

        self._load_client_metadata()

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ratings from test.db into contiguous numpy arrays."""
        conn = sqlite3.connect(str(self.test_db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, item_id, rating FROM ratings ORDER BY user_id, item_id;")
            rows = cursor.fetchall()
        finally:
            conn.close()

        n = len(rows)
        users = np.empty(n, dtype=np.int32)
        items = np.empty(n, dtype=np.int32)
        ratings = np.empty(n, dtype=np.float32)

        for idx, (u, i, r) in enumerate(rows):
            users[idx] = u
            items[idx] = i
            ratings[idx] = r

        return users, items, ratings

    def _load_client_metadata(self) -> None:
        """Inspect client databases to map users, observed items, and mean ratings."""
        total_train_sum = 0.0
        total_train_count = 0

        for cid in range(1, 5):
            db_path = self.data_dir / f"client_{cid}.db"
            if not db_path.exists():
                self.client_users[cid] = set()
                self.client_test_indices[cid] = np.array([], dtype=np.int64)
                self.client_mean_ratings[cid] = 3.5
                self.client_observed_items[cid] = set()
                continue

            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()
                # Unique users
                cursor.execute("SELECT DISTINCT user_id FROM ratings;")
                u_set = set(int(r[0]) for r in cursor.fetchall())
                self.client_users[cid] = u_set

                # Unique observed items
                cursor.execute("SELECT DISTINCT item_id FROM ratings;")
                self.client_observed_items[cid] = set(int(r[0]) for r in cursor.fetchall())

                # Client mean rating
                cursor.execute("SELECT AVG(rating), COUNT(*) FROM ratings;")
                c_avg, c_cnt = cursor.fetchone()
                avg_val = float(c_avg) if c_avg is not None else 3.5
                self.client_mean_ratings[cid] = avg_val
                if c_cnt:
                    total_train_sum += avg_val * c_cnt
                    total_train_count += c_cnt

            finally:
                conn.close()

            # Precompute test indices for this client
            if len(self.client_users[cid]) > 0:
                mask = np.isin(self.test_users, list(self.client_users[cid]))
                self.client_test_indices[cid] = np.where(mask)[0]
            else:
                self.client_test_indices[cid] = np.array([], dtype=np.int64)

        if total_train_count > 0:
            self.global_train_mean = float(total_train_sum / total_train_count)

    # --------------------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------------------

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).

        Formula: sqrt( (1 / N) * sum (y_true - y_pred)^2 )
        """
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        if len(t) == 0:
            return 0.0
        return float(np.sqrt(np.mean((t - p) ** 2)))

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Formula: (1 / N) * sum |y_true - y_pred|
        """
        t = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
        if len(t) == 0:
            return 0.0
        return float(np.mean(np.abs(t - p)))

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Return (rmse, mae) tuple."""
        return self.compute_rmse(y_true, y_pred), self.compute_mae(y_true, y_pred)

    # --------------------------------------------------------------------------
    # Prediction Generation
    # --------------------------------------------------------------------------

    def predict_client_partition(
        self,
        client_id: int,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        fallback_mode: str = "item_or_mean",
    ) -> np.ndarray:
        """
        Generate predictions for client_id's partition of test ratings.

        Parameters
        ----------
        client_id : int
            Client identifier (1..4).
        user_factors : np.ndarray
            User latent factor matrix of shape (num_users, latent_dim).
        item_factors : np.ndarray
            Item latent factor matrix of shape (num_items, latent_dim).
        fallback_mode : str
            Strategy for unobserved items:
            - 'item_or_mean': uses dot product if factor vector is non-zero,
              else falls back to client mean rating.
            - 'strict_dot': computes dot product for all pairs.
            - 'client_mean_unobserved': for items never observed locally by this client,
              falls back to client average rating.

        Returns
        -------
        np.ndarray
            Predicted ratings array for this client's test ratings.
        """
        indices = self.client_test_indices.get(client_id)
        if indices is None or len(indices) == 0:
            return np.array([], dtype=np.float32)

        u_sub = self.test_users[indices]
        i_sub = self.test_items[indices]

        # Vectorized dot product
        raw_preds = np.sum(user_factors[u_sub] * item_factors[i_sub], axis=1)

        # Handle unobserved / uninitialized item factors
        if fallback_mode == "client_mean_unobserved":
            local_obs = self.client_observed_items.get(client_id, set())
            fallback_val = self.client_mean_ratings.get(client_id, self.global_train_mean)
            for idx, item_id in enumerate(i_sub):
                if item_id not in local_obs:
                    raw_preds[idx] = fallback_val

        elif fallback_mode == "item_or_mean":
            # Check for zero or near-zero factor vectors (uninitialized factors)
            item_norms = np.linalg.norm(item_factors[i_sub], axis=1)
            uninit_mask = item_norms < 1e-5
            if np.any(uninit_mask):
                fallback_val = self.client_mean_ratings.get(client_id, self.global_train_mean)
                raw_preds[uninit_mask] = fallback_val

        # Clipping to valid rating range [1.0, 5.0]
        if self.clip_range is not None:
            np.clip(raw_preds, self.clip_range[0], self.clip_range[1], out=raw_preds)

        return raw_preds.astype(np.float32)

    # --------------------------------------------------------------------------
    # High-Level Evaluation API
    # --------------------------------------------------------------------------

    def evaluate_client(
        self,
        client_id: int,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        fallback_mode: str = "item_or_mean",
    ) -> Dict[str, float]:
        """
        Evaluate RMSE and MAE on client_id's test partition.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'rmse', 'mae', 'count'.
        """
        indices = self.client_test_indices.get(client_id, np.array([], dtype=np.int64))
        if len(indices) == 0:
            return {"rmse": 0.0, "mae": 0.0, "count": 0}

        y_true = self.test_ratings[indices]
        y_pred = self.predict_client_partition(
            client_id=client_id,
            user_factors=user_factors,
            item_factors=item_factors,
            fallback_mode=fallback_mode,
        )

        rmse, mae = self.compute_metrics(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "count": len(y_true)}

    def evaluate_models(
        self,
        client_models: Dict[int, Any],
        global_item_factors: Optional[np.ndarray] = None,
        is_local_only: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate all 4 clients and global test performance.

        Parameters
        ----------
        client_models : Dict[int, Any]
            Mapping of client_id (1..4) to MatrixFactorization or FederatedClientAgent instance.
        global_item_factors : Optional[np.ndarray]
            Global aggregated item factors. If None, extracts item factors from each client model.
        is_local_only : bool
            If True, uses client-specific local item factors with unobserved fallback.

        Returns
        -------
        Dict[str, float]
            Dictionary with:
            - global_rmse, global_mae
            - client_1_rmse, client_1_mae
            - client_2_rmse, client_2_mae
            - client_3_rmse, client_3_mae
            - client_4_rmse, client_4_mae
        """
        all_preds = np.zeros(self.num_test_samples, dtype=np.float32)
        results: Dict[str, float] = {}

        fallback = "client_mean_unobserved" if is_local_only else "item_or_mean"

        for cid in range(1, 5):
            model_obj = client_models.get(cid)
            if model_obj is None:
                continue

            # Extract user factors
            if hasattr(model_obj, "get_user_factors_for_testing"):
                u_mat = model_obj.get_user_factors_for_testing()
            elif hasattr(model_obj, "_user_factors"):
                u_mat = model_obj._user_factors
            elif hasattr(model_obj, "model") and hasattr(model_obj.model, "_user_factors"):
                u_mat = model_obj.model._user_factors
            else:
                raise AttributeError(f"Cannot extract user factors from model for client {cid}")

            # Extract item factors
            if global_item_factors is not None and not is_local_only:
                v_mat = global_item_factors
            else:
                if hasattr(model_obj, "get_item_factors"):
                    v_mat = model_obj.get_item_factors()
                elif hasattr(model_obj, "_item_factors"):
                    v_mat = model_obj._item_factors
                elif hasattr(model_obj, "model") and hasattr(model_obj.model, "get_item_factors"):
                    v_mat = model_obj.model.get_item_factors()
                else:
                    v_mat = global_item_factors

            preds_k = self.predict_client_partition(
                client_id=cid,
                user_factors=u_mat,
                item_factors=v_mat,
                fallback_mode=fallback,
            )

            indices_k = self.client_test_indices[cid]
            all_preds[indices_k] = preds_k

            y_k = self.test_ratings[indices_k]
            rmse_k, mae_k = self.compute_metrics(y_k, preds_k)
            results[f"client_{cid}_rmse"] = rmse_k
            results[f"client_{cid}_mae"] = mae_k

        # Global metrics across all 20,000 test ratings
        g_rmse, g_mae = self.compute_metrics(self.test_ratings, all_preds)
        results["global_rmse"] = g_rmse
        results["global_mae"] = g_mae

        return results

    def get_client_users(self, client_id: int) -> Set[int]:
        """Return the set of user IDs assigned to client_id."""
        return set(self.client_users.get(client_id, set()))

    def __repr__(self) -> str:
        return (
            f"Evaluator(test_db='{self.test_db_path.name}', "
            f"test_samples={self.num_test_samples}, "
            f"clients={list(self.client_users.keys())})"
        )
