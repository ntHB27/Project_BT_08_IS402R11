"""
Unit Test Suite for Local Matrix Factorization & Client Agent Subsystem.

Tests:
1. Matrix dimensions (U in R^{943 x 32}, V in R^{1682 x 32}).
2. Strict User Factor Privacy (U excluded from upload payloads).
3. Adaptive lambda_k regularization mapping and mathematical shrinkage effect.
4. Pointwise SGD loss reduction on synthetic and real SQLite client data.
5. Phase 1 Unobserved Items Mean Initialization.
6. Phase 3 Leiden Centroid Anchoring (unobserved pinning & observed penalty).
7. Prediction clipping and RMSE / MAE evaluation.
8. Local client-side user factor disk persistence.
9. FederatedClientAgent initialization and end-to-end communication workflow.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path when test script is executed directly
_CURRENT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _CURRENT_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import numpy as np

from client_ml.client_agent import FederatedClientAgent
from client_ml.local_mf import (
    ADAPTIVE_LAMBDA_MAP,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_ITEMS,
    DEFAULT_NUM_USERS,
    MatrixFactorization,
)
from data.client_data_loader import ClientDataLoader


class TestLocalMatrixFactorization(unittest.TestCase):
    """Test suite for MatrixFactorization class."""

    def setUp(self) -> None:
        self.prototype_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.prototype_dir / "data"

    def test_matrix_dimensions(self) -> None:
        """Verify U and V factor matrix dimensions match specification (943x32, 1682x32)."""
        mf = MatrixFactorization(
            num_users=DEFAULT_NUM_USERS,
            num_items=DEFAULT_NUM_ITEMS,
            latent_dim=DEFAULT_LATENT_DIM,
            random_state=42,
        )
        self.assertEqual(mf.user_factors_shape, (943, 32))
        self.assertEqual(mf.item_factors_shape, (1682, 32))

        # Check export getter
        v_copy = mf.get_item_factors()
        self.assertEqual(v_copy.shape, (1682, 32))
        self.assertEqual(v_copy.dtype, np.float32)

    def test_privacy_enforcement(self) -> None:
        """Verify user factor matrix U is strictly excluded from all upload payloads."""
        mf = MatrixFactorization(client_id=1, random_state=42)

        # Mock training interaction
        users = np.array([0, 1, 2], dtype=np.int32)
        items = np.array([10, 20, 30], dtype=np.int32)
        ratings = np.array([4.0, 5.0, 3.0], dtype=np.float32)
        mf.fit(users, items, ratings, epochs=1)

        # 1. Tuple payload contract
        payload_tuple = mf.get_upload_payload()
        self.assertIsInstance(payload_tuple, tuple)
        self.assertEqual(len(payload_tuple), 4)
        v_k, n_k, num_samples, train_loss = payload_tuple

        self.assertEqual(v_k.shape, (1682, 32))
        self.assertEqual(n_k.shape, (1682,))
        self.assertEqual(num_samples, 3)
        self.assertIsInstance(train_loss, float)

        # Verify defensive copy
        orig_val = v_k[0, 0]
        v_k[0, 0] += 999.0
        self.assertNotEqual(mf.get_item_factors()[0, 0], v_k[0, 0])
        self.assertEqual(mf.get_item_factors()[0, 0], orig_val)

        # 2. Dictionary payload contract for HTTP upload
        payload_dict = mf.get_upload_payload_dict(round_num=1, client_id=1)
        self.assertIsInstance(payload_dict, dict)
        self.assertEqual(payload_dict["client_id"], 1)
        self.assertEqual(payload_dict["round"], 1)
        self.assertEqual(payload_dict["num_samples"], 3)

        # Critical privacy assertions
        self.assertNotIn("user_factors", payload_dict)
        self.assertNotIn("_user_factors", payload_dict)
        self.assertNotIn("U", payload_dict)
        self.assertNotIn("user", payload_dict)
        self.assertNotIn("users", payload_dict)

    def test_adaptive_lambda_mapping(self) -> None:
        """Verify client_id automatically configures adaptive regularization lambda_k."""
        mf1 = MatrixFactorization(client_id=1)
        mf2 = MatrixFactorization(client_id=2)
        mf3 = MatrixFactorization(client_id=3)
        mf4 = MatrixFactorization(client_id=4)

        self.assertEqual(mf1.reg_u, 0.01)
        self.assertEqual(mf1.reg_v, 0.01)

        self.assertEqual(mf2.reg_u, 0.02)
        self.assertEqual(mf2.reg_v, 0.02)

        self.assertEqual(mf3.reg_u, 0.05)
        self.assertEqual(mf3.reg_v, 0.05)

        self.assertEqual(mf4.reg_u, 0.10)
        self.assertEqual(mf4.reg_v, 0.10)

        # Explicit overrides take precedence
        mf_custom = MatrixFactorization(client_id=4, reg_u=0.03, reg_v=0.07)
        self.assertEqual(mf_custom.reg_u, 0.03)
        self.assertEqual(mf_custom.reg_v, 0.07)

    def test_adaptive_lambda_regularization_effect(self) -> None:
        """
        Verify that higher lambda_k (Client 4) exerts stronger shrinkage on weights
        compared to lower lambda_k (Client 1) under identical data.
        """
        # Train Client 1 (lambda = 0.01) vs Client 4 (lambda = 0.10)
        users = np.array([5], dtype=np.int32)
        items = np.array([12], dtype=np.int32)
        ratings = np.array([5.0], dtype=np.float32)

        mf_dense = MatrixFactorization(client_id=1, learning_rate=0.05, random_state=123)
        mf_sparse = MatrixFactorization(client_id=4, learning_rate=0.05, random_state=123)

        # Start from identical initialized weights
        init_v = np.ones((1682, 32), dtype=np.float32) * 0.1
        mf_dense.set_item_factors(init_v)
        mf_sparse.set_item_factors(init_v)

        mf_dense.fit(users, items, ratings, epochs=20, shuffle=False)
        mf_sparse.fit(users, items, ratings, epochs=20, shuffle=False)

        norm_dense = np.linalg.norm(mf_dense.get_item_factors()[12])
        norm_sparse = np.linalg.norm(mf_sparse.get_item_factors()[12])

        # Stronger regularization suppresses factor magnitude more effectively
        # In presence of single sample, lambda=0.10 keeps norm lower than lambda=0.01
        self.assertLess(norm_sparse, norm_dense)

    def test_sgd_loss_reduction_synthetic(self) -> None:
        """Verify pointwise SGD strictly reduces MSE loss on synthetic low-rank ratings."""
        rng = np.random.default_rng(999)
        num_samples = 500
        users = rng.integers(0, 50, size=num_samples, dtype=np.int32)
        items = rng.integers(0, 100, size=num_samples, dtype=np.int32)
        # Ratings generated from underlying ground truth latent space
        ratings = np.clip(rng.normal(3.5, 0.8, size=num_samples), 1.0, 5.0).astype(np.float32)

        mf = MatrixFactorization(
            num_users=50,
            num_items=100,
            latent_dim=16,
            learning_rate=0.02,
            reg_u=0.001,
            reg_v=0.001,
            random_state=42,
        )

        initial_loss = mf.compute_loss(users, items, ratings)
        epoch_losses = mf.fit(users, items, ratings, epochs=15, shuffle=True)
        final_loss = epoch_losses[-1]

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, epoch_losses[0])

    def test_sgd_loss_reduction_sqlite(self) -> None:
        """Verify pointwise SGD reduces loss on real SQLite partition (client_4.db)."""
        db_path = self.data_dir / "client_4.db"
        if not db_path.exists():
            self.skipTest("client_4.db not found")

        loader = ClientDataLoader(db_path)
        users, items, ratings = loader.get_train_samples()

        mf = MatrixFactorization(client_id=4, learning_rate=0.01, random_state=42)
        initial_rmse = mf.evaluate_rmse(users, items, ratings)
        epoch_losses = mf.fit(users, items, ratings, epochs=5, shuffle=True)
        final_rmse = mf.evaluate_rmse(users, items, ratings)

        self.assertLess(final_rmse, initial_rmse)
        self.assertLess(epoch_losses[-1], epoch_losses[0])

    def test_phase1_unobserved_items_mean_initialization(self) -> None:
        """
        Verify Phase 1: Unobserved items (n_{k,i} == 0) are initialized
        to the exact mean vector of observed items.
        """
        mf = MatrixFactorization(num_users=10, num_items=10, latent_dim=4, random_state=42)

        # Set custom item factors for clear tracking
        item_factors = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
        mf.set_item_factors(item_factors)

        # Suppose items 0, 1, 2 are observed; items 3..9 are unobserved
        counts = np.zeros(10, dtype=np.uint32)
        counts[0] = 10
        counts[1] = 5
        counts[2] = 20

        observed_indices = [0, 1, 2]
        unobserved_indices = list(range(3, 10))

        expected_mean = np.mean(item_factors[observed_indices], axis=0)

        mean_vec = mf.initialize_unobserved_items_mean(
            observed_indices=observed_indices,
            missing_indices=unobserved_indices,
            interaction_counts=counts,
        )

        self.assertTrue(np.allclose(mean_vec, expected_mean))

        # Check that observed items are NOT modified
        updated_factors = mf.get_item_factors()
        self.assertTrue(np.allclose(updated_factors[observed_indices], item_factors[observed_indices]))

        # Check that ALL unobserved items are set to expected_mean
        for unobs in unobserved_indices:
            self.assertTrue(np.allclose(updated_factors[unobs], expected_mean))

    def test_phase3_unobserved_items_centroid_anchoring(self) -> None:
        """
        Verify Phase 3: Unobserved items are anchored to their community centroid mu_{C(i)}.
        """
        mf = MatrixFactorization(num_users=10, num_items=6, latent_dim=4, random_state=42)

        # Community map: items 0,1 -> comm 1; items 2,3 -> comm 2; items 4,5 -> comm 1
        community_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2}
        centroids = {
            1: np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            2: np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32),
        }

        mf.set_community_info(community_map, centroids, activate_anchoring=True)

        # Items 1 and 3 are unobserved
        counts = np.array([5, 0, 12, 0, 7, 8], dtype=np.uint32)
        anchored_count = mf.anchor_unobserved_items_to_centroids(
            missing_indices=[1, 3],
            interaction_counts=counts,
        )

        self.assertEqual(anchored_count, 2)
        v = mf.get_item_factors()

        # Item 1 is in community 1 -> anchored to centroids[1]
        self.assertTrue(np.allclose(v[1], centroids[1]))
        # Item 3 is in community 2 -> anchored to centroids[2]
        self.assertTrue(np.allclose(v[3], centroids[2]))

    def test_phase3_centroid_anchoring_sgd_penalty(self) -> None:
        """
        Verify that Phase 3 centroid anchoring penalty pulls observed items
        towards their community centroid during SGD training.
        """
        # Centroid target at [5.0, 5.0, ...]
        centroid_val = np.ones(32, dtype=np.float32) * 5.0
        community_map = {0: 1}
        centroids = {1: centroid_val}

        # Model A: Vanilla (unanchored)
        mf_unanchored = MatrixFactorization(client_id=1, learning_rate=0.01, random_state=42)
        # Model B: Anchored with lambda_anchor = 0.05
        mf_anchored = MatrixFactorization(
            client_id=1,
            learning_rate=0.01,
            lambda_anchor=0.05,
            random_state=42,
        )

        # Set identical initial factor for item 0
        init_v0 = np.zeros(32, dtype=np.float32)
        v_init = np.zeros((1682, 32), dtype=np.float32)
        v_init[0] = init_v0
        mf_unanchored.set_item_factors(v_init)
        mf_anchored.set_item_factors(v_init)

        mf_anchored.set_community_info(community_map, centroids, activate_anchoring=True)

        # Single training interaction
        users = np.array([0], dtype=np.int32)
        items = np.array([0], dtype=np.int32)
        ratings = np.array([3.0], dtype=np.float32)

        mf_unanchored.fit(users, items, ratings, epochs=1, shuffle=False)
        mf_anchored.fit(users, items, ratings, epochs=1, shuffle=False)

        v_unanchored = mf_unanchored.get_item_factors()[0]
        v_anchored = mf_anchored.get_item_factors()[0]

        dist_unanchored = np.linalg.norm(v_unanchored - centroid_val)
        dist_anchored = np.linalg.norm(v_anchored - centroid_val)

        # The anchoring penalty should pull v_anchored closer to the centroid
        self.assertLess(dist_anchored, dist_unanchored)

    def test_predictions_and_clipping(self) -> None:
        """Verify rating predictions and [1.0, 5.0] range clipping."""
        mf = MatrixFactorization(num_users=5, num_items=5, latent_dim=4, random_state=42)

        pred = mf.predict(0, 0, clip_range=(1.0, 5.0))
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 5.0)

        users = np.array([0, 1, 2], dtype=np.int32)
        items = np.array([1, 2, 3], dtype=np.int32)
        batch_preds = mf.predict_batch(users, items, clip_range=(1.0, 5.0))
        self.assertEqual(len(batch_preds), 3)
        self.assertTrue(np.all(batch_preds >= 1.0))
        self.assertTrue(np.all(batch_preds <= 5.0))

    def test_save_and_load_user_factors(self) -> None:
        """Verify local persistence of private user factors U."""
        mf = MatrixFactorization(random_state=42)
        users = np.array([0, 1], dtype=np.int32)
        items = np.array([0, 1], dtype=np.int32)
        ratings = np.array([4.0, 5.0], dtype=np.float32)
        mf.fit(users, items, ratings, epochs=2)

        orig_u = mf.get_user_factors_for_testing()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_user_factors.npy"
            mf.save_user_factors(save_path)
            self.assertTrue(save_path.exists())

            # Load into a new instance
            mf_new = MatrixFactorization(random_state=99)
            self.assertFalse(np.allclose(mf_new.get_user_factors_for_testing(), orig_u))

            mf_new.load_user_factors(save_path)
            self.assertTrue(np.allclose(mf_new.get_user_factors_for_testing(), orig_u))

    def test_edge_cases_and_input_validation(self) -> None:
        """Verify robust error handling for invalid shapes and empty training arrays."""
        mf = MatrixFactorization()

        # 1. Invalid item factor shape
        with self.assertRaises(ValueError):
            mf.set_item_factors(np.zeros((100, 32)))

        # 2. Invalid count vector shape
        with self.assertRaises(ValueError):
            mf.set_interaction_counts(np.zeros(100))

        # 3. Empty training arrays
        empty_losses = mf.fit(
            users=np.array([], dtype=np.int32),
            items=np.array([], dtype=np.int32),
            ratings=np.array([], dtype=np.float32),
            epochs=5,
        )
        self.assertEqual(empty_losses, [])

        # 4. Partial community mapping (unobserved item without centroid)
        mf.set_community_info(community_map={0: 99}, centroids={1: np.ones(32)})
        # Item 0 has community 99 which is not in centroids -> should not crash
        anchored_count = mf.anchor_unobserved_items_to_centroids(missing_indices=[0, 1])
        self.assertEqual(anchored_count, 0)


class TestFederatedClientAgent(unittest.TestCase):
    """Test suite for FederatedClientAgent class."""

    def setUp(self) -> None:
        self.prototype_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.prototype_dir / "data"
        self.db_path = self.data_dir / "client_4.db"

    def test_client_agent_initialization(self) -> None:
        """Verify Client Agent correctly initializes with ClientDataLoader and MatrixFactorization."""
        if not self.db_path.exists():
            self.skipTest("client_4.db not found")

        agent = FederatedClientAgent(
            client_id=4,
            db_path=self.db_path,
            rounds=5,
            epochs_per_round=2,
            random_state=42,
        )

        self.assertEqual(agent.client_id, 4)
        self.assertEqual(agent.rounds, 5)
        self.assertEqual(agent.epochs_per_round, 2)
        self.assertEqual(len(agent.ratings), 5600)
        # Client 4 has 871 missing items out of 1682 (~51.8% cold-start items)
        self.assertEqual(len(agent.missing_item_ids), 871)
        self.assertEqual(agent.model.reg_u, 0.10)
        self.assertEqual(agent.model.reg_v, 0.10)

    def test_client_agent_mock_communication_round(self) -> None:
        """Verify client execution round with mock HTTP server endpoints."""
        if not self.db_path.exists():
            self.skipTest("client_4.db not found")

        agent = FederatedClientAgent(
            client_id=4,
            db_path=self.db_path,
            rounds=1,
            epochs_per_round=1,
            random_state=42,
        )

        mock_global_v = np.ones((1682, 32), dtype=np.float32) * 0.05
        mock_community_map = {str(i): (i % 5) for i in range(1682)}
        mock_centroids = {str(c): (np.ones(32, dtype=np.float32) * float(c)).tolist() for c in range(5)}

        # Mock _http_get and _http_post
        def mock_get(endpoint: str):
            if "get_global_factors/0" in endpoint:
                return 200, {"round": 0, "global_factors": mock_global_v.tolist()}
            if "get_global_factors/1" in endpoint:
                return 200, {
                    "round": 1,
                    "global_factors": mock_global_v.tolist(),
                    "community_map": mock_community_map,
                    "centroids": mock_centroids,
                }
            return 404, None

        def mock_post(endpoint: str, data: dict):
            # Assert that user factors are NEVER in the POST payload
            self.assertNotIn("user_factors", data)
            self.assertNotIn("_user_factors", data)
            self.assertNotIn("U", data)
            self.assertEqual(data["client_id"], 4)
            self.assertEqual(data["round"], 1)
            self.assertEqual(len(data["item_factors"]), 1682)
            self.assertEqual(len(data["item_counts"]), 1682)
            return 200, {"status": "received"}

        agent._http_get = MagicMock(side_effect=mock_get)
        agent._http_post = MagicMock(side_effect=mock_post)

        round_stats = agent.run_round(round_num=1)

        self.assertEqual(round_stats["client_id"], 4)
        self.assertEqual(round_stats["round"], 1)
        self.assertTrue(round_stats["upload_success"])
        self.assertTrue(round_stats["has_community_map"])
        self.assertGreater(round_stats["train_loss"], 0.0)

        # Verify global factors and community artifacts updated on model
        self.assertIsNotNone(agent.community_map)
        self.assertIsNotNone(agent.centroids)
        self.assertEqual(len(agent.community_map), 1682)


if __name__ == "__main__":
    unittest.main()
