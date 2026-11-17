"""
Comprehensive Unit and Integration Test Suite for Fed-LCCF Cloud Server.

Tests:
1. Health endpoint response and diagnostics.
2. Round 0 seed factor initialization (shape 1682x32, Gaussian N(0, 1/sqrt(d))).
3. Thread-safe 4-client barrier synchronization (RoundBarrierManager).
4. Phase 1 Vanilla FedAvg aggregation math and zero-interaction handling.
5. Phase 2 Leiden clustering engine (cosine similarity, k-NN graph, modularity partition, centroids).
6. Phase 3 Community shrinkage math verification (gamma=10.0, dynamic centroids).
7. End-to-end multi-round workflow via FastAPI TestClient (Rounds 0..6).
8. Validation, error handling, and polling fallback.
"""

import asyncio
import os
import sys
import threading
import time
import unittest
from typing import Any, Dict, List

import numpy as np

# Ensure workspace root is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from fastapi.testclient import TestClient

from fedlccf_prototype.cloud_server.aggregator import (
    DEFAULT_GAMMA,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_ITEMS,
    FederatedAggregator,
    PHASE_LEIDEN_TRANSITION,
    PHASE_SHRINKAGE,
    PHASE_WARMUP,
)
from fedlccf_prototype.cloud_server.barrier import (
    DEFAULT_EXPECTED_CLIENTS,
    RoundBarrierManager,
)
from fedlccf_prototype.cloud_server.leiden_engine import (
    DEFAULT_K_NEIGHBORS,
    LeidenClusteringEngine,
)
from fedlccf_prototype.cloud_server.server import create_app


class TestCloudServerSuite(unittest.TestCase):
    """Full test suite covering cloud server components."""

    def setUp(self) -> None:
        """Create isolated application and client instance per test."""
        self.num_items = 1682
        self.latent_dim = 32
        self.expected_clients = 4
        self.gamma = 10.0
        self.seed = 42

        self.app = create_app(
            expected_clients=self.expected_clients,
            num_items=self.num_items,
            latent_dim=self.latent_dim,
            gamma=self.gamma,
            seed=self.seed,
        )
        self.client = TestClient(self.app)

    # ==========================================================================
    # 1. Health Probe & Round 0 Seed Tests
    # ==========================================================================

    def test_health_check_endpoint(self) -> None:
        """Verify GET /health returns 200, healthy status, round, and uptime."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["current_round"], 1)
        self.assertEqual(data["phase"], PHASE_WARMUP)
        self.assertEqual(data["total_clients_registered"], self.expected_clients)
        self.assertGreaterEqual(data["uptime_seconds"], 0.0)

    def test_round_0_seed_factors(self) -> None:
        """Verify GET /api/v1/fl/get_global_factors/0 returns synchronized seed factors."""
        resp = self.client.get("/api/v1/fl/get_global_factors/0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["round"], 0)
        self.assertEqual(data["phase"], PHASE_WARMUP)

        factors = np.asarray(data["global_factors"], dtype=np.float32)
        self.assertEqual(factors.shape, (self.num_items, self.latent_dim))

        # Check factors are non-trivial Gaussian values
        std_val = float(np.std(factors))
        expected_std = 1.0 / np.sqrt(self.latent_dim)
        self.assertAlmostEqual(std_val, expected_std, delta=0.03)

        # Confirm reproducibility
        app2 = create_app(seed=self.seed)
        c2 = TestClient(app2)
        resp2 = c2.get("/api/v1/fl/get_global_factors/0")
        factors2 = np.asarray(resp2.json()["global_factors"], dtype=np.float32)
        np.testing.assert_array_almost_equal(factors, factors2)

    # ==========================================================================
    # 2. Barrier Synchronization Tests
    # ==========================================================================

    def test_barrier_manager_registration_and_status(self) -> None:
        """Test unit logic of RoundBarrierManager under incremental client uploads."""
        barrier = RoundBarrierManager(expected_clients=4)
        self.assertFalse(barrier.is_round_complete(1))

        # Client 1
        is_comp, count, clients = barrier.register_upload(1, 1, {"data": "c1"})
        self.assertFalse(is_comp)
        self.assertEqual(count, 1)
        self.assertEqual(clients, [1])

        # Client 2
        is_comp, count, clients = barrier.register_upload(2, 1, {"data": "c2"})
        self.assertFalse(is_comp)
        self.assertEqual(count, 2)
        self.assertEqual(clients, [1, 2])

        # Client 3
        is_comp, count, clients = barrier.register_upload(3, 1, {"data": "c3"})
        self.assertFalse(is_comp)
        self.assertEqual(count, 3)
        self.assertEqual(clients, [1, 2, 3])

        # Client 4: Reaches barrier!
        is_comp, count, clients = barrier.register_upload(4, 1, {"data": "c4"})
        self.assertTrue(is_comp)
        self.assertEqual(count, 4)
        self.assertEqual(clients, [1, 2, 3, 4])
        self.assertTrue(barrier.is_round_complete(1))

        # Inspect status
        status = barrier.get_round_status(1)
        self.assertTrue(status["round_completed"])
        self.assertEqual(status["uploaded_count"], 4)
        self.assertEqual(status["uploaded_clients"], [1, 2, 3, 4])

        # Test duplicate upload from existing client updates without incrementing count
        is_comp, count, clients = barrier.register_upload(4, 1, {"data": "c4_v2"})
        self.assertEqual(count, 4)

    def test_barrier_thread_safety_concurrent_uploads(self) -> None:
        """Test barrier manager concurrency safety under multi-threaded client uploads."""
        barrier = RoundBarrierManager(expected_clients=4)

        def worker_upload(cid: int) -> None:
            time.sleep(0.01 * cid)
            barrier.register_upload(cid, 1, {"payload": f"client_{cid}"})

        threads = [threading.Thread(target=worker_upload, args=(cid,)) for cid in range(1, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(barrier.is_round_complete(1))
        uploads = barrier.get_uploads(1)
        self.assertEqual(len(uploads), 4)
        self.assertEqual(sorted(list(uploads.keys())), [1, 2, 3, 4])

    def test_barrier_async_wait_event(self) -> None:
        """Test async wait_for_round event notification."""
        barrier = RoundBarrierManager(expected_clients=2)

        async def scenario() -> None:
            # Start waiting in task
            async def waiter() -> bool:
                return await barrier.wait_for_round(1, timeout=5.0)

            wait_task = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
            self.assertFalse(wait_task.done())

            # Upload client 1
            barrier.register_upload(1, 1, {"dummy": 1})
            await asyncio.sleep(0.05)
            self.assertFalse(wait_task.done())

            # Upload client 2 -> completes barrier
            barrier.register_upload(2, 1, {"dummy": 2})
            res = await wait_task
            self.assertTrue(res)

        asyncio.run(scenario())

    # ==========================================================================
    # 3. Phase 1 Vanilla FedAvg Math Tests
    # ==========================================================================

    def test_phase_1_fedavg_exact_mathematics(self) -> None:
        """
        Verify exact count-weighted average math:
            V_i = sum(n_{k,i} * V_{k,i}) / sum(n_{k,i})
        """
        aggregator = FederatedAggregator(
            num_items=4,
            latent_dim=2,
            gamma=10.0,
            random_state=42,
        )

        # Set specific initial factors
        init_factors = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
        ], dtype=np.float32)
        aggregator._global_factors = init_factors.copy()
        aggregator._factors_history[0] = init_factors.copy()

        # 4 clients with known weights and vectors
        # Item 0: counts [10, 20, 30, 40], vectors [1.0, 2.0, 3.0, 4.0]
        # Expected: (10*1 + 20*2 + 30*3 + 40*4) / 100 = 300 / 100 = 3.0
        # Item 1: counts [5, 15, 0, 0], vectors [2.0, 6.0, 0.0, 0.0]
        # Expected: (5*2 + 15*6) / 20 = (10 + 90) / 20 = 5.0
        # Item 2: counts [0, 0, 0, 0] -> Unobserved!
        # Expected: retain previous factors [30.0, 30.0]
        # Item 3: counts [1, 1, 1, 1], vectors [1.0, 2.0, 3.0, 4.0]
        # Expected: (1+2+3+4)/4 = 2.5
        uploads = {
            1: {
                "factors": [[1.0, 1.0], [2.0, 2.0], [99.0, 99.0], [1.0, 1.0]],
                "counts": [10, 5, 0, 1],
            },
            2: {
                "factors": [[2.0, 2.0], [6.0, 6.0], [99.0, 99.0], [2.0, 2.0]],
                "counts": [20, 15, 0, 1],
            },
            3: {
                "factors": [[3.0, 3.0], [0.0, 0.0], [99.0, 99.0], [3.0, 3.0]],
                "counts": [30, 0, 0, 1],
            },
            4: {
                "factors": [[4.0, 4.0], [0.0, 0.0], [99.0, 99.0], [4.0, 4.0]],
                "counts": [40, 0, 0, 1],
            },
        }

        new_v = aggregator.aggregate_round(1, uploads)

        # Verification of item 0
        np.testing.assert_allclose(new_v[0], [3.0, 3.0], rtol=1e-5)
        # Verification of item 1
        np.testing.assert_allclose(new_v[1], [5.0, 5.0], rtol=1e-5)
        # Verification of unobserved item 2 (retained previous factors)
        np.testing.assert_allclose(new_v[2], [30.0, 30.0], rtol=1e-5)
        # Verification of item 3
        np.testing.assert_allclose(new_v[3], [2.5, 2.5], rtol=1e-5)

    # ==========================================================================
    # 4. Phase 2 Leiden Clustering Engine Tests
    # ==========================================================================

    def test_cosine_similarity_matrix_properties(self) -> None:
        """Verify cosine similarity matrix properties (shape, diagonal 0, bounds [-1, 1])."""
        engine = LeidenClusteringEngine(
            k_neighbors=20,
            seed=42,
            num_items=100,
            latent_dim=32,
        )
        rng = np.random.RandomState(42)
        V = rng.randn(100, 32).astype(np.float32)

        S = engine.compute_cosine_similarity(V)
        self.assertEqual(S.shape, (100, 100))

        # Diagonal must be strictly 0.0
        np.testing.assert_allclose(np.diag(S), 0.0, atol=1e-7)

        # Symmetry: S_ij == S_ji
        np.testing.assert_allclose(S, S.T, atol=1e-6)

        # Bounds: [-1.0, 1.0]
        self.assertTrue(np.all(S >= -1.0 - 1e-6))
        self.assertTrue(np.all(S <= 1.0 + 1e-6))

        # Check known orthogonal vectors produce similarity 0.0
        V_orth = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        e2 = LeidenClusteringEngine(k_neighbors=1, num_items=2, latent_dim=2)
        S_orth = e2.compute_cosine_similarity(V_orth)
        self.assertAlmostEqual(float(S_orth[0, 1]), 0.0, places=5)

    def test_knn_graph_construction(self) -> None:
        """Verify k-NN graph connects top-k positive neighbors with positive symmetric weights."""
        engine = LeidenClusteringEngine(
            k_neighbors=5,
            seed=42,
            num_items=50,
            latent_dim=16,
        )
        rng = np.random.RandomState(42)
        V = rng.randn(50, 16).astype(np.float32)
        S = engine.compute_cosine_similarity(V)

        graph, num_edges = engine.build_knn_graph(S)
        self.assertEqual(graph.vcount(), 50)
        self.assertGreater(num_edges, 0)
        self.assertEqual(graph.ecount(), num_edges)

        # Check edge weights are strictly positive
        weights = graph.es["weight"]
        self.assertTrue(all(w > 0.0 for w in weights))

    def test_leiden_community_detection_1682_items(self) -> None:
        """
        Run Leiden community detection on full 1,682 items matrix.
        Verify:
        - All 1,682 items assigned.
        - Number of communities M_c in [10, 30].
        - Community centroids non-empty and shape (32,).
        - Modularity score is positive.
        """
        engine = LeidenClusteringEngine(
            k_neighbors=20,
            seed=42,
            num_items=1682,
            latent_dim=32,
        )
        rng = np.random.RandomState(42)
        V = rng.randn(1682, 32).astype(np.float32)

        res = engine.detect_communities(V)

        # All 1,682 items assigned
        self.assertEqual(len(res["community_map"]), 1682)
        assigned_items = set(res["community_map"].keys())
        self.assertEqual(assigned_items, set(range(1682)))

        num_comm = res["num_communities"]
        # Should typically discover ~10-30 communities for MovieLens 1682 items
        self.assertGreaterEqual(num_comm, 5)
        self.assertLessEqual(num_comm, 50)
        self.assertGreater(res["modularity"], 0.0)

        # Verify centroids
        centroids = res["centroids"]
        self.assertEqual(len(centroids), num_comm)
        for cid, mu in centroids.items():
            self.assertEqual(mu.shape, (32,))
            self.assertEqual(mu.dtype, np.float32)

        # Verify mathematical correctness of centroid for community 0
        comm_0_items = [i for i, c in res["community_map"].items() if c == 0]
        expected_mu_0 = np.mean(V[comm_0_items], axis=0)
        np.testing.assert_allclose(centroids[0], expected_mu_0, atol=1e-5)

    # ==========================================================================
    # 5. Phase 3 Community Shrinkage Math Tests
    # ==========================================================================

    def test_phase_3_community_shrinkage_formula(self) -> None:
        """
        Verify Phase 3 community shrinkage formula:
            v_i = [N_i / (N_i + gamma)] * v_i_raw + [gamma / (N_i + gamma)] * mu_c
        with gamma = 10.0.
        """
        gamma = 10.0
        aggregator = FederatedAggregator(
            num_items=3,
            latent_dim=2,
            gamma=gamma,
            random_state=42,
        )

        # Mock community partition: items 0, 1, 2 belong to community 0
        mu_0 = np.array([10.0, 20.0], dtype=np.float32)
        aggregator._community_map = {0: 0, 1: 0, 2: 0}
        aggregator._centroids = {0: mu_0.copy()}
        aggregator._num_communities = 1
        aggregator._current_round = 5
        aggregator._current_phase = PHASE_SHRINKAGE

        # Item 0: Dense (N_0 = 990), raw = [0.0, 0.0]
        # Expected: (990 / 1000) * [0, 0] + (10 / 1000) * [10, 20] = [0.10, 0.20]
        # Item 1: Sparse (N_1 = 1), raw = [0.0, 0.0]
        # Expected: (1 / 11) * [0, 0] + (10 / 11) * [10, 20] = [9.0909, 18.1818]
        # Item 2: Unobserved (N_2 = 0)
        # Expected: mu_0 = [10.0, 20.0]
        uploads = {
            1: {
                "factors": [[0.0, 0.0], [0.0, 0.0], [99.0, 99.0]],
                "counts": [990, 1, 0],
            },
        }

        v_new = aggregator.aggregate_round(6, uploads)

        # Dense item 0
        expected_dense = (990.0 / 1000.0) * np.zeros(2) + (10.0 / 1000.0) * mu_0
        np.testing.assert_allclose(v_new[0], expected_dense, rtol=1e-5)

        # Sparse item 1
        expected_sparse = (1.0 / 11.0) * np.zeros(2) + (10.0 / 11.0) * mu_0
        np.testing.assert_allclose(v_new[1], expected_sparse, rtol=1e-5)

        # Unobserved item 2 (anchored strictly to centroid)
        np.testing.assert_allclose(v_new[2], mu_0, rtol=1e-5)

    # ==========================================================================
    # 6. End-to-End Multi-Round Integration (Rounds 0..6)
    # ==========================================================================

    def test_end_to_end_simulation_through_fastapi_client(self) -> None:
        """
        Simulate complete 4-client protocol over Rounds 0..6:
        - Pull round 0 initial factors.
        - Rounds 1..4: Phase 1 Vanilla FedAvg.
        - Round 5: Phase 1 aggregation followed by automatic Phase 2 Leiden clustering.
        - Check Round 5 response includes valid community_map and centroids.
        - Round 6: Phase 3 Community Shrinkage aggregation.
        """
        # Step 0: Pull Round 0
        r0_resp = self.client.get("/api/v1/fl/get_global_factors/0")
        self.assertEqual(r0_resp.status_code, 200)
        r0_data = r0_resp.json()
        current_factors = np.asarray(r0_data["global_factors"], dtype=np.float32)
        self.assertEqual(current_factors.shape, (self.num_items, self.latent_dim))

        # Rounds 1 through 6
        for r in range(1, 7):
            # Inspect round status before uploads
            st_pre = self.client.get(f"/api/v1/fl/round_status?round={r}").json()
            self.assertFalse(st_pre["round_completed"])

            # 4 clients upload factors
            for cid in range(1, 5):
                # Simulate local update: factor + small noise
                client_factors = (current_factors + 0.01 * cid).tolist()
                # Simulate realistic counts: some positive, some zero
                client_counts = [5 if (i + cid) % 2 == 0 else 0 for i in range(self.num_items)]

                up_resp = self.client.post(
                    "/api/v1/fl/upload_factors",
                    json={
                        "client_id": cid,
                        "round": r,
                        "item_factors": client_factors,
                        "item_counts": client_counts,
                        "num_samples": 5000,
                        "local_loss": 0.85,
                    },
                )
                self.assertEqual(up_resp.status_code, 200)
                data = up_resp.json()
                if cid < 4:
                    self.assertFalse(data["barrier_reached"])
                else:
                    self.assertTrue(data["barrier_reached"])

            # Check round status post uploads
            st_post = self.client.get(f"/api/v1/fl/round_status?round={r}").json()
            self.assertTrue(st_post["round_completed"])
            self.assertEqual(st_post["uploaded_count"], 4)

            # Retrieve aggregated global factors for round r
            gf_resp = self.client.get(f"/api/v1/fl/get_global_factors/{r}")
            self.assertEqual(gf_resp.status_code, 200)
            gf_data = gf_resp.json()
            self.assertEqual(gf_data["round"], r)

            new_v = np.asarray(gf_data["global_factors"], dtype=np.float32)
            self.assertEqual(new_v.shape, (self.num_items, self.latent_dim))
            current_factors = new_v

            # Verify phase transitions
            if r <= 4:
                self.assertEqual(gf_data["phase"], PHASE_WARMUP)
                self.assertIsNone(gf_data.get("community_map"))
            elif r == 5:
                # Post-Round 5: Leiden community detection must have completed!
                self.assertEqual(gf_data["phase"], PHASE_LEIDEN_TRANSITION)
                cmap = gf_data.get("community_map")
                cents = gf_data.get("centroids")
                self.assertIsNotNone(cmap)
                self.assertIsNotNone(cents)
                self.assertEqual(len(cmap), self.num_items)
                self.assertGreater(len(cents), 0)
                self.assertIn("num_communities", gf_data)
            elif r == 6:
                # Phase 3: Community Shrinkage
                self.assertEqual(gf_data["phase"], PHASE_SHRINKAGE)
                cmap = gf_data.get("community_map")
                cents = gf_data.get("centroids")
                self.assertIsNotNone(cmap)
                self.assertIsNotNone(cents)

    # ==========================================================================
    # 7. Polling Fallback and Validation Tests
    # ==========================================================================

    def test_polling_fallback_returns_202_when_not_ready(self) -> None:
        """Verify GET /api/v1/fl/get_global_factors/{round}?poll=true returns 202 if pending."""
        resp = self.client.get("/api/v1/fl/get_global_factors/1?poll=true")
        self.assertEqual(resp.status_code, 202)
        data = resp.json()
        self.assertEqual(data["status"], "waiting")
        self.assertEqual(data["round"], 1)

    def test_upload_factors_validation_rejects_missing_payload(self) -> None:
        """Verify upload endpoint rejects requests missing both factors and counts."""
        resp = self.client.post(
            "/api/v1/fl/upload_factors",
            json={
                "client_id": 1,
                "round": 1,
            },
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
