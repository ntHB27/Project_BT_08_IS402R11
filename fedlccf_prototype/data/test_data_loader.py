"""
Unit & Integration Tests for Fed-LCCF Data Subsystem:
Validates preprocess_data.py outputs, 0-based indexing, SQLite schemas,
WAL pragma, B-tree indexes, Non-IID client partitions, and ClientDataLoader API.
"""

import json
from pathlib import Path
import sqlite3
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .client_data_loader import ClientDataLoader

DATA_DIR = Path(__file__).resolve().parent


class TestDataSubsystem(unittest.TestCase):
    """Test suite for Data Preparation and ClientDataLoader."""

    def test_01_generated_files_exist(self):
        """Verify that all expected data assets exist and are non-empty."""
        expected_files = [
            DATA_DIR / "__init__.py",
            DATA_DIR / "preprocess_data.py",
            DATA_DIR / "client_data_loader.py",
            DATA_DIR / "item_mapping.json",
            DATA_DIR / "test.csv",
            DATA_DIR / "test.db",
            DATA_DIR / "client_1.db",
            DATA_DIR / "client_2.db",
            DATA_DIR / "client_3.db",
            DATA_DIR / "client_4.db",
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Missing required file: {f}")
            self.assertGreater(f.stat().st_size, 0, f"File is empty: {f}")

    def test_02_row_counts_and_partition_ratios(self):
        """Verify exact row counts and distribution ratios across all partitions."""
        expected_counts = {
            "client_1.db": 36000,
            "client_2.db": 24000,
            "client_3.db": 14400,
            "client_4.db": 5600,
            "test.db": 20000,
        }
        total_train = 0
        for db_name, expected in expected_counts.items():
            db_path = DATA_DIR / db_name
            conn = sqlite3.connect(str(db_path))
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM ratings;").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(cnt, expected, f"{db_name} row count {cnt} != expected {expected}")
            if "client_" in db_name:
                total_train += cnt

        self.assertEqual(total_train, 80000, "Total train ratings must be exactly 80,000")

        # Test test.csv
        df_test = pd.read_csv(DATA_DIR / "test.csv")
        self.assertEqual(len(df_test), 20000, "test.csv must have exactly 20,000 rows")

    def test_03_zero_based_bounds(self):
        """Verify that user_id in [0, 942], item_id in [0, 1681], rating in [1.0, 5.0]."""
        databases = ["client_1.db", "client_2.db", "client_3.db", "client_4.db", "test.db"]
        for db_name in databases:
            db_path = DATA_DIR / db_name
            conn = sqlite3.connect(str(db_path))
            try:
                c = conn.cursor()
                c.execute("SELECT MIN(user_id), MAX(user_id), MIN(item_id), MAX(item_id), MIN(rating), MAX(rating) FROM ratings;")
                u_min, u_max, i_min, i_max, r_min, r_max = c.fetchone()
            finally:
                conn.close()

            self.assertGreaterEqual(u_min, 0, f"{db_name} user_id below 0")
            self.assertLessEqual(u_max, 942, f"{db_name} user_id above 942")
            self.assertGreaterEqual(i_min, 0, f"{db_name} item_id below 0")
            self.assertLessEqual(i_max, 1681, f"{db_name} item_id above 1681")
            self.assertGreaterEqual(r_min, 1.0, f"{db_name} rating below 1.0")
            self.assertLessEqual(r_max, 5.0, f"{db_name} rating above 5.0")

    def test_04_user_disjointness_and_test_coverage(self):
        """Verify that client partitions are user-disjoint and test set covers all 943 users."""
        client_users = {}
        for k in range(1, 5):
            loader = ClientDataLoader(DATA_DIR / f"client_{k}.db")
            users = set(loader.get_local_user_ids())
            client_users[k] = users

        # Pairwise disjointness
        for i in range(1, 5):
            for j in range(i + 1, 5):
                overlap = client_users[i].intersection(client_users[j])
                self.assertEqual(len(overlap), 0, f"Overlap between client {i} and {j}: {overlap}")

        all_train_users = set().union(*client_users.values())
        self.assertEqual(len(all_train_users), 943, "Train users must cover exactly 943 users")
        self.assertEqual(all_train_users, set(range(943)))

        # Test set covers all 943 users
        test_loader = ClientDataLoader(DATA_DIR / "test.db")
        test_users = set(test_loader.get_local_user_ids())
        self.assertEqual(len(test_users), 943, "Test set must cover all 943 users (0 unseen users)")
        self.assertEqual(test_users, set(range(943)))

    def test_05_client_4_cold_start_sparsity(self):
        """Verify that Client 4 has ~50% catalog unobserved (cold-start dominance)."""
        loader_c4 = ClientDataLoader(DATA_DIR / "client_4.db")
        missing_items = loader_c4.get_missing_item_ids(num_items=1682)
        obs_items = loader_c4.get_observed_item_ids()
        self.assertEqual(len(missing_items) + len(obs_items), 1682)

        missing_pct = (len(missing_items) / 1682) * 100
        self.assertGreaterEqual(missing_pct, 45.0, f"Client 4 missing % {missing_pct:.1f}% below 45%")
        self.assertLessEqual(missing_pct, 55.0, f"Client 4 missing % {missing_pct:.1f}% above 55%")

    def test_06_sqlite_schemas_indexes_and_query_plans(self):
        """Verify PRAGMA integrity, WAL mode, and index usage in EXPLAIN QUERY PLAN."""
        databases = ["client_1.db", "client_2.db", "client_3.db", "client_4.db", "test.db"]
        for db_name in databases:
            db_path = DATA_DIR / db_name
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()
                integrity = cursor.execute("PRAGMA integrity_check;").fetchone()[0]
                self.assertEqual(integrity, "ok", f"Integrity check failed for {db_name}")

                journal = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
                self.assertEqual(journal.lower(), "wal", f"Journal mode not wal for {db_name}")

                # Check indexes exist
                idx_rows = cursor.execute("PRAGMA index_list(ratings);").fetchall()
                idx_names = set(r[1] for r in idx_rows)
                for exp_idx in ["idx_ratings_user_id", "idx_ratings_item_id", "idx_ratings_user_item"]:
                    self.assertIn(exp_idx, idx_names, f"Missing index {exp_idx} in {db_name}")

                # Check query plans
                u_plan = list(cursor.execute("EXPLAIN QUERY PLAN SELECT item_id, rating FROM ratings WHERE user_id = ?;", (0,)))
                u_plan_str = " ".join(str(p) for p in u_plan)
                self.assertTrue("idx_ratings_user_id" in u_plan_str or "idx_ratings_user_item" in u_plan_str,
                                f"User index not used in {db_name}: {u_plan_str}")

                i_plan = list(cursor.execute("EXPLAIN QUERY PLAN SELECT item_id, COUNT(*) FROM ratings GROUP BY item_id;"))
                i_plan_str = " ".join(str(p) for p in i_plan)
                self.assertTrue("idx_ratings_item_id" in i_plan_str,
                                f"Item index not used in {db_name}: {i_plan_str}")
            finally:
                conn.close()

    def test_07_item_mapping_metadata(self):
        """Verify item_mapping.json covers all 1,682 items with complete metadata."""
        mapping_path = DATA_DIR / "item_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.assertEqual(len(mapping), 1682, "item_mapping.json must contain 1,682 items")
        for i in range(1682):
            key = str(i)
            self.assertIn(key, mapping, f"Key {key} missing from item_mapping.json")
            entry = mapping[key]
            self.assertEqual(entry["raw_item_id"], i + 1)
            self.assertIsInstance(entry["title"], str)
            self.assertGreater(len(entry["title"]), 0)
            self.assertIsInstance(entry["genres"], list)

    def test_08_client_data_loader_interface(self):
        """Verify all methods of ClientDataLoader class."""
        for client_id, expected_ratings in [(1, 36000), (2, 24000), (3, 14400), (4, 5600)]:
            loader = ClientDataLoader(DATA_DIR / f"client_{client_id}.db")

            # get_total_ratings
            self.assertEqual(loader.get_total_ratings(), expected_ratings)

            # load_sparse_matrix
            mat = loader.load_sparse_matrix(num_users=943, num_items=1682)
            self.assertIsInstance(mat, sp.csr_matrix)
            self.assertEqual(mat.shape, (943, 1682))
            self.assertEqual(mat.nnz, expected_ratings)

            # get_interaction_counts
            counts = loader.get_interaction_counts(num_items=1682)
            self.assertEqual(counts.shape, (1682,))
            self.assertEqual(counts.dtype, np.uint32)
            self.assertEqual(int(counts.sum()), expected_ratings)

            # get_missing_item_ids
            missing = loader.get_missing_item_ids(num_items=1682)
            obs = loader.get_observed_item_ids()
            self.assertEqual(len(missing) + len(obs), 1682)

            # get_train_samples
            users, items, ratings = loader.get_train_samples()
            self.assertEqual(len(users), expected_ratings)
            self.assertEqual(len(items), expected_ratings)
            self.assertEqual(len(ratings), expected_ratings)
            self.assertEqual(users.dtype, np.int32)
            self.assertEqual(items.dtype, np.int32)
            self.assertEqual(ratings.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
