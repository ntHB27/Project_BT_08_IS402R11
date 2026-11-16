"""
Client Data Access Layer for Fed-LCCF SQLite Partitions.

Provides the interface contract between Milestone 1 (Data Subsystem)
and Milestone 2 (Client Machine Learning Subsystem):
- Loads ratings into SciPy CSR sparse matrix.
- Extracts per-item interaction count vectors (n_k).
- Identifies unobserved / cold-start item IDs for centroid anchoring.
- Provides vectorized (user, item, rating) arrays for local SGD training.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

DEFAULT_NUM_USERS = 943
DEFAULT_NUM_ITEMS = 1682


class ClientDataLoader:
    """
    Data Access Layer for client SQLite databases and evaluation sets.

    Parameters
    ----------
    db_path : Union[str, Path]
        Path to the client SQLite database (e.g. client_1.db, test.db).
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """
        Create and return an optimized SQLite connection with WAL journal mode.
        Callers that use this method directly are responsible for closing the connection.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Internal context manager ensuring connection is always closed."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def get_total_ratings(self) -> int:
        """Return the total number of ratings in this client partition."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ratings;")
            return int(cursor.fetchone()[0])

    def get_local_user_ids(self) -> List[int]:
        """
        Return the sorted list of unique user IDs present in this client's database.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT user_id FROM ratings ORDER BY user_id ASC;")
            rows = cursor.fetchall()
            return [int(r[0]) for r in rows]

    def get_observed_item_ids(self) -> List[int]:
        """
        Return the sorted list of item IDs that have at least one interaction.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT item_id FROM ratings ORDER BY item_id ASC;")
            rows = cursor.fetchall()
            return [int(r[0]) for r in rows]

    def get_interaction_counts(self, num_items: int = DEFAULT_NUM_ITEMS) -> np.ndarray:
        """
        Return a 1D uint32 numpy array of shape (num_items,) containing n_{k, i},
        the frequency of ratings for each item at this client.

        Parameters
        ----------
        num_items : int
            Total number of items in the global catalog (default: 1682).

        Returns
        -------
        np.ndarray
            Array of shape (num_items,) with dtype uint32.
        """
        counts = np.zeros(num_items, dtype=np.uint32)
        with self._connect() as conn:
            cursor = conn.cursor()
            # Fast scan using covering index idx_ratings_item_id
            cursor.execute("SELECT item_id, COUNT(*) FROM ratings GROUP BY item_id;")
            for item_id, count in cursor.fetchall():
                if 0 <= item_id < num_items:
                    counts[item_id] = count
                else:
                    raise IndexError(f"item_id {item_id} out of bounds for catalog size {num_items}")
        return counts

    def get_missing_item_ids(self, num_items: int = DEFAULT_NUM_ITEMS) -> Set[int]:
        """
        Return the set of item IDs {i | n_{k, i} == 0} that have zero interactions
        on this client. These represent cold-start items to be anchored in Phase 1 & 3.

        Parameters
        ----------
        num_items : int
            Total number of items in global catalog (default: 1682).

        Returns
        -------
        Set[int]
            Set of unobserved item IDs in range [0, num_items - 1].
        """
        counts = self.get_interaction_counts(num_items=num_items)
        missing_indices = np.where(counts == 0)[0]
        return set(int(i) for i in missing_indices)

    def load_sparse_matrix(
        self,
        num_users: int = DEFAULT_NUM_USERS,
        num_items: int = DEFAULT_NUM_ITEMS,
    ) -> sp.csr_matrix:
        """
        Loads ratings into a SciPy CSR sparse matrix of shape (num_users, num_items).
        Uses global 0-based user_id and item_id. Unobserved pairs are 0.0.

        Parameters
        ----------
        num_users : int
            Total number of users in global catalog (default: 943).
        num_items : int
            Total number of items in global catalog (default: 1682).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse interaction matrix of shape (num_users, num_items) with float32 ratings.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, item_id, rating FROM ratings;")
            rows = cursor.fetchall()

        if not rows:
            return sp.csr_matrix((num_users, num_items), dtype=np.float32)

        u_ids = np.empty(len(rows), dtype=np.int32)
        i_ids = np.empty(len(rows), dtype=np.int32)
        r_vals = np.empty(len(rows), dtype=np.float32)

        for idx, (u, i, r) in enumerate(rows):
            u_ids[idx] = u
            i_ids[idx] = i
            r_vals[idx] = r

        if u_ids.max() >= num_users or u_ids.min() < 0:
            raise IndexError(f"User ID out of bounds [0, {num_users - 1}]: min={u_ids.min()}, max={u_ids.max()}")
        if i_ids.max() >= num_items or i_ids.min() < 0:
            raise IndexError(f"Item ID out of bounds [0, {num_items - 1}]: min={i_ids.min()}, max={i_ids.max()}")

        matrix = sp.csr_matrix((r_vals, (u_ids, i_ids)), shape=(num_users, num_items), dtype=np.float32)
        return matrix

    def get_train_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (users, items, ratings) as 1D numpy arrays for vectorized SGD shuffling.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            user_ids (int32), item_ids (int32), ratings (float32).
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, item_id, rating FROM ratings;")
            rows = cursor.fetchall()

        n = len(rows)
        users = np.empty(n, dtype=np.int32)
        items = np.empty(n, dtype=np.int32)
        ratings = np.empty(n, dtype=np.float32)

        for idx, (u, i, r) in enumerate(rows):
            users[idx] = u
            items[idx] = i
            ratings[idx] = r

        return users, items, ratings

    def __repr__(self) -> str:
        return f"ClientDataLoader(db_path='{self.db_path}', ratings={self.get_total_ratings()})"
