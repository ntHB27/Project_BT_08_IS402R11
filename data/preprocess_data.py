"""
MovieLens 100K Preprocessing, 0-Based Remapping, 80/20 Stratified Split,
and 4-Client Non-IID Horizontal Partitioning for Fed-LCCF.

This module prepares all data assets for the Federated Leiden Community
Collaborative Filtering (Fed-LCCF) prototype:
1. Ingests raw MovieLens 100K files (u.data, u.item, u.genre).
2. Remaps user_id (0..942) and item_id (0..1681) to 0-based contiguous integers.
3. Generates item metadata mapping (item_mapping.json).
4. Performs a per-user stratified 80/20 train/test split (80,000 train / 20,000 test).
5. Exports global test sets (test.db and test.csv).
6. Partitions the 80,000 train ratings Non-IID across 4 clients:
   - Client 1: 45% = 36,000 ratings
   - Client 2: 30% = 24,000 ratings
   - Client 3: 18% = 14,400 ratings
   - Client 4: 7% = 5,600 ratings (cold-start heavy, ~50% unobserved items)
   - Guaranteed user-disjoint partitioning to protect user factor privacy.
7. Generates indexed SQLite databases (client_1.db .. client_4.db) with WAL mode.
"""

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, milp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_data")

TOTAL_USERS = 943
TOTAL_ITEMS = 1682
TOTAL_RATINGS = 100000
TRAIN_RATINGS = 80000
TEST_RATINGS = 20000

CLIENT_RATIOS = {1: 0.45, 2: 0.30, 3: 0.18, 4: 0.07}
CLIENT_TARGET_COUNTS = {1: 36000, 2: 24000, 3: 14400, 4: 5600}


def find_default_raw_dir() -> Path:
    """Locate the raw ml-100k directory relative to script or workspace."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "ml-100k",
        Path("ml-100k").resolve(),
        Path("../ml-100k").resolve(),
        Path("../../ml-100k").resolve(),
    ]
    for c in candidates:
        if c.exists() and (c / "u.data").exists():
            return c
    raise FileNotFoundError("Could not find raw ml-100k directory with u.data")


def load_raw_data(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load raw u.data, u.item, and u.genre files.
    Validates integrity and performs 0-based remapping.
    """
    data_path = raw_dir / "u.data"
    item_path = raw_dir / "u.item"
    genre_path = raw_dir / "u.genre"

    logger.info("Loading raw ratings from %s", data_path)
    df_ratings = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )

    if len(df_ratings) != TOTAL_RATINGS:
        raise ValueError(f"Expected {TOTAL_RATINGS} ratings, got {len(df_ratings)}")

    raw_user_min, raw_user_max = df_ratings["user_id"].min(), df_ratings["user_id"].max()
    raw_item_min, raw_item_max = df_ratings["item_id"].min(), df_ratings["item_id"].max()

    if raw_user_min != 1 or raw_user_max != TOTAL_USERS:
        raise ValueError(f"Raw user_id range invalid: [{raw_user_min}, {raw_user_max}]")
    if raw_item_min != 1 or raw_item_max != TOTAL_ITEMS:
        raise ValueError(f"Raw item_id range invalid: [{raw_item_min}, {raw_item_max}]")

    # 0-based remapping
    df_ratings["user_id"] = df_ratings["user_id"] - 1
    df_ratings["item_id"] = df_ratings["item_id"] - 1

    assert df_ratings["user_id"].min() == 0 and df_ratings["user_id"].max() == TOTAL_USERS - 1
    assert df_ratings["item_id"].min() == 0 and df_ratings["item_id"].max() == TOTAL_ITEMS - 1

    # Load genres
    genres: List[str] = []
    with open(genre_path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if line:
                name, _ = line.split("|")
                genres.append(name)

    # Load items
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genres
    df_items = pd.read_csv(
        item_path,
        sep="|",
        header=None,
        names=item_cols,
        encoding="latin-1",
    )
    df_items["item_id"] = df_items["item_id"] - 1

    logger.info("Successfully loaded and remapped 0-based data: %d ratings, %d users, %d items",
                len(df_ratings), TOTAL_USERS, TOTAL_ITEMS)
    return df_ratings, df_items, genres


def generate_item_mapping(df_items: pd.DataFrame, genres: List[str], output_path: Path) -> None:
    """Generate and save item_mapping.json for all 1,682 items."""
    mapping: Dict[str, Dict] = {}
    for _, row in df_items.iterrows():
        item_id = int(row["item_id"])
        item_genres = [g for g in genres if row[g] == 1]
        release_date = str(row["release_date"]) if pd.notna(row["release_date"]) else ""
        title = str(row["title"])

        mapping[str(item_id)] = {
            "raw_item_id": item_id + 1,
            "title": title,
            "release_date": release_date,
            "genres": item_genres,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.info("Saved item metadata mapping to %s (%d items)", output_path, len(mapping))


def perform_stratified_split(df_ratings: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user stratified 80/20 train/test split.
    Guarantees that every user has both train and test interactions.
    Produces exactly 80,000 train ratings and 20,000 test ratings.
    """
    train_list: List[pd.DataFrame] = []
    test_list: List[pd.DataFrame] = []

    for _, grp in df_ratings.groupby("user_id"):
        n = len(grp)
        n_train = int(np.round(n * 0.8))
        shuffled = grp.sample(frac=1.0, random_state=seed)
        train_list.append(shuffled.iloc[:n_train])
        test_list.append(shuffled.iloc[n_train:])

    train_df = pd.concat(train_list).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    test_df = pd.concat(test_list).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    if len(train_df) != TRAIN_RATINGS:
        raise ValueError(f"Train split size mismatch: expected {TRAIN_RATINGS}, got {len(train_df)}")
    if len(test_df) != TEST_RATINGS:
        raise ValueError(f"Test split size mismatch: expected {TEST_RATINGS}, got {len(test_df)}")

    assert train_df["user_id"].nunique() == TOTAL_USERS, "Train missing some users"
    assert test_df["user_id"].nunique() == TOTAL_USERS, "Test missing some users"

    logger.info("Stratified 80/20 split completed: 80,000 train ratings, 20,000 test ratings")
    return train_df, test_df


def partition_train_non_iid(
    train_df: pd.DataFrame,
    df_items: pd.DataFrame,
    genres: List[str],
) -> Dict[int, pd.DataFrame]:
    """
    Horizontally partition the 80,000 train ratings across 4 clients.
    - Exact target ratios: C1=45% (36,000), C2=30% (24,000), C3=18% (14,400), C4=7% (5,600).
    - Strict user-disjoint assignment to preserve user factor privacy.
    - Genre preference affinity to create realistic Non-IID distribution:
      * C4: Action & Sci-Fi (cold-start heavy, ~50% catalog missing)
      * C3: Comedy, Romance & Children's
      * C2: Thriller, Crime & Mystery
      * C1: Drama & mainstream
    - Solved via exact Mixed-Integer Linear Programming (MILP).
    """
    logger.info("Partitioning 80,000 train ratings across 4 clients with genre-skewed Non-IID affinity...")

    # Calculate user genre profiles from training set
    merged = train_df.merge(df_items[["item_id"] + genres], on="item_id")
    user_genre = merged.groupby("user_id")[genres].sum()
    user_genre_ratio = user_genre.div(user_genre.sum(axis=1), axis=0)

    user_counts = train_df.groupby("user_id").size().values  # shape (943,)
    targets = [CLIENT_TARGET_COUNTS[1], CLIENT_TARGET_COUNTS[2], CLIENT_TARGET_COUNTS[3], CLIENT_TARGET_COUNTS[4]]

    # Affinity scores for each client
    aff_c4 = (user_genre_ratio.get("Action", 0.0) + user_genre_ratio.get("Sci-Fi", 0.0)).values
    aff_c3 = (user_genre_ratio.get("Comedy", 0.0) + user_genre_ratio.get("Romance", 0.0) + user_genre_ratio.get("Children's", 0.0)).values
    aff_c2 = (user_genre_ratio.get("Thriller", 0.0) + user_genre_ratio.get("Crime", 0.0) + user_genre_ratio.get("Mystery", 0.0)).values
    aff_c1 = (user_genre_ratio.get("Drama", 0.0) + user_genre_ratio.get("unknown", 0.0)).values

    affinities = np.stack([aff_c1, aff_c2, aff_c3, aff_c4], axis=1)  # shape (943, 4)

    # Solve MILP
    c = -affinities.flatten()
    A_rows, b_l, b_u = [], [], []

    # Constraint 1: Each user assigned to exactly 1 client
    for u in range(TOTAL_USERS):
        row = np.zeros((TOTAL_USERS, 4))
        row[u, :] = 1.0
        A_rows.append(row.flatten())
        b_l.append(1.0)
        b_u.append(1.0)

    # Constraint 2: Client capacity equals exact target rating count
    for k in range(4):
        row = np.zeros((TOTAL_USERS, 4))
        row[:, k] = user_counts
        A_rows.append(row.flatten())
        b_l.append(targets[k])
        b_u.append(targets[k])

    A = np.array(A_rows)
    constraints = LinearConstraint(A, b_l, b_u)
    bounds = Bounds(0, 1)
    integrality = np.ones(c.shape[0])

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    if not res.success:
        raise RuntimeError(f"MILP solver failed to partition ratings: status={res.status}, msg={res.message}")

    x = np.round(res.x).reshape((TOTAL_USERS, 4)).astype(int)

    client_dfs: Dict[int, pd.DataFrame] = {}
    assigned_user_sets = []

    for k in range(4):
        client_id = k + 1
        user_indices = np.where(x[:, k] == 1)[0]
        assigned_user_sets.append(set(user_indices))
        client_sub = train_df[train_df["user_id"].isin(user_indices)].copy()
        client_sub = client_sub.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        cnt = len(client_sub)
        expected_cnt = CLIENT_TARGET_COUNTS[client_id]
        if cnt != expected_cnt:
            raise ValueError(f"Client {client_id} rating count {cnt} != expected {expected_cnt}")

        obs_items = client_sub["item_id"].nunique()
        missing_items = TOTAL_ITEMS - obs_items
        missing_pct = (missing_items / TOTAL_ITEMS) * 100

        logger.info(
            "Client %d: %d users, %d ratings (%.1f%% target), %d items observed, %d missing items (%.1f%% catalog)",
            client_id, len(user_indices), cnt, (cnt / TRAIN_RATINGS) * 100,
            obs_items, missing_items, missing_pct,
        )
        client_dfs[client_id] = client_sub

    # Verify disjointness
    for i in range(4):
        for j in range(i + 1, 4):
            overlap = assigned_user_sets[i].intersection(assigned_user_sets[j])
            if overlap:
                raise ValueError(f"User overlap detected between Client {i+1} and Client {j+1}: {overlap}")

    total_assigned_users = sum(len(s) for s in assigned_user_sets)
    if total_assigned_users != TOTAL_USERS:
        raise ValueError(f"Total assigned users {total_assigned_users} != {TOTAL_USERS}")

    return client_dfs


def create_sqlite_db(df: pd.DataFrame, db_path: Path) -> None:
    """
    Creates an optimized SQLite database containing the ratings table,
    creates required B-tree indexes, and configures WAL journal mode.
    """
    if db_path.exists():
        db_path.unlink()
    # Remove associated WAL and SHM files if present
    wal_file = Path(str(db_path) + "-wal")
    shm_file = Path(str(db_path) + "-shm")
    if wal_file.exists():
        wal_file.unlink()
    if shm_file.exists():
        shm_file.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Performance PRAGMAs
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
    cursor.execute("PRAGMA temp_store = MEMORY;")

    # Schema
    cursor.execute("""
        CREATE TABLE ratings (
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            PRIMARY KEY (user_id, item_id)
        );
    """)

    # Bulk insert with explicit native Python int and float types
    records = [
        (int(row[0]), int(row[1]), float(row[2]), int(row[3]))
        for row in df[["user_id", "item_id", "rating", "timestamp"]].itertuples(index=False)
    ]
    cursor.executemany(
        "INSERT INTO ratings (user_id, item_id, rating, timestamp) VALUES (?, ?, ?, ?);",
        records,
    )

    # B-Tree Indexes
    cursor.execute("CREATE INDEX idx_ratings_user_id ON ratings(user_id);")
    cursor.execute("CREATE INDEX idx_ratings_item_id ON ratings(item_id);")
    cursor.execute("CREATE INDEX idx_ratings_user_item ON ratings(user_id, item_id);")

    conn.commit()

    # Verification
    row_count = cursor.execute("SELECT COUNT(*) FROM ratings;").fetchone()[0]
    integrity = cursor.execute("PRAGMA integrity_check;").fetchone()[0]
    journal = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    if integrity != "ok":
        raise RuntimeError(f"Database integrity check failed for {db_path}: {integrity}")
    if row_count != len(df):
        raise ValueError(f"Row count mismatch in {db_path}: expected {len(df)}, got {row_count}")

    logger.info("Successfully created %s (rows: %d, journal: %s, integrity: %s)",
                db_path.name, row_count, journal, integrity)


def verify_database_query_plans(db_path: Path) -> None:
    """Verify that indexes are utilized as intended in SQLite query plans."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Test user query plan
    user_plan = list(cursor.execute("EXPLAIN QUERY PLAN SELECT item_id, rating FROM ratings WHERE user_id = ?;", (0,)))
    user_plan_str = " ".join(str(p) for p in user_plan)
    assert "idx_ratings_user_id" in user_plan_str or "idx_ratings_user_item" in user_plan_str, \
        f"User index not utilized in {db_path}: {user_plan}"

    # Test item grouping query plan
    item_plan = list(cursor.execute("EXPLAIN QUERY PLAN SELECT item_id, COUNT(*) FROM ratings GROUP BY item_id;"))
    item_plan_str = " ".join(str(p) for p in item_plan)
    assert "idx_ratings_item_id" in item_plan_str, \
        f"Item index not utilized in {db_path}: {item_plan}"

    conn.close()


def run_pipeline(raw_dir: Path, output_dir: Path, seed: int = 42) -> None:
    """Execute end-to-end data preparation and partitioning pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Fed-LCCF Data Preparation pipeline...")
    logger.info("Raw data directory: %s", raw_dir)
    logger.info("Output directory: %s", output_dir)

    # 1. Ingestion & Remapping
    df_ratings, df_items, genres = load_raw_data(raw_dir)

    # 2. Item Mapping JSON
    item_mapping_path = output_dir / "item_mapping.json"
    generate_item_mapping(df_items, genres, item_mapping_path)

    # 3. Stratified 80/20 Train/Test Split
    train_df, test_df = perform_stratified_split(df_ratings, seed=seed)

    # 4. Save Test Set (test.csv and test.db)
    test_csv_path = output_dir / "test.csv"
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Saved test evaluation set to %s (rows: %d)", test_csv_path.name, len(test_df))

    test_db_path = output_dir / "test.db"
    create_sqlite_db(test_df, test_db_path)
    verify_database_query_plans(test_db_path)

    # 5. Non-IID Horizontal Partitioning
    client_dfs = partition_train_non_iid(train_df, df_items, genres)

    # 6. Create Client SQLite Databases
    for client_id in range(1, 5):
        db_path = output_dir / f"client_{client_id}.db"
        create_sqlite_db(client_dfs[client_id], db_path)
        verify_database_query_plans(db_path)

    logger.info("=== Fed-LCCF Data Preparation Completed Successfully ===")
    logger.info("Outputs:")
    logger.info("  - %s (%d items)", item_mapping_path, len(df_items))
    logger.info("  - %s (%d rows)", test_csv_path, len(test_df))
    logger.info("  - %s (%d rows)", test_db_path, len(test_df))
    for client_id in range(1, 5):
        p = output_dir / f"client_{client_id}.db"
        logger.info("  - %s (%d rows)", p, len(client_dfs[client_id]))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Fed-LCCF MovieLens 100K Preprocessing & SQLite Partitioning")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Path to raw ml-100k directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Target output directory for databases and mappings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    raw_dir = args.raw_data_dir if args.raw_data_dir else find_default_raw_dir()
    run_pipeline(raw_dir=raw_dir, output_dir=args.output_dir, seed=args.seed)
