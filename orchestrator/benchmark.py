"""
Fed-LCCF Comparative Benchmark & Evaluation Engine.

Executes and evaluates the 3 experimental scenarios across 20 communication rounds:
1. Scenario 1: Local Only
   - 4 clients train independently on their local SQLite partitions for 20 rounds.
   - Zero communication, zero factor sharing, zero server aggregation.
   - Evaluated on test set using local factors and unobserved item fallback.
2. Scenario 2: Vanilla FedAvg
   - 4 clients participate in Federated Learning for 20 rounds.
   - Standard count-weighted FedAvg aggregation across all 20 rounds (gamma=0.0, no Leiden clustering).
   - Global factors broadcast to clients each round.
3. Scenario 3: Fed-LCCF
   - 4 clients participate in Federated Learning for 20 rounds.
   - Rounds 1–5: Vanilla FedAvg aggregation.
   - Post-Round 5: Leiden community detection on item factors (k=20).
   - Rounds 6–20: Community-aware FedAvg with community shrinkage (gamma=10.0),
     dynamic centroid updating, and on-device centroid anchoring.

Outputs:
- benchmark_results.csv: exactly 60 data rows (3 scenarios x 20 rounds).
- convergence_plot.png: high-resolution 300 DPI publication figure.
"""

import argparse
import csv
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Set headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure prototype root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROTOTYPE_DIR = _CURRENT_DIR.parent
_WORKSPACE_ROOT = _PROTOTYPE_DIR.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from client_ml.local_mf import (
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITEMS,
    DEFAULT_NUM_USERS,
    MatrixFactorization,
)
from cloud_server.aggregator import (
    DEFAULT_GAMMA,
    FederatedAggregator,
)
from data.client_data_loader import ClientDataLoader
from orchestrator.evaluator import Evaluator

logger = logging.getLogger("fedlccf.benchmark")

DEFAULT_ROUNDS = 20
DEFAULT_EPOCHS = 1
CSV_HEADER = [
    "scenario",
    "round",
    "global_rmse",
    "global_mae",
    "client_1_rmse",
    "client_1_mae",
    "client_2_rmse",
    "client_2_mae",
    "client_3_rmse",
    "client_3_mae",
    "client_4_rmse",
    "client_4_mae",
    "runtime_seconds",
]


class BenchmarkRunner:
    """
    Coordinates execution of comparative evaluation across Local Only, Vanilla FedAvg, and Fed-LCCF.

    Parameters
    ----------
    data_dir : Optional[Union[str, Path]]
        Directory containing client SQLite databases and test.db.
    rounds : int
        Number of communication/training rounds (default: 20).
    epochs_per_round : int
        Local training epochs per round (default: 1).
    learning_rate : float
        SGD learning rate (default: 0.01).
    latent_dim : int
        Embedding dimensionality (default: 32).
    gamma : float
        Phase 3 Leiden community shrinkage parameter for Fed-LCCF (default: 10.0).
    seed : int
        Base random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        rounds: int = DEFAULT_ROUNDS,
        epochs_per_round: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        gamma: float = DEFAULT_GAMMA,
        seed: int = 42,
    ) -> None:
        self.prototype_dir = _PROTOTYPE_DIR
        self.data_dir = Path(data_dir).resolve() if data_dir else self.prototype_dir / "data"
        self.rounds = int(rounds)
        self.epochs = int(epochs_per_round)
        self.lr = float(learning_rate)
        self.latent_dim = int(latent_dim)
        self.gamma = float(gamma)
        self.seed = int(seed)

        # Initialize Evaluator
        self.evaluator = Evaluator(
            test_db_path=self.data_dir / "test.db",
            data_dir=self.data_dir,
        )

        # Load Client Data Loaders
        self.loaders: Dict[int, ClientDataLoader] = {}
        for cid in range(1, 5):
            db_p = self.data_dir / f"client_{cid}.db"
            if not db_p.exists():
                raise FileNotFoundError(f"Missing client database: {db_p}")
            self.loaders[cid] = ClientDataLoader(db_p)

    def _init_client_models(self) -> Dict[int, MatrixFactorization]:
        """Instantiate fresh MatrixFactorization models for Clients 1 to 4."""
        models: Dict[int, MatrixFactorization] = {}
        for cid in range(1, 5):
            models[cid] = MatrixFactorization(
                num_users=DEFAULT_NUM_USERS,
                num_items=DEFAULT_NUM_ITEMS,
                latent_dim=self.latent_dim,
                learning_rate=self.lr,
                client_id=cid,
                random_state=self.seed + cid,
            )
            # Initialize interaction counts
            counts = self.loaders[cid].get_interaction_counts(DEFAULT_NUM_ITEMS)
            models[cid].set_interaction_counts(counts)
            # Phase 1: Initialize unobserved items with mean of observed items
            models[cid].initialize_unobserved_items_mean()
        return models

    def run_scenario_local_only(self) -> List[Dict[str, Any]]:
        """
        Execute Scenario 1: Local Only.
        4 clients train completely independently on their local data for 20 rounds
        without any central server communication or item factor sharing.
        """
        logger.info("Executing Scenario 1: Local Only (%d rounds, %d epochs/round)...", self.rounds, self.epochs)
        models = self._init_client_models()
        records: List[Dict[str, Any]] = []

        for r in range(1, self.rounds + 1):
            t_start = time.time()

            # Train each client independently
            for cid in range(1, 5):
                u, i, ratings = self.loaders[cid].get_train_samples()
                models[cid].fit(u, i, ratings, epochs=self.epochs, shuffle=True)

            t_round = time.time() - t_start

            # Evaluate on test.db
            metrics = self.evaluator.evaluate_models(models, global_item_factors=None, is_local_only=True)

            row = {
                "scenario": "Local Only",
                "round": r,
                "global_rmse": metrics["global_rmse"],
                "global_mae": metrics["global_mae"],
                "client_1_rmse": metrics["client_1_rmse"],
                "client_1_mae": metrics["client_1_mae"],
                "client_2_rmse": metrics["client_2_rmse"],
                "client_2_mae": metrics["client_2_mae"],
                "client_3_rmse": metrics["client_3_rmse"],
                "client_3_mae": metrics["client_3_mae"],
                "client_4_rmse": metrics["client_4_rmse"],
                "client_4_mae": metrics["client_4_mae"],
                "runtime_seconds": t_round,
            }
            records.append(row)
            logger.info(
                "Local Only [Round %02d/%02d] Global RMSE: %.4f | Client 4 RMSE: %.4f | Time: %.2fs",
                r, self.rounds, metrics["global_rmse"], metrics["client_4_rmse"], t_round
            )

        return records

    def run_scenario_vanilla_fedavg(self) -> List[Dict[str, Any]]:
        """
        Execute Scenario 2: Vanilla FedAvg.
        4 clients participate in Federated Learning for 20 rounds with standard
        count-weighted FedAvg aggregation across all rounds (no Leiden clustering, gamma=0.0).
        """
        logger.info("Executing Scenario 2: Vanilla FedAvg (%d rounds, %d epochs/round)...", self.rounds, self.epochs)
        models = self._init_client_models()
        records: List[Dict[str, Any]] = []

        # Previous global factors
        global_factors = np.zeros((DEFAULT_NUM_ITEMS, self.latent_dim), dtype=np.float32)

        for r in range(1, self.rounds + 1):
            t_start = time.time()

            # Local SGD training on each client
            uploads: Dict[int, Dict[str, Any]] = {}
            for cid in range(1, 5):
                u, i, ratings = self.loaders[cid].get_train_samples()
                models[cid].fit(u, i, ratings, epochs=self.epochs, shuffle=True)
                uploads[cid] = models[cid].get_upload_payload_dict(round_num=r, client_id=cid)

            # Central Server Count-Weighted FedAvg Aggregation (gamma=0.0, no Leiden)
            factors_list = [np.asarray(uploads[cid]["item_factors"], dtype=np.float32) for cid in sorted(uploads.keys())]
            counts_list = [np.asarray(uploads[cid]["item_counts"], dtype=np.float32) for cid in sorted(uploads.keys())]

            all_factors = np.stack(factors_list, axis=0)  # (4, 1682, 32)
            all_counts = np.stack(counts_list, axis=0)    # (4, 1682)
            total_counts = np.sum(all_counts, axis=0)     # (1682,)

            observed_mask = total_counts > 0.0
            weighted_sum = np.sum(all_counts[:, :, None] * all_factors, axis=0)

            new_global_factors = global_factors.copy()
            new_global_factors[observed_mask] = weighted_sum[observed_mask] / total_counts[observed_mask, None]
            global_factors = new_global_factors.astype(np.float32)

            # Broadcast global factors back to clients
            for cid in range(1, 5):
                models[cid].set_item_factors(global_factors)

            t_round = time.time() - t_start

            # Evaluate on test.db
            metrics = self.evaluator.evaluate_models(models, global_item_factors=global_factors, is_local_only=False)

            row = {
                "scenario": "Vanilla FedAvg",
                "round": r,
                "global_rmse": metrics["global_rmse"],
                "global_mae": metrics["global_mae"],
                "client_1_rmse": metrics["client_1_rmse"],
                "client_1_mae": metrics["client_1_mae"],
                "client_2_rmse": metrics["client_2_rmse"],
                "client_2_mae": metrics["client_2_mae"],
                "client_3_rmse": metrics["client_3_rmse"],
                "client_3_mae": metrics["client_3_mae"],
                "client_4_rmse": metrics["client_4_rmse"],
                "client_4_mae": metrics["client_4_mae"],
                "runtime_seconds": t_round,
            }
            records.append(row)
            logger.info(
                "Vanilla FedAvg [Round %02d/%02d] Global RMSE: %.4f | Client 4 RMSE: %.4f | Time: %.2fs",
                r, self.rounds, metrics["global_rmse"], metrics["client_4_rmse"], t_round
            )

        return records

    def run_scenario_fed_lccf(self) -> List[Dict[str, Any]]:
        """
        Execute Scenario 3: Fed-LCCF.
        4 clients participate in Federated Learning for 20 rounds:
        - Rounds 1–5: Vanilla FedAvg.
        - Post-Round 5: Leiden community detection on global item factors (k=20).
        - Rounds 6–20: Community-aware FedAvg with community shrinkage (gamma=10.0),
          dynamic centroid updating, and on-device centroid anchoring.
        """
        logger.info("Executing Scenario 3: Fed-LCCF (%d rounds, %d epochs/round, gamma=%.1f)...", self.rounds, self.epochs, self.gamma)
        models = self._init_client_models()
        aggregator = FederatedAggregator(
            num_items=DEFAULT_NUM_ITEMS,
            latent_dim=self.latent_dim,
            gamma=self.gamma,
            random_state=self.seed,
        )
        records: List[Dict[str, Any]] = []

        for r in range(1, self.rounds + 1):
            t_start = time.time()

            # In Phase 3 (Rounds 6-20), update clients with community map and centroids
            if r >= 6 and aggregator.community_map is not None and aggregator.centroids is not None:
                for cid in range(1, 5):
                    models[cid].set_community_info(
                        community_map=aggregator.community_map,
                        centroids=aggregator.centroids,
                        activate_anchoring=True,
                    )
                    models[cid].anchor_unobserved_items_to_centroids()

            # Local training on each client
            uploads: Dict[int, Dict[str, Any]] = {}
            for cid in range(1, 5):
                u, i, ratings = self.loaders[cid].get_train_samples()
                models[cid].fit(u, i, ratings, epochs=self.epochs, shuffle=True)
                uploads[cid] = models[cid].get_upload_payload_dict(round_num=r, client_id=cid)

            # Central Aggregation (Vanilla FedAvg for r<=5; Leiden clustering at r=5; Shrinkage for r>=6)
            global_factors = aggregator.aggregate_round(r, uploads)

            # Broadcast global factors back to all clients
            for cid in range(1, 5):
                models[cid].set_item_factors(global_factors)

            t_round = time.time() - t_start

            # Evaluate on test.db
            metrics = self.evaluator.evaluate_models(models, global_item_factors=global_factors, is_local_only=False)

            row = {
                "scenario": "Fed-LCCF",
                "round": r,
                "global_rmse": metrics["global_rmse"],
                "global_mae": metrics["global_mae"],
                "client_1_rmse": metrics["client_1_rmse"],
                "client_1_mae": metrics["client_1_mae"],
                "client_2_rmse": metrics["client_2_rmse"],
                "client_2_mae": metrics["client_2_mae"],
                "client_3_rmse": metrics["client_3_rmse"],
                "client_3_mae": metrics["client_3_mae"],
                "client_4_rmse": metrics["client_4_rmse"],
                "client_4_mae": metrics["client_4_mae"],
                "runtime_seconds": t_round,
            }
            records.append(row)
            phase_tag = f"Phase {aggregator.current_phase}"
            logger.info(
                "Fed-LCCF [%s Round %02d/%02d] Global RMSE: %.4f | Client 4 RMSE: %.4f | Time: %.2fs",
                phase_tag, r, self.rounds, metrics["global_rmse"], metrics["client_4_rmse"], t_round
            )

        return records

    def run_all(self) -> List[Dict[str, Any]]:
        """
        Execute all 3 benchmark scenarios sequentially and return concatenated rows.

        Returns
        -------
        List[Dict[str, Any]]
            Exactly 60 rows (3 scenarios x 20 rounds).
        """
        all_records: List[Dict[str, Any]] = []

        t_total_start = time.time()
        print("=" * 80)
        print("Fed-LCCF Comparative Benchmark: Running 3 Scenarios x 20 Rounds")
        print("=" * 80)

        # 1. Local Only
        print("\n[*] Starting Scenario 1/3: Local Only ...")
        rec_local = self.run_scenario_local_only()
        all_records.extend(rec_local)

        # 2. Vanilla FedAvg
        print("\n[*] Starting Scenario 2/3: Vanilla FedAvg ...")
        rec_fedavg = self.run_scenario_vanilla_fedavg()
        all_records.extend(rec_fedavg)

        # 3. Fed-LCCF
        print("\n[*] Starting Scenario 3/3: Fed-LCCF ...")
        rec_fedlccf = self.run_scenario_fed_lccf()
        all_records.extend(rec_fedlccf)

        total_time = time.time() - t_total_start
        print("\n" + "=" * 80)
        print(f"[PASS] All 3 Scenarios (60 rounds) completed in {total_time:.2f} seconds!")
        print("=" * 80)

        return all_records

    def save_csv(self, records: List[Dict[str, Any]], output_path: Union[str, Path]) -> Path:
        """
        Export benchmark records to CSV.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            List of 60 row dictionaries.
        output_path : Union[str, Path]
            Destination CSV path.

        Returns
        -------
        Path
            Resolved output file path.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        logger.info("Exported %d benchmark rows to %s", len(records), out_p)
        return out_p

    def plot_convergence(
        self,
        records: List[Dict[str, Any]],
        output_plot_path: Union[str, Path],
    ) -> Path:
        """
        Generate high-resolution 300 DPI convergence plot.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Benchmark records list.
        output_plot_path : Union[str, Path]
            Destination path for the PNG plot.

        Returns
        -------
        Path
            Resolved path of the generated plot.
        """
        out_p = Path(output_plot_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Organize data by scenario
        data: Dict[str, Dict[str, List[float]]] = {
            "Local Only": {"rounds": [], "global_rmse": [], "client_4_rmse": []},
            "Vanilla FedAvg": {"rounds": [], "global_rmse": [], "client_4_rmse": []},
            "Fed-LCCF": {"rounds": [], "global_rmse": [], "client_4_rmse": []},
        }

        for row in records:
            scen = row["scenario"]
            if scen in data:
                data[scen]["rounds"].append(row["round"])
                data[scen]["global_rmse"].append(row["global_rmse"])
                data[scen]["client_4_rmse"].append(row["client_4_rmse"])

        # Configure Matplotlib styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        # Color and style definitions
        styles = {
            "Local Only": {"color": "#6c757d", "linestyle": "--", "marker": "o", "label": "Local Only (No FL)"},
            "Vanilla FedAvg": {"color": "#1f77b4", "linestyle": "-", "marker": "s", "label": "Vanilla FedAvg (gamma=0.0)"},
            "Fed-LCCF": {"color": "#d62728", "linestyle": "-", "marker": "^", "label": "Fed-LCCF (Leiden + Shrinkage gamma=10.0)"},
        }

        # ----------------------------------------------------------------------
        # Subplot 1: Global Test RMSE vs. Rounds
        # ----------------------------------------------------------------------
        for scen, style in styles.items():
            if scen in data and len(data[scen]["rounds"]) > 0:
                ax1.plot(
                    data[scen]["rounds"],
                    data[scen]["global_rmse"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=2.0,
                    label=style["label"],
                    alpha=0.9,
                )

        # Vertical transition marker at Round 5
        ax1.axvline(x=5, color="#d62728", linestyle="--", linewidth=2.0, alpha=0.85, label="Start of Leiden Shrinkage (Round 5 -> 6)")
        ax1.text(5.3, 1.65, "Start of Leiden Shrinkage\n(Round 5 -> 6)", color="#d62728", fontsize=9, fontweight="bold", va="center")
        ax1.set_title("Global Test RMSE Convergence (All 943 Users)", fontweight="bold", pad=12)
        ax1.set_xlabel("Communication Round", fontweight="bold")
        ax1.set_ylabel("Test Set RMSE", fontweight="bold")
        ax1.set_xticks(range(1, 21, 2))
        ax1.set_xlim(0.5, 23.5)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")

        # Annotate final round values with offsets to avoid collision
        offsets = {"Local Only": 0.020, "Vanilla FedAvg": -0.010, "Fed-LCCF": -0.040}
        for scen, style in styles.items():
            if scen in data and len(data[scen]["global_rmse"]) >= 20:
                val = data[scen]["global_rmse"][-1]
                y_pos = val + offsets.get(scen, 0.0)
                ax1.annotate(
                    f"{scen}: {val:.4f}",
                    xy=(20, val),
                    xytext=(20.5, y_pos),
                    arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                    fontweight="bold",
                    color=style["color"],
                    fontsize=8.5,
                    verticalalignment="center",
                )

        # ----------------------------------------------------------------------
        # Subplot 2: Client 4 (Cold-Start Partition) Test RMSE vs. Rounds
        # ----------------------------------------------------------------------
        for scen, style in styles.items():
            if scen in data and len(data[scen]["rounds"]) > 0:
                ax2.plot(
                    data[scen]["rounds"],
                    data[scen]["client_4_rmse"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=2.0,
                    label=style["label"],
                    alpha=0.9,
                )

        ax2.axvline(x=5, color="#d62728", linestyle="--", linewidth=2.0, alpha=0.85, label="Start of Leiden Shrinkage (Round 5 -> 6)")
        ax2.text(5.3, 1.65, "Start of Leiden Shrinkage\n(Round 5 -> 6)", color="#d62728", fontsize=9, fontweight="bold", va="center")
        ax2.set_title("Client 4 Test RMSE (Cold-Start Partition, 51.8% Missing Items)", fontweight="bold", pad=12)
        ax2.set_xlabel("Communication Round", fontweight="bold")
        ax2.set_ylabel("Client 4 Test Set RMSE", fontweight="bold")
        ax2.set_xticks(range(1, 21, 2))
        ax2.set_xlim(0.5, 23.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")

        # Annotate final round values
        for scen, style in styles.items():
            if scen in data and len(data[scen]["client_4_rmse"]) >= 20:
                val = data[scen]["client_4_rmse"][-1]
                y_pos = val + offsets.get(scen, 0.0)
                ax2.annotate(
                    f"{scen}: {val:.4f}",
                    xy=(20, val),
                    xytext=(20.5, y_pos),
                    arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
                    fontweight="bold",
                    color=style["color"],
                    fontsize=8.5,
                    verticalalignment="center",
                )

        fig.suptitle(
            "Fed-LCCF Comparative Evaluation: Convergence Across 20 Rounds (MovieLens 100K)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Generated 300 DPI convergence plot: %s", out_p)
        return out_p

    @classmethod
    def plot_from_csv(cls, csv_path: Union[str, Path], output_plot_path: Union[str, Path]) -> Path:
        """Render convergence plot directly from existing CSV records."""
        records: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                records.append({
                    "scenario": r["scenario"],
                    "round": int(r["round"]),
                    "global_rmse": float(r["global_rmse"]),
                    "global_mae": float(r["global_mae"]),
                    "client_1_rmse": float(r["client_1_rmse"]),
                    "client_1_mae": float(r["client_1_mae"]),
                    "client_2_rmse": float(r["client_2_rmse"]),
                    "client_2_mae": float(r["client_2_mae"]),
                    "client_3_rmse": float(r["client_3_rmse"]),
                    "client_3_mae": float(r["client_3_mae"]),
                    "client_4_rmse": float(r["client_4_rmse"]),
                    "client_4_mae": float(r["client_4_mae"]),
                    "runtime_seconds": float(r["runtime_seconds"]),
                })
        dummy = cls.__new__(cls)
        return dummy.plot_convergence(records, output_plot_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for benchmark."""
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.benchmark",
        description="Fed-LCCF Comparative Benchmark & Convergence Plotter",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Number of rounds (default: 20)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Local epochs per round (default: 1)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate (default: 0.01)")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM, help="Latent dim d (default: 32)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Phase 3 shrinkage gamma (default: 10.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing client SQLite databases")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV path (default: benchmark_results.csv)")
    parser.add_argument("--output-plot", type=str, default=None, help="Output PNG plot path (default: convergence_plot.png)")
    parser.add_argument("--skip-plot", action="store_true", help="Skip generating the PNG plot")
    parser.add_argument("--plot-only", action="store_true", help="Re-generate plot from existing CSV without re-running training")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    prototype_dir = _PROTOTYPE_DIR
    csv_path = Path(args.output_csv).resolve() if args.output_csv else prototype_dir / "benchmark_results.csv"
    plot_path = Path(args.output_plot).resolve() if args.output_plot else prototype_dir / "convergence_plot.png"

    if args.plot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot generate plot, CSV missing at: {csv_path}")
        BenchmarkRunner.plot_from_csv(csv_path, plot_path)
        print(f"[+] Re-generated 300 DPI convergence plot from {csv_path}: {plot_path}")
        return 0

    runner = BenchmarkRunner(
        data_dir=args.data_dir,
        rounds=args.rounds,
        epochs_per_round=args.epochs,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        gamma=args.gamma,
        seed=args.seed,
    )

    records = runner.run_all()

    # Save CSV
    runner.save_csv(records, csv_path)
    print(f"[+] Saved benchmark CSV: {csv_path} ({len(records)} rows)")

    # Generate Plot
    if not args.skip_plot:
        runner.plot_convergence(records, plot_path)
        print(f"[+] Saved 300 DPI convergence plot: {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
