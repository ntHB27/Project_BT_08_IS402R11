"""
Federated Client Agent for Fed-LCCF.

Simulates a local client worker participating in the 20-round Federated Learning protocol:
1. Loads client-specific SQLite partition ratings via ClientDataLoader.
2. Manages local MatrixFactorization instance with adaptive L2 regularization.
3. Coordinates HTTP communication with central FastAPI aggregation server:
   - Round 1: Pull Round 0 initial global factors (if available) and perform
     Phase 1 observed items mean initialization for unobserved items.
   - Rounds 1-20:
     * Fit local MF for E epochs (default E=5).
     * Upload local item factors V_k and counts n_k via POST /api/v1/fl/upload_factors.
     * Poll / await aggregated global factors from GET /api/v1/fl/get_global_factors/{round}.
     * Update local V_k with global aggregated factors.
     * When Leiden community structures arrive (Phase 3, post-Round 5), activate
       centroid anchoring for unobserved items and SGD anchoring penalty.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

# Ensure workspace root is in sys.path when script is executed directly
_CURRENT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _CURRENT_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import numpy as np

from client_ml.local_mf import (
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITEMS,
    DEFAULT_NUM_USERS,
    MatrixFactorization,
)
from data.client_data_loader import ClientDataLoader

# Optional requests import with urllib fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("FederatedClientAgent")


class FederatedClientAgent:
    """
    Autonomous client worker agent for Federated Collaborative Filtering.

    Parameters
    ----------
    client_id : int
        Client identifier (1..4).
    db_path : Union[str, Path]
        Path to local client SQLite database (e.g. client_1.db).
    server_url : str
        FastAPI server URL (default: http://127.0.0.1:8000).
    rounds : int
        Total federated rounds to participate in (default: 20).
    epochs_per_round : int
        Local SGD training epochs per communication round (default: 5).
    learning_rate : float
        Local SGD learning rate (default: 0.01).
    latent_dim : int
        Latent embedding dimension d (default: 32).
    num_users : int
        Number of users in global catalog (default: 943).
    num_items : int
        Number of items in global catalog (default: 1682).
    random_state : Optional[int]
        Random seed for reproducibility.
    poll_interval : float
        Polling interval in seconds when awaiting server aggregation (default: 1.0).
    poll_timeout : float
        Maximum seconds to wait for server round aggregation before timing out (default: 120.0).
    http_timeout : float
        HTTP request timeout in seconds (default: 30.0).
    """

    def __init__(
        self,
        client_id: int,
        db_path: Union[str, Path],
        server_url: str = "http://127.0.0.1:8000",
        rounds: int = 20,
        epochs_per_round: int = 5,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_users: int = DEFAULT_NUM_USERS,
        num_items: int = DEFAULT_NUM_ITEMS,
        random_state: Optional[int] = 42,
        poll_interval: float = 1.0,
        poll_timeout: float = 120.0,
        http_timeout: float = 30.0,
    ) -> None:
        self.client_id = int(client_id)
        self.db_path = Path(db_path).resolve()
        self.server_url = server_url.rstrip("/")
        self.rounds = int(rounds)
        self.epochs_per_round = int(epochs_per_round)
        self.learning_rate = float(learning_rate)
        self.latent_dim = int(latent_dim)
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.random_state = random_state
        self.poll_interval = float(poll_interval)
        self.poll_timeout = float(poll_timeout)
        self.http_timeout = float(http_timeout)

        # Initialize data loader and load partition
        self.loader = ClientDataLoader(self.db_path)
        self.users, self.items, self.ratings = self.loader.get_train_samples()
        self.interaction_counts = self.loader.get_interaction_counts(num_items=self.num_items)
        self.missing_item_ids = self.loader.get_missing_item_ids(num_items=self.num_items)

        # Initialize local Matrix Factorization model with adaptive lambda_k
        self.model = MatrixFactorization(
            num_users=self.num_users,
            num_items=self.num_items,
            latent_dim=self.latent_dim,
            learning_rate=self.learning_rate,
            client_id=self.client_id,
            random_state=self.random_state,
        )
        self.model.set_interaction_counts(self.interaction_counts)

        # Phase 3 state
        self.community_map: Optional[Dict[str, int]] = None
        self.centroids: Optional[Dict[str, List[float]]] = None

        # Telemetry
        self.round_history: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # Network Layer Helpers
    # --------------------------------------------------------------------------

    def _http_get(self, endpoint: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Perform HTTP GET request with urllib fallback."""
        url = f"{self.server_url}{endpoint}"
        if HAS_REQUESTS:
            try:
                resp = requests.get(url, timeout=self.http_timeout)
                if resp.status_code == 200:
                    return resp.status_code, resp.json()
                return resp.status_code, None
            except Exception as e:
                logger.debug("requests GET %s failed: %s", url, e)
                return 0, None

        # Fallback to urllib
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as response:
                status = response.getcode()
                body = response.read().decode("utf-8")
                return status, json.loads(body)
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception as e:
            logger.debug("urllib GET %s failed: %s", url, e)
            return 0, None

    def _http_post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Perform HTTP POST request with urllib fallback."""
        url = f"{self.server_url}{endpoint}"
        if HAS_REQUESTS:
            try:
                resp = requests.post(url, json=data, timeout=self.http_timeout)
                if resp.status_code in (200, 201):
                    try:
                        return resp.status_code, resp.json()
                    except Exception:
                        return resp.status_code, {}
                return resp.status_code, None
            except Exception as e:
                logger.debug("requests POST %s failed: %s", url, e)
                return 0, None

        # Fallback to urllib
        json_bytes = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=json_bytes,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as response:
                status = response.getcode()
                body = response.read().decode("utf-8")
                return status, json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception as e:
            logger.debug("urllib POST %s failed: %s", url, e)
            return 0, None

    # --------------------------------------------------------------------------
    # Protocol Operations
    # --------------------------------------------------------------------------

    def pull_global_factors(
        self,
        round_num: int,
        poll: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Pull global aggregated factors from the server for round_num.

        Parameters
        ----------
        round_num : int
            Round number to query.
        poll : bool
            Whether to poll repeatedly until factors are available or timeout expires.

        Returns
        -------
        Optional[Dict[str, Any]]
            Server response payload containing global factors and community info.
        """
        endpoint = f"/api/v1/fl/get_global_factors/{round_num}?poll=true" if poll else f"/api/v1/fl/get_global_factors/{round_num}"
        start_time = time.time()

        while True:
            status, payload = self._http_get(endpoint)
            if status == 200 and payload is not None:
                return payload

            if not poll:
                return None

            elapsed = time.time() - start_time
            if elapsed >= self.poll_timeout:
                logger.warning(
                    "Client %d timed out after %.1fs awaiting global factors for round %d",
                    self.client_id,
                    elapsed,
                    round_num,
                )
                return None

            time.sleep(self.poll_interval)

    def upload_factors(self, round_num: int) -> bool:
        """
        Package local item factor updates and upload to central server.

        Parameters
        ----------
        round_num : int
            Current round number.

        Returns
        -------
        bool
            True if upload was accepted (HTTP 200/201).
        """
        payload = self.model.get_upload_payload_dict(
            round_num=round_num,
            client_id=self.client_id,
        )
        endpoint = "/api/v1/fl/upload_factors"
        status, _ = self._http_post(endpoint, payload)
        return status in (200, 201)

    # --------------------------------------------------------------------------
    # Federated Execution Loop
    # --------------------------------------------------------------------------

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Execute a single communication round of the federated protocol.

        Parameters
        ----------
        round_num : int
            Current round (1..rounds).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing round metrics and execution timings.
        """
        round_start = time.time()

        # Step 1: In Round 1, check for Round 0 initial global factors
        if round_num == 1:
            r0_data = self.pull_global_factors(0, poll=False)
            if r0_data is not None:
                factors = r0_data.get("global_factors") or r0_data.get("factors")
                if factors is not None:
                    self.model.set_item_factors(np.asarray(factors, dtype=np.float32))
                    logger.info("Client %d initialized with Round 0 global factors", self.client_id)

            # Phase 1: Initialize unobserved items to the mean of observed items
            self.model.initialize_unobserved_items_mean(interaction_counts=self.interaction_counts)

        # Step 2: Configure Phase 3 centroid anchoring if community info is available
        if self.community_map is not None and self.centroids is not None:
            self.model.set_community_info(
                community_map=self.community_map,
                centroids=self.centroids,
                activate_anchoring=True,
            )
            self.model.anchor_unobserved_items_to_centroids(missing_indices=self.missing_item_ids)

        # Step 3: Local Matrix Factorization training for E epochs
        t_train_start = time.time()
        losses = self.model.fit(
            users=self.users,
            items=self.items,
            ratings=self.ratings,
            epochs=self.epochs_per_round,
            shuffle=True,
        )
        train_duration = time.time() - t_train_start
        final_loss = losses[-1] if losses else 0.0

        # Step 4: Upload item factors V_k and counts n_k
        upload_ok = self.upload_factors(round_num=round_num)
        if not upload_ok:
            logger.error("Client %d failed to upload factors for round %d", self.client_id, round_num)

        # Step 5: Await global aggregated factors for round_num
        global_data = self.pull_global_factors(round_num=round_num, poll=True)
        if global_data is not None:
            factors = global_data.get("global_factors") or global_data.get("factors")
            if factors is not None:
                self.model.set_item_factors(np.asarray(factors, dtype=np.float32))

            # Check if server distributed Phase 2 / Phase 3 Leiden community artifacts
            cmap = global_data.get("community_map")
            cents = global_data.get("centroids")
            if cmap is not None and cents is not None:
                self.community_map = cmap
                self.centroids = cents
                logger.info(
                    "Client %d received Leiden community map (%d items, %d centroids)",
                    self.client_id,
                    len(cmap),
                    len(cents),
                )

        total_duration = time.time() - round_start
        round_stats = {
            "client_id": self.client_id,
            "round": round_num,
            "train_loss": float(final_loss),
            "train_duration": float(train_duration),
            "total_duration": float(total_duration),
            "upload_success": bool(upload_ok),
            "has_community_map": bool(self.community_map is not None),
        }
        self.round_history.append(round_stats)
        return round_stats

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the full federated client workflow for all configured rounds.

        Returns
        -------
        List[Dict[str, Any]]
            List of per-round execution statistics.
        """
        logger.info(
            "Starting FederatedClientAgent for Client %d (%d ratings, %d rounds, E=%d)",
            self.client_id,
            len(self.ratings),
            self.rounds,
            self.epochs_per_round,
        )

        for r in range(1, self.rounds + 1):
            stats = self.run_round(r)
            logger.info(
                "Client %d [Round %02d/%02d] Loss=%.4f | Train=%.2fs | Total=%.2fs",
                self.client_id,
                r,
                self.rounds,
                stats["train_loss"],
                stats["train_duration"],
                stats["total_duration"],
            )

        return self.round_history


def parse_args(args=None) -> argparse.Namespace:
    """Parse CLI arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Fed-LCCF Federated Client Agent")
    parser.add_argument("--client_id", "--client-id", dest="client_id", type=int, required=True, help="Client ID (1..4)")
    parser.add_argument("--db_path", "--db-path", dest="db_path", type=str, required=True, help="Path to client SQLite database")
    parser.add_argument("--server_url", "--server-url", dest="server_url", type=str, default="http://127.0.0.1:8000", help="FastAPI Server URL")
    parser.add_argument("--rounds", type=int, default=20, help="Total federated rounds")
    parser.add_argument("--epochs_per_round", "--epochs-per-round", dest="epochs_per_round", type=int, default=5, help="Local SGD epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--latent_dim", "--latent-dim", dest="latent_dim", type=int, default=32, help="Latent factor dimension d")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--poll_interval", "--poll-interval", dest="poll_interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--poll_timeout", "--poll-timeout", dest="poll_timeout", type=float, default=120.0, help="Polling timeout in seconds")
    return parser.parse_args(args)


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    agent = FederatedClientAgent(
        client_id=args.client_id,
        db_path=args.db_path,
        server_url=args.server_url,
        rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        random_state=args.seed,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )
    agent.run()


if __name__ == "__main__":
    main()
