"""
FastAPI Central Aggregation Server for Fed-LCCF.

Exposes REST API endpoints coordinating the 20-round Federated Learning protocol:
1. GET  /health                                 : Server liveness probe and state.
2. POST /api/v1/fl/upload_factors              : Client factors and counts upload.
3. GET  /api/v1/fl/get_global_factors/{round}  : Global factor retrieval (including round 0).
4. GET  /api/v1/fl/round_status                : Round barrier inspection and synchronization state.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path when script is executed directly
_CURRENT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _CURRENT_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from cloud_server.aggregator import (
    DEFAULT_GAMMA,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_ITEMS,
    DEFAULT_RANDOM_SEED,
    FederatedAggregator,
)
from cloud_server.barrier import (
    DEFAULT_EXPECTED_CLIENTS,
    RoundBarrierManager,
)

logger = logging.getLogger("fedlccf.server")


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------

class FactorUploadRequest(BaseModel):
    """Client factor and interaction count upload payload schema."""

    client_id: int = Field(..., description="Client ID (1..4)", ge=1)
    round: int = Field(..., description="Federated communication round index (1..20)", ge=1)
    item_factors: Optional[List[List[float]]] = Field(None, description="Item factor matrix (1682 x 32)")
    factors: Optional[List[List[float]]] = Field(None, description="Alias for item_factors")
    item_counts: Optional[List[int]] = Field(None, description="Item interaction frequency vector (length 1682)")
    counts: Optional[List[int]] = Field(None, description="Alias for item_counts")
    num_samples: Optional[int] = Field(0, description="Total local ratings count")
    local_loss: Optional[float] = Field(0.0, description="Training loss after local epoch")

    @model_validator(mode="after")
    def validate_factors_and_counts(self) -> "FactorUploadRequest":
        # Resolve factors alias
        if self.item_factors is None and self.factors is None:
            raise ValueError("Either 'item_factors' or 'factors' must be provided.")
        if self.item_factors is None:
            self.item_factors = self.factors

        # Resolve counts alias
        if self.item_counts is None and self.counts is None:
            raise ValueError("Either 'item_counts' or 'counts' must be provided.")
        if self.item_counts is None:
            self.item_counts = self.counts

        return self


class FactorUploadResponse(BaseModel):
    """Response returned upon receiving a client upload."""

    status: str = "accepted"
    client_id: int
    round: int
    uploaded_count: int
    barrier_reached: bool


class GlobalFactorsResponse(BaseModel):
    """Global aggregated factors and community partition response."""

    round: int
    phase: int
    global_factors: List[List[float]]
    factors: List[List[float]]
    community_map: Optional[Dict[str, int]] = None
    centroids: Optional[Dict[str, List[float]]] = None
    num_communities: Optional[int] = None


class RoundStatusResponse(BaseModel):
    """Synchronization barrier status response."""

    current_round: int
    round: int
    expected_clients: int
    uploaded_clients: List[int]
    uploaded_count: int
    round_completed: bool
    phase: int


class HealthResponse(BaseModel):
    """Server health probe and diagnostic status."""

    status: str
    current_round: int
    phase: int
    total_clients_registered: int
    uptime_seconds: float


# ------------------------------------------------------------------------------
# Application Factory
# ------------------------------------------------------------------------------

def create_app(
    expected_clients: Optional[int] = None,
    num_items: int = DEFAULT_NUM_ITEMS,
    latent_dim: int = DEFAULT_LATENT_DIM,
    gamma: Optional[float] = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> FastAPI:
    """
    Construct and configure the Fed-LCCF FastAPI central server.

    Parameters
    ----------
    expected_clients : Optional[int]
        Number of participating clients (defaults to EXPECTED_CLIENTS env var or 4).
    num_items : int
        Number of items (default: 1682).
    latent_dim : int
        Latent embedding dimension (default: 32).
    gamma : Optional[float]
        Shrinkage coefficient (defaults to GAMMA env var or 10.0).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """
    if expected_clients is None:
        expected_clients = int(os.environ.get("EXPECTED_CLIENTS", DEFAULT_EXPECTED_CLIENTS))
    if gamma is None:
        gamma = float(os.environ.get("GAMMA", DEFAULT_GAMMA))
    app = FastAPI(
        title="Fed-LCCF Central Aggregation Server",
        description="Federated Leiden Community Collaborative Filtering Cloud Server",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core state components
    barrier = RoundBarrierManager(expected_clients=expected_clients)
    aggregator = FederatedAggregator(
        num_items=num_items,
        latent_dim=latent_dim,
        gamma=gamma,
        random_state=seed,
    )
    start_time = time.time()

    # Store references on app.state for access across testing and endpoints
    app.state.barrier = barrier
    app.state.aggregator = aggregator
    app.state.start_time = start_time

    # --------------------------------------------------------------------------
    # 1. Health Probe Endpoint
    # --------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    async def health_check() -> HealthResponse:
        """Liveness probe reporting current federated round, phase, and uptime."""
        uptime = round(time.time() - start_time, 2)
        cur_round = barrier.current_round
        cur_phase = aggregator.current_phase
        return HealthResponse(
            status="healthy",
            current_round=cur_round,
            phase=cur_phase,
            total_clients_registered=expected_clients,
            uptime_seconds=uptime,
        )

    # --------------------------------------------------------------------------
    # 2. Factor Upload Endpoint
    # --------------------------------------------------------------------------
    @app.post(
        "/api/v1/fl/upload_factors",
        response_model=FactorUploadResponse,
        status_code=status.HTTP_200_OK,
        tags=["Federated Learning"],
    )
    async def upload_factors(
        payload: FactorUploadRequest = Body(...),
    ) -> FactorUploadResponse:
        """
        Receive item factor updates from a participating client.

        Once all expected clients upload for the round, server aggregation is
        automatically triggered, updating global factors and transitioning phases.
        """
        client_id = payload.client_id
        round_num = payload.round

        # Register upload in thread-safe barrier
        is_complete, uploaded_count, uploaded_clients = barrier.register_upload(
            client_id=client_id,
            round_num=round_num,
            payload=payload,
        )

        # Trigger aggregation if barrier reached
        if is_complete and not aggregator.has_round(round_num):
            uploads = barrier.get_uploads(round_num)
            logger.info("Executing server aggregation for round %d...", round_num)
            aggregator.aggregate_round(round_num=round_num, client_uploads=uploads)
            barrier.mark_round_completed(round_num)
            barrier.set_current_round(round_num + 1)

        return FactorUploadResponse(
            status="accepted",
            client_id=client_id,
            round=round_num,
            uploaded_count=uploaded_count,
            barrier_reached=is_complete,
        )

    # --------------------------------------------------------------------------
    # 3. Global Factor Retrieval Endpoint
    # --------------------------------------------------------------------------
    @app.get(
        "/api/v1/fl/get_global_factors/{round}",
        response_model=GlobalFactorsResponse,
        tags=["Federated Learning"],
    )
    async def get_global_factors(
        round: int = FastAPIPath(..., ge=0, description="Target round number (0 for initial seed)"),
        timeout: float = Query(30.0, ge=0.0, le=180.0, description="Max seconds to await barrier completion"),
        poll: bool = Query(False, description="If True, return immediately (202 if not ready) without waiting"),
    ) -> Any:
        """
        Retrieve global item factors for a given communication round.

        - Round 0: Returns the initial synchronized seed factors immediately.
        - Rounds 1..20: Returns aggregated factors once all clients finish uploading.
          If not yet ready and poll=False, awaits barrier completion up to timeout.
        """
        # Round 0 is initialized at server startup
        if round == 0:
            payload = aggregator.get_factors_payload_dict(0)
            if payload is not None:
                return payload
            raise HTTPException(status_code=500, detail="Round 0 factors not initialized")

        # If already aggregated, return immediately if community artifacts are ready
        if aggregator.has_round(round):
            if round >= 5 and (aggregator.community_map is None or aggregator.centroids is None):
                if poll or timeout <= 0:
                    return JSONResponse(
                        status_code=status.HTTP_202_ACCEPTED,
                        content={
                            "status": "waiting",
                            "round": round,
                            "message": "Phase 2 Leiden clustering in progress.",
                        },
                    )
            else:
                payload = aggregator.get_factors_payload_dict(round)
                if payload is not None:
                    return payload

        # If client requested immediate polling response
        if poll or timeout <= 0:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "waiting",
                    "round": round,
                    "message": f"Round {round} aggregation is pending client uploads.",
                },
            )

        # Await barrier completion asynchronously
        ready = await barrier.wait_for_round(round, timeout=timeout)
        if ready:
            poll_start = time.monotonic()
            while not aggregator.has_round(round) or (round >= 5 and (aggregator.community_map is None or aggregator.centroids is None)):
                if (time.monotonic() - poll_start) >= timeout:
                    break
                await asyncio.sleep(0.02)

            if aggregator.has_round(round) and (round < 5 or (aggregator.community_map is not None and aggregator.centroids is not None)):
                payload = aggregator.get_factors_payload_dict(round)
                if payload is not None:
                    return payload

        # Timed out awaiting completion
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "waiting",
                "round": round,
                "message": f"Round {round} still awaiting uploads; polling suggested.",
            },
        )

    # --------------------------------------------------------------------------
    # 4. Round Status Endpoint
    # --------------------------------------------------------------------------
    @app.get(
        "/api/v1/fl/round_status",
        response_model=RoundStatusResponse,
        tags=["Federated Learning"],
    )
    async def get_round_status(
        round: Optional[int] = Query(None, ge=1, description="Round to inspect (defaults to current active round)"),
    ) -> RoundStatusResponse:
        """Inspect barrier progress, uploaded client IDs, and phase for a round."""
        target_round = barrier.current_round if round is None else round
        status_info = barrier.get_round_status(target_round)
        phase = aggregator.get_phase_for_round(target_round)

        return RoundStatusResponse(
            current_round=status_info["current_round"],
            round=status_info["round"],
            expected_clients=status_info["expected_clients"],
            uploaded_clients=status_info["uploaded_clients"],
            uploaded_count=status_info["uploaded_count"],
            round_completed=status_info["round_completed"],
            phase=phase,
        )

    return app


# Default singleton application instance
app = create_app()


# ------------------------------------------------------------------------------
# CLI Entrypoint for Direct Server Execution
# ------------------------------------------------------------------------------

def parse_args(args=None) -> argparse.Namespace:
    """Parse CLI arguments for server execution."""
    parser = argparse.ArgumentParser(description="Fed-LCCF Central Aggregation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--clients", type=int, default=DEFAULT_EXPECTED_CLIENTS, help="Expected clients (default: 4)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Shrinkage gamma (default: 10.0)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    return parser.parse_args(args)


def main() -> None:
    """CLI execution entrypoint using uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    logger.info("Starting Fed-LCCF Central Aggregation Server on %s:%d...", args.host, args.port)

    # Propagate configuration to environment variables
    os.environ["EXPECTED_CLIENTS"] = str(args.clients)
    os.environ["GAMMA"] = str(args.gamma)

    global app
    app = create_app(expected_clients=args.clients, gamma=args.gamma)
    if "cloud_server.server" in sys.modules:
        sys.modules["cloud_server.server"].app = app

    # Allow custom environment port override
    port = int(os.environ.get("PORT", args.port))
    host = os.environ.get("HOST", args.host)

    uvicorn.run(
        "cloud_server.server:app" if args.reload else app,
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
