"""
Round Barrier Synchronization Module for Fed-LCCF Cloud Server.

Provides a thread-safe barrier coordinating concurrent client uploads per federated round:
- Synchronizes uploads from K clients (default K=4).
- Triggers server aggregation once all K uploads are registered for round t.
- Supports both asyncio.Event notification for non-blocking coroutines and
  polling fallback with status inspections for HTTP clients.
"""

import asyncio
from collections import defaultdict
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("fedlccf.barrier")

DEFAULT_EXPECTED_CLIENTS = 4


class RoundBarrierManager:
    """
    Thread-safe synchronization barrier managing client uploads and round transitions.

    Parameters
    ----------
    expected_clients : int
        Number of participating clients required to reach the barrier (default: 4).
    """

    def __init__(self, expected_clients: int = DEFAULT_EXPECTED_CLIENTS) -> None:
        if expected_clients <= 0:
            raise ValueError(f"expected_clients must be positive; got {expected_clients}")

        self.expected_clients = int(expected_clients)
        self._lock = threading.RLock()

        # Storage: round_num -> {client_id: upload_payload}
        self._uploads: Dict[int, Dict[int, Any]] = defaultdict(dict)

        # Asyncio events for coroutine synchronization: round_num -> asyncio.Event
        self._events: Dict[int, asyncio.Event] = {}

        # Event loop references associated with events
        self._event_loops: Dict[int, asyncio.AbstractEventLoop] = {}

        # Storage for loop-agnostic waiters: round_num -> list of (event_loop, asyncio.Event)
        self._waiters: Dict[int, List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]]] = defaultdict(list)

        # Set of completed round indices
        self._completed_rounds: Set[int] = set()

        # Current active federated round
        self._current_round: int = 1

        # Timestamps for diagnostics: round_num -> {"start": float, "completed": Optional[float]}
        self._round_timestamps: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"start": time.time(), "completed": 0.0}
        )

    # --------------------------------------------------------------------------
    # Registration & Barrier State
    # --------------------------------------------------------------------------

    def register_upload(
        self,
        client_id: int,
        round_num: int,
        payload: Any,
    ) -> Tuple[bool, int, List[int]]:
        """
        Register a client's factor upload for the specified communication round.

        Thread-safe: acquires internal lock before mutating state.

        Parameters
        ----------
        client_id : int
            Unique identifier of the client (e.g. 1, 2, 3, 4).
        round_num : int
            Federated communication round index.
        payload : Any
            The client upload payload (factors, counts, metadata).

        Returns
        -------
        Tuple[bool, int, List[int]]
            - is_complete: True if the barrier has now been reached for round_num.
            - uploaded_count: Number of clients that have uploaded for round_num so far.
            - uploaded_clients: List of client IDs uploaded for round_num.
        """
        with self._lock:
            client_dict = self._uploads[round_num]
            client_dict[int(client_id)] = payload

            uploaded_clients = sorted(list(client_dict.keys()))
            uploaded_count = len(uploaded_clients)
            is_complete = uploaded_count >= self.expected_clients

            if is_complete and round_num not in self._completed_rounds:
                self._completed_rounds.add(round_num)
                self._round_timestamps[round_num]["completed"] = time.time()
                logger.info(
                    "Round %d barrier REACHED (%d/%d clients: %s)",
                    round_num,
                    uploaded_count,
                    self.expected_clients,
                    uploaded_clients,
                )
                self._signal_event_unlocked(round_num)

            return is_complete, uploaded_count, uploaded_clients

    def mark_round_completed(self, round_num: int) -> None:
        """
        Explicitly mark round as aggregated and signal awaiting clients.

        Parameters
        ----------
        round_num : int
            Communication round index.
        """
        with self._lock:
            self._completed_rounds.add(round_num)
            if self._round_timestamps[round_num]["completed"] == 0.0:
                self._round_timestamps[round_num]["completed"] = time.time()
            self._signal_event_unlocked(round_num)

    def _signal_event_unlocked(self, round_num: int) -> None:
        """Signal all registered event loop waiters for round_num across thread/coroutine boundaries."""
        waiters = list(self._waiters.get(round_num, []))
        for loop, ev in waiters:
            if loop.is_closed():
                continue
            if loop.is_running():
                try:
                    loop.call_soon_threadsafe(ev.set)
                except RuntimeError:
                    pass
            else:
                try:
                    ev.set()
                except RuntimeError:
                    pass

        # Also signal legacy single-event if registered
        if round_num in self._events:
            ev = self._events[round_num]
            loop = self._event_loops.get(round_num)
            if loop is not None and not loop.is_closed() and loop.is_running():
                try:
                    loop.call_soon_threadsafe(ev.set)
                except RuntimeError:
                    pass
            else:
                try:
                    ev.set()
                except RuntimeError:
                    pass

    # --------------------------------------------------------------------------
    # Status & Inspection Queries
    # --------------------------------------------------------------------------

    def is_round_complete(self, round_num: int) -> bool:
        """Check whether round_num has reached the barrier or was marked complete."""
        with self._lock:
            if round_num in self._completed_rounds:
                return True
            return len(self._uploads.get(round_num, {})) >= self.expected_clients

    def get_uploads(self, round_num: int) -> Dict[int, Any]:
        """
        Retrieve a copy of all client upload payloads for round_num.

        Returns
        -------
        Dict[int, Any]
            Dictionary mapping client_id -> payload.
        """
        with self._lock:
            return dict(self._uploads.get(round_num, {}))

    def get_uploaded_clients(self, round_num: int) -> List[int]:
        """Return list of client IDs that have uploaded for round_num."""
        with self._lock:
            return sorted(list(self._uploads.get(round_num, {}).keys()))

    def get_round_status(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Get structured status summary for a round (or current round).

        Parameters
        ----------
        round_num : Optional[int]
            Round index to inspect, or None for current active round.

        Returns
        -------
        Dict[str, Any]
            Status dictionary containing barrier progress.
        """
        with self._lock:
            target_round = self._current_round if round_num is None else int(round_num)
            uploaded_clients = sorted(list(self._uploads.get(target_round, {}).keys()))
            uploaded_count = len(uploaded_clients)
            completed = (
                target_round in self._completed_rounds
                or uploaded_count >= self.expected_clients
            )

            return {
                "current_round": self._current_round,
                "round": target_round,
                "expected_clients": self.expected_clients,
                "uploaded_clients": uploaded_clients,
                "uploaded_count": uploaded_count,
                "round_completed": completed,
            }

    @property
    def current_round(self) -> int:
        """Get the current active round."""
        with self._lock:
            return self._current_round

    def set_current_round(self, round_num: int) -> None:
        """Set the current active round."""
        with self._lock:
            self._current_round = int(round_num)

    # --------------------------------------------------------------------------
    # Async Await Coordination
    # --------------------------------------------------------------------------

    def get_or_create_event(self, round_num: int) -> asyncio.Event:
        """
        Thread-safely obtain or create an asyncio.Event for round_num.

        Parameters
        ----------
        round_num : int
            Round index.

        Returns
        -------
        asyncio.Event
            Event that will be set once round aggregation finishes.
        """
        with self._lock:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            ev = asyncio.Event()
            self._events[round_num] = ev
            if loop is not None:
                self._event_loops[round_num] = loop
                self._waiters[round_num].append((loop, ev))

            if self.is_round_complete(round_num):
                ev.set()

            return ev

    async def wait_for_round(
        self,
        round_num: int,
        timeout: Optional[float] = 60.0,
    ) -> bool:
        """
        Asynchronously await barrier completion for round_num across any event loop.

        Parameters
        ----------
        round_num : int
            Target round number.
        timeout : Optional[float]
            Maximum seconds to wait before returning False. None waits indefinitely.

        Returns
        -------
        bool
            True if round is completed within timeout, False otherwise.
        """
        if self.is_round_complete(round_num):
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            ev = asyncio.Event()
            with self._lock:
                if self.is_round_complete(round_num):
                    return True
                self._waiters[round_num].append((loop, ev))

            try:
                if timeout is not None and timeout > 0:
                    await asyncio.wait_for(ev.wait(), timeout=timeout)
                else:
                    await ev.wait()
                return True
            except (asyncio.TimeoutError, TimeoutError):
                return self.is_round_complete(round_num)
            finally:
                with self._lock:
                    if (loop, ev) in self._waiters.get(round_num, []):
                        self._waiters[round_num].remove((loop, ev))
        else:
            # Fallback for environments without running async loop
            start_t = time.monotonic()
            while not self.is_round_complete(round_num):
                if timeout is not None and (time.monotonic() - start_t) >= timeout:
                    return self.is_round_complete(round_num)
                await asyncio.sleep(0.02)
            return True

    # --------------------------------------------------------------------------
    # Reset & Cleanup
    # --------------------------------------------------------------------------

    def reset_round(self, round_num: int) -> None:
        """Reset state for a specific round."""
        with self._lock:
            self._uploads.pop(round_num, None)
            self._completed_rounds.discard(round_num)
            self._events.pop(round_num, None)
            self._event_loops.pop(round_num, None)
            self._waiters.pop(round_num, None)
            self._round_timestamps.pop(round_num, None)

    def reset(self) -> None:
        """Reset the barrier manager completely."""
        with self._lock:
            self._uploads.clear()
            self._completed_rounds.clear()
            self._events.clear()
            self._event_loops.clear()
            self._waiters.clear()
            self._round_timestamps.clear()
            self._current_round = 1
