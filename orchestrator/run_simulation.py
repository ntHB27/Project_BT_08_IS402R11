"""
Fed-LCCF Local Multi-Process Simulation Orchestrator.

Orchestrates central FastAPI aggregation server and 4 client worker processes
concurrently on a single machine using Python virtualenv. Manages startup health
checks, barrier tracking, real-time logging to logs/, and graceful tree termination.
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, TextIO, Union
import urllib.error
import urllib.request

logger = logging.getLogger("fedlccf.orchestrator")


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a TCP port is available on the specified host interface.

    Parameters
    ----------
    port : int
        TCP port number to test.
    host : str
        Host interface IP address (default: "127.0.0.1").

    Returns
    -------
    bool
        True if the port is free to bind; False if in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.bind((host, port))
            return True
        except (OSError, socket.error):
            return False


def _kill_process_tree(pid: int) -> None:
    """
    Recursively terminate a process and all its child/grandchild processes.

    Uses psutil with a Windows taskkill fallback.
    Idempotent: safely handles already terminated or non-existent processes.

    Parameters
    ----------
    pid : int
        Process identifier to kill.
    """
    if pid is None or pid <= 0:
        return

    try:
        import psutil

        if not psutil.pid_exists(pid):
            return

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Wait briefly for process cleanup
        _, alive = psutil.wait_procs(children + [parent], timeout=1.5)
        if not alive:
            return
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass


class LocalSimulationOrchestrator:
    """
    Orchestrates local multi-process Fed-LCCF simulation.

    Manages subprocess lifecycles for the central FastAPI server and client agents,
    providing health polling, barrier synchronization tracking, structured log capture,
    and two-tier graceful/forceful process tree teardown.
    """

    def __init__(
        self,
        rounds: int = 20,
        clients: int = 4,
        host: str = "127.0.0.1",
        port: int = 8000,
        server_url: Optional[str] = None,
        data_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        mode: str = "full",
        epochs: int = 5,
        lr: float = 0.01,
        latent_dim: int = 32,
        gamma: float = 10.0,
        seed: int = 42,
        health_timeout: float = 30.0,
        poll_interval: float = 1.0,
        poll_timeout: float = 120.0,
        clean_logs: bool = False,
        verbose: bool = False,
    ) -> None:
        self.rounds = int(rounds)
        self.clients = int(clients)
        self.host = str(host)
        self.port = int(port)
        self.mode = str(mode).lower()
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.latent_dim = int(latent_dim)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.health_timeout = float(health_timeout)
        self.poll_interval = float(poll_interval)
        self.poll_timeout = float(poll_timeout)
        self.clean_logs = bool(clean_logs)
        self.verbose = bool(verbose)

        # Path resolutions
        self.orchestrator_dir = Path(__file__).resolve().parent
        self.prototype_root = self.orchestrator_dir.parent
        self.workspace_root = self.prototype_root.parent

        self.data_dir = Path(data_dir).resolve() if data_dir else self.prototype_root / "data"
        self.log_dir = Path(log_dir).resolve() if log_dir else self.orchestrator_dir / "logs"
        self.client_script_path = self.prototype_root / "client_ml" / "client_agent.py"
        self.server_script_path = self.prototype_root / "cloud_server" / "server.py"

        # Networking
        self.connect_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        self.effective_server_url = server_url or f"http://{self.connect_host}:{self.port}"
        self.server_url = self.effective_server_url

        self.manage_server = self.mode in ("full", "server-only")
        self.manage_clients = self.mode in ("full", "clients-only")

        # Process & I/O Tracking
        self.server_process: Optional[subprocess.Popen] = None
        self.client_processes: Dict[int, subprocess.Popen] = {}
        self.log_files: Dict[str, TextIO] = {}
        self._is_stopped = False
        self.start_time: Optional[float] = None

    def _log_orchestrator(self, message: str) -> None:
        """Log structured message to stdout and to orchestrator.log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[Orchestrator] {message}"
        print(formatted)

        # Write to orchestrator.log if log_dir is prepared
        if "orchestrator" not in self.log_files and self.log_dir.is_dir():
            try:
                orch_log = self.log_dir / "orchestrator.log"
                self.log_files["orchestrator"] = open(orch_log, "a", encoding="utf-8")
            except Exception:
                pass

        if "orchestrator" in self.log_files:
            try:
                self.log_files["orchestrator"].write(f"{timestamp} {formatted}\n")
                self.log_files["orchestrator"].flush()
            except Exception:
                pass

    def validate_preflight(self) -> bool:
        """
        Validate environment prerequisites, SQLite partitions, and port availability.

        Returns
        -------
        bool
            True if all checks pass.

        Raises
        ------
        ValueError
            If clients < 1 or rounds < 1.
        FileNotFoundError
            If data directory or any client SQLite database is missing.
        RuntimeError
            If target port is already in use by another process.
        """
        if self.clients < 1:
            raise ValueError(f"clients must be >= 1, got {self.clients}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory missing: {self.data_dir}")

        if self.manage_clients or self.mode == "dry-run":
            for cid in range(1, self.clients + 1):
                db_path = self.data_dir / f"client_{cid}.db"
                if not db_path.exists():
                    raise FileNotFoundError(
                        f"Database for client {cid} missing at {db_path}. "
                        "Run 'python data/preprocess_data.py' first."
                    )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.clean_logs:
            for log_file in self.log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                except Exception as e:
                    logger.debug("Failed to purge %s: %s", log_file, e)

        # Port validation
        if self.manage_server:
            if not check_port_available(self.port, self.connect_host):
                raise RuntimeError(
                    f"Port {self.port} on {self.connect_host} is already in use. "
                    "Stop conflicting service or specify a different --port."
                )

        return True

    def start_server(self) -> subprocess.Popen:
        """
        Launch FastAPI aggregation server subprocess redirecting output to logs/server.log.

        Returns
        -------
        subprocess.Popen
            Running server process handle.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        server_log_path = self.log_dir / "server.log"
        self.log_files["server"] = open(server_log_path, "w", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace_root)
        env["PYTHONUNBUFFERED"] = "1"
        env["EXPECTED_CLIENTS"] = str(self.clients)
        env["GAMMA"] = str(self.gamma)

        cmd = [
            sys.executable,
            str(self.server_script_path),
            "--host", self.host,
            "--port", str(self.port),
            "--clients", str(self.clients),
            "--gamma", str(self.gamma),
        ]
        self._log_orchestrator(f"Starting Central FastAPI Server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=self.log_files["server"],
            stderr=subprocess.STDOUT,
            cwd=str(self.workspace_root),
            env=env,
        )
        self._log_orchestrator(f"Server started (PID: {self.server_process.pid})")
        return self.server_process

    def wait_for_server_healthy(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Poll /health endpoint until server responds HTTP 200 with healthy state.

        Fails fast (<0.5s) if the server process terminates abruptly.

        Parameters
        ----------
        timeout : Optional[float]
            Maximum seconds to wait (defaults to self.health_timeout).

        Returns
        -------
        Dict[str, Any]
            Health payload returned by server.

        Raises
        ------
        RuntimeError
            If server subprocess terminates prematurely.
        TimeoutError
            If server does not become healthy within the timeout window.
        """
        effective_timeout = timeout if timeout is not None else self.health_timeout
        health_url = f"{self.effective_server_url}/health"
        deadline = time.time() + effective_timeout
        self._log_orchestrator(f"Awaiting server readiness on {health_url} ...")

        while time.time() < deadline:
            if self.server_process and self.server_process.poll() is not None:
                tail = self._tail_log("server", lines=20)
                raise RuntimeError(
                    f"Server process terminated unexpectedly with exit code {self.server_process.returncode}:\n{tail}"
                )

            # If managing a server process and port is still free, wait for it to bind
            if self.server_process and check_port_available(self.port, self.connect_host):
                time.sleep(0.05)
                continue

            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "FedLCCF-Orchestrator"})
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    if resp.status == 200:
                        payload = json.loads(resp.read().decode("utf-8"))
                        if payload.get("status") == "healthy":
                            self._log_orchestrator(
                                f"Server healthy (phase={payload.get('phase', 1)}, uptime={payload.get('uptime_seconds', 0.0):.1f}s)"
                            )
                            return payload
            except Exception:
                time.sleep(0.05)

        raise TimeoutError(f"Server on {health_url} failed to become healthy within {effective_timeout}s.")

    def start_clients(self) -> Dict[int, subprocess.Popen]:
        """
        Launch all federated client worker processes concurrently.

        Returns
        -------
        Dict[int, subprocess.Popen]
            Mapping of client_id to active subprocess handle.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace_root)
        env["PYTHONUNBUFFERED"] = "1"

        for cid in range(1, self.clients + 1):
            db_path = self.data_dir / f"client_{cid}.db"
            client_log_path = self.log_dir / f"client_{cid}.log"
            self.log_files[f"client_{cid}"] = open(client_log_path, "w", encoding="utf-8")

            cmd = [
                sys.executable,
                str(self.client_script_path),
                "--client_id", str(cid),
                "--db_path", str(db_path),
                "--server_url", self.effective_server_url,
                "--rounds", str(self.rounds),
                "--epochs_per_round", str(self.epochs),
                "--lr", str(self.lr),
                "--latent_dim", str(self.latent_dim),
                "--seed", str(self.seed + cid),
                "--poll_interval", str(self.poll_interval),
                "--poll_timeout", str(self.poll_timeout),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=self.log_files[f"client_{cid}"],
                stderr=subprocess.STDOUT,
                cwd=str(self.workspace_root),
                env=env,
            )
            self.client_processes[cid] = proc

        self._log_orchestrator(
            f"Launched {self.clients} client workers (PIDs: "
            f"{ {cid: p.pid for cid, p in self.client_processes.items()} })"
        )
        return self.client_processes

    def monitor_simulation(self) -> bool:
        """
        Poll round status, stream progress to stdout, and detect client failures.

        Returns
        -------
        bool
            True if all clients completed successfully; False if any crashed.
        """
        status_url = f"{self.effective_server_url}/api/v1/fl/round_status"
        last_reported_round = 0
        last_round_start = time.time()

        phase_names = {1: "Vanilla FedAvg", 2: "Leiden Clustering", 3: "Community Shrinkage"}

        while True:
            # Check for server process crash
            if self.manage_server and self.server_process and self.server_process.poll() is not None:
                tail = self._tail_log("server", lines=25)
                self._log_orchestrator(
                    f"Server process terminated unexpectedly with exit code {self.server_process.returncode}:\n{tail}"
                )
                return False

            # Check for client crashes
            for cid, proc in self.client_processes.items():
                if proc.poll() is not None and proc.returncode != 0:
                    tail = self._tail_log(f"client_{cid}", lines=25)
                    self._log_orchestrator(
                        f"Client {cid} crashed with exit code {proc.returncode}:\n{tail}"
                    )
                    return False

            # Check if all clients finished
            if all(proc.poll() is not None for proc in self.client_processes.values()):
                break

            # Poll barrier status for progress updates
            try:
                req = urllib.request.Request(status_url, headers={"User-Agent": "FedLCCF-Orchestrator"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        st = json.loads(resp.read().decode("utf-8"))
                        active_round = st.get("round", 1)
                        phase = st.get("phase", 1)

                        if active_round != last_reported_round and st.get("round_completed"):
                            round_dur = time.time() - last_round_start
                            p_name = phase_names.get(phase, f"Phase {phase}")
                            self._log_orchestrator(
                                f"Round {active_round:02d}/{self.rounds:02d} Complete | "
                                f"Phase {phase} ({p_name}) | Duration: {round_dur:.2f}s"
                            )
                            if active_round == 5:
                                self._log_orchestrator(
                                    "*** Completed Round 5: Transitioning to Phase 2 (Leiden Clustering) ***"
                                )
                            elif active_round == 6:
                                self._log_orchestrator(
                                    f"*** Phase 3 Active: Community Shrinkage (gamma={self.gamma}) ***"
                                )
                            last_reported_round = active_round
                            last_round_start = time.time()
            except Exception:
                pass

            time.sleep(min(self.poll_interval, 0.5))

        # Validate exit codes
        all_ok = all(proc.returncode == 0 for proc in self.client_processes.values())
        return all_ok

    def run(self) -> int:
        """
        Execute the complete orchestrator lifecycle.

        Returns
        -------
        int
            Exit status code (0 for success, non-zero for failure).
        """
        self.start_time = time.time()
        print("=" * 80)
        print(f"Fed-LCCF Local Simulation Orchestrator (Mode: {self.mode})")
        print(f"Rounds: {self.rounds} | Clients: {self.clients} | Port: {self.port} | Gamma: {self.gamma}")
        print(f"Logs: {self.log_dir}")
        print("=" * 80)

        self._setup_signals()

        try:
            self.validate_preflight()

            if self.mode == "dry-run":
                self._log_orchestrator("[PASS] Dry-run validations passed successfully.")
                return 0

            if self.manage_server:
                self.start_server()
                self.wait_for_server_healthy()

            if self.mode == "server-only":
                self._log_orchestrator(f"Server running on {self.effective_server_url}. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1.0)

            if self.manage_clients:
                self.start_clients()
                success = self.monitor_simulation()

                elapsed = time.time() - self.start_time
                if success:
                    self._log_orchestrator("All clients completed successfully.")
                    print("=" * 80)
                    print(f"[PASS] {self.rounds}-Round Simulation Completed Successfully in {elapsed:.2f}s!")
                    print(f"All client and server logs saved to: {self.log_dir}")
                    print("=" * 80)
                    return 0
                else:
                    self._log_orchestrator(f"[FAIL] Simulation failed after {elapsed:.2f}s. Check logs in {self.log_dir}.")
                    return 1

            return 0

        except KeyboardInterrupt:
            print("\n[Orchestrator] Interrupted by user. Shutting down gracefully...")
            return 130
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully terminate child processes, tree-kill lingering tasks, and close log handles."""
        if self._is_stopped:
            return
        self._is_stopped = True

        procs = list(self.client_processes.values())
        if self.server_process and self.manage_server:
            procs.append(self.server_process)

        # Step 1: Graceful SIGTERM / terminate
        for p in procs:
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass

        # Step 2: Brief wait for graceful exit
        time.sleep(0.5)

        # Step 3: Forceful recursive tree kill
        for p in procs:
            if p and p.poll() is None:
                try:
                    _kill_process_tree(p.pid)
                except Exception:
                    pass

        if self.server_process and self.server_process.poll() is None:
            try:
                _kill_process_tree(self.server_process.pid)
            except Exception:
                pass

        # Step 4: Flush and close all open log files
        for name, handle in list(self.log_files.items()):
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        self.log_files.clear()

    def _setup_signals(self) -> None:
        """Configure POSIX and Windows signal handlers for graceful shutdown."""
        def _handler(signum, frame):
            self.stop()
            sys.exit(128 + signum)

        try:
            signal.signal(signal.SIGINT, _handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handler)
        except Exception:
            pass

    def _tail_log(self, name: str, lines: int = 20) -> str:
        """Read the last N lines of a log file."""
        if name in self.log_files:
            try:
                self.log_files[name].flush()
            except Exception:
                pass
        log_path = self.log_dir / f"{name}.log"
        if not log_path.exists():
            return "<log file missing>"
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"<failed to read log: {e}>"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for run_simulation."""
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.run_simulation",
        description="Fed-LCCF Local Multi-Process Simulation Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=20, help="Total federated rounds (1..20)")
    parser.add_argument("--clients", type=int, default=4, help="Number of client worker processes (1..4)")
    parser.add_argument(
        "--mode",
        choices=["full", "server-only", "clients-only", "dry-run"],
        default="full",
        help="Execution mode",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--server-url", type=str, default=None, help="Server URL for client connections")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing client SQLite databases")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log output files")
    parser.add_argument("--clean-logs", action="store_true", help="Clean old logs before execution")
    parser.add_argument("--epochs", type=int, default=5, help="Local SGD epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent factor dimension d")
    parser.add_argument("--gamma", type=float, default=10.0, help="Phase 3 shrinkage parameter gamma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for client workers")
    parser.add_argument("--health-timeout", type=float, default=30.0, help="Max seconds to await server /health")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--poll-timeout", type=float, default=120.0, help="Client polling timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for run_simulation."""
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    orc = LocalSimulationOrchestrator(
        rounds=args.rounds,
        clients=args.clients,
        host=args.host,
        port=args.port,
        server_url=args.server_url,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        gamma=args.gamma,
        seed=args.seed,
        health_timeout=args.health_timeout,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        clean_logs=args.clean_logs,
        verbose=args.verbose,
    )
    return orc.run()


if __name__ == "__main__":
    sys.exit(main())
