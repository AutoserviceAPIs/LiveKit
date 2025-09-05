# agent_supervisor.py
import asyncio, json, os, sys, logging
from pathlib import Path
from typing import Optional, Set
from contextlib import contextmanager, suppress

# import/define these from your codebase
# from .agent_common import normalize_lang, MODULE_BY_LANG, log

logging.getLogger("asyncio").disabled = True

@contextmanager
def _room_lang_lock(lock_path: str):
    """
    Cross-process lock using O_CREAT|O_EXCL; raises FileExistsError if taken.
    Lock is released when context exits.
    """
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        with suppress(FileNotFoundError):
            os.unlink(lock_path)


class Supervisor:
    """
    Unified supervisor.

    - If single_spawn=True: behaves like OneShotSupervisor (spawns at most once
      per (room,lang); subsequent calls are no-ops once READY was seen).
    - If single_spawn=False: behaves like LanguageSupervisor (spawns on every call),
      but still guards against *concurrent* duplicate spawns via a file lock.
    """

    def __init__(self, room: str, ready_timeout: float = 18.0, *, single_spawn: bool = True):
        self.room = room
        self.ready_timeout = ready_timeout
        self.single_spawn = single_spawn

        # Process tracking (most-recent child only; READY gate keeps duplication away)
        self.child_proc: Optional[asyncio.subprocess.Process] = None
        self.child_lang: Optional[str] = None

        # Only used when single_spawn=True
        self._spawned: Set[str] = set()
        self._spawning: Set[str] = set()

    # ---------- helpers ----------
    def _state_path(self) -> Path:
        return Path(f"/tmp/{self.room}-handoff.json")

    def _ready_flag(self, lang: str) -> Path:
        return Path(f"/tmp/{self.room}-READY-{lang}")

    async def _wait_until_ready(self, ready_flag: Path, timeout: float) -> None:
        step = 0.1
        waited = 0.0
        while waited < timeout:
            if ready_flag.exists():
                # Consume the flag so later waits don't see stale READY
                with suppress(Exception):
                    ready_flag.unlink()
                return
            await asyncio.sleep(step)
            waited += step
        raise TimeoutError("New agent did not signal READY in time")

    # ---------- API ----------
    async def start_agent(self, lang: str, state_snapshot: Optional[dict] = None) -> None:
        log.info("***********SUPERVISOR***************")
        canon = normalize_lang(lang)
        module = MODULE_BY_LANG.get(canon)
        print(f"[SUPERVISOR] start_agent(lang={canon}) room={self.room}", flush=True)

        if not module:
            print(f"[SUPERVISOR] unsupported lang {lang!r} (canon={canon})", flush=True)
            raise ValueError(f"Unsupported language '{lang}'")

        key = f"{self.room}:{canon}"

        # single_spawn guards (dedupe steady-state like old OneShotSupervisor)
        if self.single_spawn:
            if key in self._spawned:
                print(f"[SUPERVISOR] {key} already READY; skip.", flush=True)
                return
            if key in self._spawning:
                print(f"[SUPERVISOR] {key} already spawning; skip.", flush=True)
                return

        # Cross-process spawn lock so even racing calls can't double-spawn
        lock_path = f"/tmp/{self.room}-{canon}.spawn.lock"
        try:
            with _room_lang_lock(lock_path):
                if self.single_spawn:
                    self._spawning.add(key)

                # 1) Write snapshot (best-effort)
                state_path = self._state_path()
                try:
                    state_path.write_text(json.dumps(state_snapshot or {}), encoding="utf-8")
                    print(f"[SUPERVISOR] wrote snapshot → {state_path}", flush=True)
                except Exception as e:
                    print(f"[SUPERVISOR] failed to write snapshot: {e}", flush=True)

                # 2) Clear old READY flag
                ready_flag = self._ready_flag(canon)
                try:
                    if ready_flag.exists():
                        ready_flag.unlink()
                    print(f"[SUPERVISOR] cleared READY flag → {ready_flag}", flush=True)
                except Exception as e:
                    print(f"[SUPERVISOR] could not clear READY flag: {e}", flush=True)

                print(f"[SUPERVISOR] waiting for READY at {ready_flag} (timeout={self.ready_timeout}s)", flush=True)

                # 3) Env for child (unbuffered prints)
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["HANDOFF_STATE_PATH"] = str(state_path)
                env["HANDOFF_LANG"] = canon
                env["HANDOFF_ROOM"] = self.room

                # 4) Spawn child with the CLI subcommand your agent expects
                #    livekit.agents.cli expects: python -m app.agent_<lang> connect --room <ROOM>
                cmd = [sys.executable, "-m", module, "connect", "--room", self.room]
                print(f"[SUPERVISOR] spawning: {' '.join(cmd)}  (room={self.room}, lang={canon})", flush=True)

                self.child_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.child_lang = canon

                # 5) Stream child logs
                async def _stream(pipe, prefix):
                    while True:
                        line = await pipe.readline()
                        if not line:
                            break
                        print(f"[{prefix}] {line.decode(errors='ignore').rstrip()}", flush=True)

                if self.child_proc.stdout:
                    asyncio.create_task(_stream(self.child_proc.stdout, f"{canon.upper()}-OUT"))
                if self.child_proc.stderr:
                    asyncio.create_task(_stream(self.child_proc.stderr, f"{canon.upper()}-ERR"))

                # 6) Wait for READY or child exit (whichever happens first)
                async def _wait_ready():
                    await self._wait_until_ready(ready_flag, timeout=self.ready_timeout)

                ready_task = asyncio.create_task(_wait_ready())
                exit_task = asyncio.create_task(self.child_proc.wait())

                done, pending = await asyncio.wait(
                    {ready_task, exit_task},
                    return_when=asyncio.FIRST_COMPLETED
                )

                if exit_task in done and not ready_task.done():
                    rc = self.child_proc.returncode
                    for t in pending:
                        t.cancel()
                    print(f"[SUPERVISOR] child exited BEFORE READY (rc={rc})", flush=True)
                    raise RuntimeError(f"Child {module} exited before READY (rc={rc})")

                for t in pending:
                    t.cancel()
                print("[SUPERVISOR] READY received", flush=True)

                if self.single_spawn:
                    self._spawned.add(key)

        except FileExistsError:
            # Another process/thread already holds the spawn lock → skip this attempt.
            print(f"[SUPERVISOR] spawn lock held → {self.room}:{canon} already spawning/running; skip.", flush=True)
            return
        finally:
            if self.single_spawn:
                self._spawning.discard(key)

    async def handoff(self, lang: str, state_snapshot: Optional[dict] = None) -> None:
        # Alias to keep old call sites intact
        log.info("***********SUPERVISOR: handoff***************")
        await self.start_agent(lang, state_snapshot)

    async def stop_child(self) -> None:
        log.info("***********SUPERVISOR: Stop child***************")
        if not self.child_proc:
            return
        try:
            self.child_proc.terminate()
            await asyncio.wait_for(self.child_proc.wait(), timeout=3.0)
        except Exception:
            with suppress(Exception):
                self.child_proc.kill()
        finally:
            self.child_proc = None
            self.child_lang = None
        log.info("***********SUPERVISOR***************")

    def reset_spawn_record(self) -> None:
        """Allow re-spawn for (room,lang) when single_spawn=True."""
        self._spawned.clear()
        self._spawning.clear()
