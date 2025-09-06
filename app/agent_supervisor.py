import asyncio, json, os, sys, logging
from pathlib import Path
from typing import Optional, Set
from contextlib import contextmanager, suppress
from .agent_common import normalize_lang, MODULE_BY_LANG  # <-- import the helpers

log = logging.getLogger("agent_supervisor")
log.propagate = False

@contextmanager
def _room_lang_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        with suppress(FileNotFoundError):
            os.unlink(lock_path)

class Supervisor:
    def __init__(self, room: str, ready_timeout: float = 25.0, *, single_spawn: bool = True):
        self.room = room
        self.ready_timeout = ready_timeout
        self.single_spawn = single_spawn
        self.child_proc: Optional[asyncio.subprocess.Process] = None
        self.child_lang: Optional[str] = None
        self._spawned: Set[str] = set()
        self._spawning: Set[str] = set()

    def _state_path(self) -> Path:
        return Path(f"/tmp/{self.room}-handoff.json")

    def _ready_flag(self, lang: str) -> Path:
        return Path(f"/tmp/{self.room}-READY-{lang}")

    async def _wait_until_ready(self, ready_flag: Path, timeout: float) -> None:
        step = 0.1; waited = 0.0
        while waited < timeout:
            if ready_flag.exists():
                with suppress(Exception): ready_flag.unlink()
                return
            await asyncio.sleep(step); waited += step
        raise TimeoutError("New agent did not signal READY in time")

    async def start_agent(self, lang: str, state_snapshot: Optional[dict] = None) -> None:
        canon = normalize_lang(lang)
        module = MODULE_BY_LANG.get(canon)
        print(f"[SUPERVISOR] start_agent(lang={canon}) room={self.room}", flush=True)
        if not module:
            raise ValueError(f"Unsupported language '{lang}' (canon={canon})")

        key = f"{self.room}:{canon}"
        if self.single_spawn:
            if key in self._spawned:
                print(f"[SUPERVISOR] {key} already READY; skip.", flush=True); return
            if key in self._spawning:
                print(f"[SUPERVISOR] {key} already spawning; skip.", flush=True); return

        lock_path = f"/tmp/{self.room}-{canon}.spawn.lock"
        try:
            with _room_lang_lock(lock_path):
                if self.single_spawn:
                    self._spawning.add(key)

                state_path = self._state_path()
                try:
                    state_path.write_text(json.dumps(state_snapshot or {}), encoding="utf-8")
                    print(f"[SUPERVISOR] wrote snapshot → {state_path}", flush=True)
                except Exception as e:
                    print(f"[SUPERVISOR] failed to write snapshot: {e}", flush=True)

                ready_flag = self._ready_flag(canon)
                with suppress(Exception):
                    if ready_flag.exists():
                        ready_flag.unlink()
                print(f"[SUPERVISOR] waiting for READY at {ready_flag} (timeout={self.ready_timeout}s)", flush=True)

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["HANDOFF_STATE_PATH"] = str(state_path)
                env["HANDOFF_LANG"] = canon
                env["HANDOFF_ROOM"] = self.room

                # IMPORTANT: agent_<lang> must support "connect --room <ROOM>"
                cmd = [sys.executable, "-m", module, "connect", "--room", self.room]
                print(f"[SUPERVISOR] spawning: {' '.join(cmd)}", flush=True)

                self.child_proc = await asyncio.create_subprocess_exec(
                    *cmd, env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.child_lang = canon

                async def _stream(pipe, prefix):
                    while True:
                        line = await pipe.readline()
                        if not line: break
                        print(f"[{prefix}] {line.decode(errors='ignore').rstrip()}", flush=True)

                if self.child_proc.stdout:
                    asyncio.create_task(_stream(self.child_proc.stdout, f"{canon.upper()}-OUT"))
                if self.child_proc.stderr:
                    asyncio.create_task(_stream(self.child_proc.stderr, f"{canon.upper()}-ERR"))

                ready_task = asyncio.create_task(self._wait_until_ready(ready_flag, self.ready_timeout))
                exit_task  = asyncio.create_task(self.child_proc.wait())
                done, pending = await asyncio.wait({ready_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)

                if exit_task in done and not ready_task.done():
                    rc = self.child_proc.returncode
                    for t in pending: t.cancel()
                    print(f"[SUPERVISOR] child exited BEFORE READY (rc={rc})", flush=True)
                    raise RuntimeError(f"Child {module} exited before READY (rc={rc})")

                for t in pending: t.cancel()
                print("[SUPERVISOR] READY received", flush=True)
                if self.single_spawn:
                    self._spawned.add(key)

        except FileExistsError:
            print(f"[SUPERVISOR] spawn lock held → {self.room}:{canon} already spawning/running; skip.", flush=True)
            return
        finally:
            if self.single_spawn:
                self._spawning.discard(key)

    async def handoff(self, lang: str, state_snapshot: Optional[dict] = None) -> None:
        print("[SUPERVISOR] handoff()", flush=True)
        await self.start_agent(lang, state_snapshot)
