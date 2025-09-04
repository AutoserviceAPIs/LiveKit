from .agent_common import normalize_lang
import asyncio, json, os, sys, logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("agent")


MODULE_BY_LANG = {
    "es": "app.agent_es",
    "fr": "app.agent_fr", 
    "vi": "app.agent_vi", 
    "hi": "app.agent_hi", 
}


class LanguageSupervisor:
    def __init__(self, room: str, ready_timeout: float = 18.0):
        self.room = room
        self.ready_timeout = ready_timeout
        self.child_proc: asyncio.subprocess.Process | None = None
        self.child_lang: str | None = None

    def _state_path(self) -> Path:
        return Path(f"/tmp/{self.room}-handoff.json")

    def _ready_flag(self, lang: str) -> Path:
        return Path(f"/tmp/{self.room}-READY-{lang}")

    async def start_agent(self, lang: str, state_snapshot: dict | None = None):
        log.info("***********SUPERVISOR***************")
        print(f"[SUPERVISOR] start_agent(lang={lang}) room={self.room}", flush=True)
        canon = normalize_lang(lang)
        module = MODULE_BY_LANG.get(canon)
        if not module:
            print(f"[SUPERVISOR] unsupported lang {lang!r} (canon={canon})", flush=True)
            raise ValueError(f"Unsupported language '{lang}'")

        # 1) Write snapshot
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

        # 3) Env for child
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"             # child prints unbuffered
        env["HANDOFF_STATE_PATH"] = str(state_path)
        env["HANDOFF_LANG"] = canon
        env["HANDOFF_ROOM"] = self.room

        # 4) Spawn WITHOUT extra CLI args; your agent module will read HANDOFF_* envs
        cmd = [sys.executable, "-m", module]
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
        print(f"[SUPERVISOR] waiting for READY at {ready_flag} (timeout={self.ready_timeout}s)", flush=True)

        async def _wait_ready():
            await self._wait_until_ready(ready_flag, timeout=self.ready_timeout)

        ready_task = asyncio.create_task(_wait_ready())
        exit_task = asyncio.create_task(self.child_proc.wait())

        done, pending = await asyncio.wait({ready_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)

        if exit_task in done and not ready_task.done():
            rc = self.child_proc.returncode
            print(f"[SUPERVISOR] child exited BEFORE READY (rc={rc})", flush=True)
            for t in pending:
                t.cancel()
            print(f"[SUPERVISOR] child exited BEFORE READY (rc={rc})", flush=True)
            raise RuntimeError(f"Child {module} exited before READY (rc={rc})")

        for t in pending:
            t.cancel()

        print("[SUPERVISOR] READY received", flush=True)


    async def _wait_until_ready(self, ready_flag: Path, timeout: float = 6.0):
        step = 0.1
        waited = 0.0
        while waited < timeout:
            if ready_flag.exists():
                try:
                    ready_flag.unlink()
                except Exception:
                    pass
                return
            await asyncio.sleep(step)
            waited += step
        raise TimeoutError("New agent did not signal READY in time")

 
    async def handoff(self, lang: str, state_snapshot: dict | None = None):
        log.info("***********SUPERVISOR: handoff***************")
        await self.start_agent(lang, state_snapshot)

    async def stop_child(self):
        log.info("***********SUPERVISOR: Stop child***************")
        if not self.child_proc:
            return
        try:
            self.child_proc.terminate()
            await asyncio.wait_for(self.child_proc.wait(), timeout=3.0)
        except Exception:
            try:
                self.child_proc.kill()
            except Exception:
                pass
        finally:
            self.child_proc = None
            self.child_lang = None


        log.info("***********SUPERVISOR***************")


class OneShotSupervisor(LanguageSupervisor):
    """Supervisor that spawns a single child per (room,lang), streams its logs,
    and waits for READY or child exit (whichever comes first)."""

    def __init__(self, room: str, ready_timeout: float = 18.0):
        self.room = room
        self.ready_timeout = ready_timeout
        self._spawned: set[str] = set()
        self.child_proc: asyncio.subprocess.Process | None = None
        self.child_lang: str | None = None

    def _state_path(self) -> Path:
        return Path(f"/tmp/{self.room}-handoff.json")

    def _ready_flag(self, lang: str) -> Path:
        return Path(f"/tmp/{self.room}-READY-{lang}")

    async def start_agent(self, lang: str, state_snapshot: dict | None = None):
        log.info("***********SUPERVISOR1***************")
        canon = normalize_lang(lang)
        module = MODULE_BY_LANG.get(canon)
        print(f"[SUPERVISOR] start_agent(lang={canon}) room={self.room}", flush=True)

        if not module:
            print(f"[SUPERVISOR] unsupported lang: {lang!r} (canon={canon})", flush=True)
            raise ValueError(f"Unsupported language '{lang}'")

        key = f"{self.room}:{canon}"
        if key in self._spawned:
            print(f"[SUPERVISOR] {key} already spawned; skipping.", flush=True)
            return

        # 1) Write snapshot
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
        except Exception:
            pass

        # 3) Env for child (unbuffered prints)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HANDOFF_STATE_PATH"] = str(state_path)
        env["HANDOFF_LANG"] = canon
        env["HANDOFF_ROOM"] = self.room

        # 4) Spawn child WITHOUT extra CLI args. The agent module uses HANDOFF_* envs.
        cmd = [sys.executable, "-m", module]
        print(f"[SUPERVISOR] spawning: {' '.join(cmd)}  (room={self.room}, lang={canon})", flush=True)

        self.child_proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.child_lang = canon

        # 5) Stream child logs immediately so crashes aren’t silent
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
        print(f"[SUPERVISOR] waiting for READY at {ready_flag} (timeout={self.ready_timeout}s)", flush=True)

        async def _wait_ready():
            await self._wait_until_ready(ready_flag, timeout=self.ready_timeout)

        ready_task = asyncio.create_task(_wait_ready())
        exit_task  = asyncio.create_task(self.child_proc.wait())

        done, pending = await asyncio.wait({ready_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)

        if exit_task in done and not ready_task.done():
            rc = self.child_proc.returncode
            for t in pending:
                t.cancel()
            print(f"[SUPERVISOR] child exited BEFORE READY (rc={rc})", flush=True)
            raise RuntimeError(f"Child {module} exited before READY (rc={rc})")

        for t in pending:
            t.cancel()

        print("[SUPERVISOR] READY received", flush=True)
        self._spawned.add(key)

