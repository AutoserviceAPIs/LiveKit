from .agent_common import normalize_lang
import asyncio, json, os, sys
from pathlib import Path
from typing import Optional

MODULE_BY_LANG = {
    "es": "app.agent_es",
    "fr": "app.agent_fr", 
    "vi": "app.agent_vi", 
    "hi": "app.agent_hi", 
}

class LanguageSupervisor:
    """
    Spawns a language-specific agent into the SAME LiveKit room,
    waits for a READY sentinel, and returns control so the caller (EN agent)
    can shut itself down.
    """

    def __init__(self, room: str, ready_timeout: float = 15.0):
        self.room = room
        self.ready_timeout = ready_timeout
        self.child_proc: Optional[asyncio.subprocess.Process] = None
        self.child_lang: Optional[str] = None

    # ---------- paths / sentinels ----------

    def _state_path(self) -> Path:
        return Path(f"/tmp/{self.room}-handoff.json")

    def _ready_flag(self, lang: str) -> Path:
        return Path(f"/tmp/{self.room}-READY-{lang}")

    # ---------- core API ----------

    async def start_agent(self, lang: str, state_snapshot: dict | None = None):
        """
        Spawn the language agent process (es/fr) and wait until it signals READY.
        Does NOT kill the current (EN) process; caller should silence itself before calling.
        """
        canon = normalize_lang(lang)
        if canon not in MODULE_BY_LANG:
            raise ValueError(f"Unsupported language '{lang}'")

        module = MODULE_BY_LANG[canon]

        # 1) write the handoff snapshot to a temp file
        state_path = self._state_path()
        if state_snapshot is None:
            state_snapshot = {}
        try:
            state_path.write_text(json.dumps(state_snapshot), encoding="utf-8")
        except Exception as e:
            print(f"[SUPERVISOR] failed to write state snapshot: {e}")

        # 2) remove any old READY flags
        ready_flag = self._ready_flag(lang)
        try:
            if ready_flag.exists():
                ready_flag.unlink()
        except Exception:
            pass

        # 3) inherit env + pass path to snapshot via env var
        env = os.environ.copy()
        env["HANDOFF_STATE_PATH"] = str(state_path)
        env["HANDOFF_LANG"] = lang
        env["HANDOFF_ROOM"] = self.room

        # 4) spawn: python -m app.agent_<lang> connect --room <room>
        # (Use "connect" because your CLI shows that command; it should join the given room.)
        cmd = [sys.executable, "-m", module, "connect", "--room", self.room]
        print(f"[SUPERVISOR] spawning {module} for room={self.room}")
        self.child_proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.child_lang = lang

        # 5) wait READY sentinel
        await self._wait_until_ready(ready_flag, timeout=self.ready_timeout)

    async def handoff(self, lang: str, state_snapshot: dict | None = None):
        """
        Public API used by EN agent's set_language() tool handler.
        """
        await self.start_agent(lang, state_snapshot)

    async def stop_child(self):
        """
        Optional utility to stop the last spawned child (if you ever need to).
        """
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

    # ---------- helpers ----------

    async def _wait_until_ready(self, ready_flag: Path, timeout: float = 6.0):
        """
        Poll a READY sentinel file created by the new agent right after it joins the room.
        Replace with a datachannel or Redis signal if you prefer.
        """
        step = 0.1
        waited = 0.0
        while waited < timeout:
            if ready_flag.exists():
                # consume the flag so future waits are clean
                try:
                    ready_flag.unlink()
                except Exception:
                    pass
                print("[SUPERVISOR] READY received")
                return
            await asyncio.sleep(step)
            waited += step
        raise TimeoutError("New agent did not signal READY in time")