import logging, asyncio, os, sys
from .agent_supervisor import LanguageSupervisor
from .agent_base import AutomotiveBookingAssistant
from .agent_common import run_language_agent_entrypoint, prewarm
from livekit.agents import JobContext, WorkerOptions, cli


log = logging.getLogger("agent_en")


class OneShotSupervisor(LanguageSupervisor):
    """Wrap LanguageSupervisor to avoid duplicate FR spawns and surface child logs."""
    def __init__(self, room: str, ready_timeout: float = 15.0):
        super().__init__(room=room, ready_timeout=ready_timeout)
        self._spawned = set()
        self._stdout_task = None
        self._stderr_task = None

    async def start_agent(self, lang: str, state_snapshot: dict | None = None):
        key = f"{self.room}:{lang}"
        if key in self._spawned:
            print(f"[SUPERVISOR] {key} already spawned; skipping.")
            return
        await super().start_agent(lang, state_snapshot)
        self._spawned.add(key)

        # Stream child stdout/stderr so crashes aren’t silent
        async def _stream(pipe, prefix):
            while True:
                line = await pipe.readline()
                if not line:
                    break
                print(f"[{prefix}] {line.decode().rstrip()}")

        if self.child_proc:
            if self.child_proc.stdout and not self._stdout_task:
                self._stdout_task = asyncio.create_task(_stream(self.child_proc.stdout, "FR-STDOUT"))
            if self.child_proc.stderr and not self._stderr_task:
                self._stderr_task = asyncio.create_task(_stream(self.child_proc.stderr, "FR-STDERR"))

async def entrypoint(ctx):
    # Ensure the FR child knows what room to join even if no CLI args are passed
    os.environ["HANDOFF_ROOM"] = ctx.room.name
    sup = OneShotSupervisor(room=ctx.room.name)
    await run_language_agent_entrypoint(ctx, "en", supervisor=sup)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))