# app/agent_fr.py
import os, sys, logging
from livekit.agents import WorkerOptions, JobContext
from livekit.agents.cli import cli
from .agent_common import run_language_agent_entrypoint, _acquire_child_lock, _release_child_lock

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("agent_fr")

async def entrypoint(ctx: JobContext):
    log.info("***********FRENCH AGENT***************")

    room_name = getattr(getattr(ctx, "room", None), "name", None)
    env_room = os.environ.get("HANDOFF_ROOM") or room_name or "unknown_room"
    os.environ["HANDOFF_ROOM"] = env_room
    os.environ.setdefault("HANDOFF_LANG", "fr")

    # Prevent multiple FR children for the same room
    lock = _acquire_child_lock(env_room, "fr")
    if lock is None:
        log.warning("[fr] another FR child already active for room=%s — exiting early", env_room)
        return

    try:
        log.info("[fr] entrypoint starting: HANDOFF_ROOM=%s HANDOFF_LANG=%s", env_room, os.environ.get("HANDOFF_LANG"))
        await run_language_agent_entrypoint(ctx, "fr")
    finally:
        _release_child_lock(lock)

if __name__ == "__main__":
    # Auto-fallback to `connect --room` if called without subcommand
    if len(sys.argv) == 1:
        room = os.environ.get("HANDOFF_ROOM", "unknown_room")
        sys.argv += ["connect", "--room", room]
        print(f"[FR] no subcommand; defaulting to: {sys.argv!r}", flush=True)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
