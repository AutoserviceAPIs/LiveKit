# app/agent_fr.py
import os, sys, logging

# Make stdout unbuffered so child logs show up immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from livekit.agents import WorkerOptions, JobContext         # <-- correct imports
from livekit.agents.cli import cli                           # <-- cli comes from .cli
from .agent_common import run_language_agent_entrypoint       # <-- async function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("agent_fr")

async def entrypoint(ctx: JobContext):
    """
    The ONLY entrypoint for the FR worker. Must be async and awaited by the runtime.
    """
    log.info("***********FRENCH AGENT***************")

    # Ensure HANDOFF_* are present for supervisor handoff path.
    room_name = getattr(getattr(ctx, "room", None), "name", None)
    env_room = os.environ.get("HANDOFF_ROOM") or room_name or "unknown_room"
    os.environ["HANDOFF_ROOM"] = env_room
    os.environ.setdefault("HANDOFF_LANG", "fr")

    log.info(
        "[fr] entrypoint starting: HANDOFF_ROOM=%s HANDOFF_LANG=%s",
        os.environ.get("HANDOFF_ROOM"),
        os.environ.get("HANDOFF_LANG"),
    )

    # IMPORTANT: await the shared runner (it's async)
    await run_language_agent_entrypoint(ctx, "fr")

# Keep this file minimal: do NOT import or define any `prewarm` here unless you really need it.
# If you DO need a prewarm, it must be a SYNC function:  def prewarm(proc): ...

if __name__ == "__main__":
    # Pass ONLY the entrypoint; omit prewarm entirely unless required/supported by your SDK version.
    # Passing prewarm_fnc=None can trip some versions; just omit it.
    assert callable(entrypoint), "entrypoint is not callable"
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
