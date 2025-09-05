# app/agent_en.py
import logging, os, typing as t
from livekit.agents import JobContext, WorkerOptions, cli
from .agent_supervisor import Supervisor
from .agent_common import run_language_agent_entrypoint, prewarm

log = logging.getLogger("agent_en")

def _resolve_room_name(ctx: JobContext) -> str:
    """
    Returns the room name as a plain string.
    Works whether launched via `connect --room ...` or via env.
    Handles cases where attributes are Room objects.
    """
    # 1) Try job-bound fields first (preferred when using CLI `connect --room`)
    job = getattr(ctx, "job", None)
    if job is not None:
        # Some builds expose job.room_name (str); others expose job.room (str or Room)
        val = getattr(job, "room_name", None)
        if isinstance(val, str) and val:
            return val

        val = getattr(job, "room", None)
        if isinstance(val, str) and val:
            return val
        if getattr(val, "name", None):  # Room object
            name = getattr(val, "name")
            if isinstance(name, str) and name:
                return name

    # 2) If already connected (or partially prepared), ctx.room may exist
    room_obj = getattr(ctx, "room", None)
    if getattr(room_obj, "name", None):
        name = getattr(room_obj, "name")
        if isinstance(name, str) and name:
            return name

    # 3) Fallback to env (when launched by your Supervisor)
    env_room = os.getenv("HANDOFF_ROOM")
    if env_room:
        return env_room

    raise RuntimeError("Could not resolve room name (no job.room_name, no job.room/.name, no HANDOFF_ROOM).")

async def entrypoint(ctx: JobContext):
    log.info("***********EN AGENT ENTRYPOINT***************")

    room_name = _resolve_room_name(ctx)

    # Export for glue that relies on HANDOFF_* and READY flags
    os.environ["HANDOFF_ROOM"] = room_name          # <-- guaranteed str now
    os.environ["HANDOFF_LANG"] = "en"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # One-shot
