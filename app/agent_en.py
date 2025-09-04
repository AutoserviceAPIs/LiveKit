import logging, asyncio, os, sys
from .agent_supervisor import OneShotSupervisor
from .agent_base import AutomotiveBookingAssistant
from .agent_common import run_language_agent_entrypoint, prewarm
from livekit.agents import JobContext, WorkerOptions, cli


log = logging.getLogger("agent_en")

async def entrypoint(ctx):
    # Loud banner to confirm EN process is running
    log.info("***********EN AGENT ENTRYPOINT***************")

    # Make sure child inherits the room name (even if no CLI args)
    os.environ["HANDOFF_ROOM"] = ctx.room.name
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    sup = OneShotSupervisor(room=ctx.room.name, ready_timeout=18.0)
    await run_language_agent_entrypoint(ctx, "en", supervisor=sup)

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
