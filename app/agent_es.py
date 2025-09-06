import os, sys, logging, asyncio, traceback, inspect
from livekit.agents import JobContext, WorkerOptions, cli
from .agent_supervisor import Supervisor
from .agent_common import run_language_agent_entrypoint  # runner should NOT connect/shutdown if we ask it not to

log = logging.getLogger("agent_es")
log.propagate = False

# Avoid duplicate root handlers
if not logging.getLogger().handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)


async def entrypoint(ctx):
    # Loud banner to confirm EN process is running
    log.info("***********ES AGENT ENTRYPOINT***************")

    # Mark env for child/handoff
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["HANDOFF_ROOM"] = ctx.room.name
    os.environ["HANDOFF_LANG"] = "es"

    # 3) Supervisor (single spawn)
    sup = Supervisor(room=ctx.room.name, ready_timeout=18.0, single_spawn=True)
    await run_language_agent_entrypoint(ctx, "fr", supervisor=sup, tools=None, skip_connect=True, skip_shutdown=True)

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="test"))
