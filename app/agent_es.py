import logging
from .agent_supervisor import LanguageSupervisor
from .agent_base import AutomotiveBookingAssistant
from .agent_common import run_language_agent_entrypoint
from .app_common import prewarm
from livekit.agents import JobContext, WorkerOptions, cli

log = logging.getLogger("agent_es")

async def entrypoint(ctx):
    log.info("entrypoint[ES] starting")
    await run_language_agent_entrypoint(ctx, "es")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
