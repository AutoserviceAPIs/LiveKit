import logging
from .agent_supervisor import LanguageSupervisor
from .agent_base import AutomotiveBookingAssistant
from .agent_common import run_language_agent_entrypoint, prewarm
from livekit.agents import JobContext, WorkerOptions, cli

log = logging.getLogger("agent_fr")

async def entrypoint(ctx):
    log.info("entrypoint[FR] starting")
    await run_language_agent_entrypoint(ctx, "fr")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))