import logging, os
from .agent_supervisor import LanguageSupervisor, OneShotSupervisor
from .agent_base import AutomotiveBookingAssistant
from .agent_common import run_language_agent_entrypoint, prewarm
from livekit.agents import JobContext, WorkerOptions, cli

# --- logging: make sure prints/logs show up immediately ---
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

log = logging.getLogger("agent_fr")

async def entrypoint(ctx):
    """
    French agent entrypoint:
      - Ensures HANDOFF_ROOM/HANDOFF_LANG are set (for supervisor handoff)
      - Logs the resolved env clearly
      - Delegates to run_language_agent_entrypoint(ctx, "fr")
        which MUST: join the room, then touch /tmp/{HANDOFF_ROOM}-READY-fr, then greet.
    """
    log.info("***********FRENCH AGENT***************")

    # Ensure the child knows the room even if spawned without CLI args
    room_name = None
    if getattr(ctx, "room", None) and getattr(ctx.room, "name", None):
        room_name = ctx.room.name
    env_room = os.environ.get("HANDOFF_ROOM") or room_name or "unknown_room"
    os.environ["HANDOFF_ROOM"] = env_room

    # Mark this process as the FR handoff child (lets the child bypass singleflight)
    os.environ.setdefault("HANDOFF_LANG", "fr")

    log.info(f"[fr] entrypoint starting: HANDOFF_ROOM={os.environ.get('HANDOFF_ROOM')} "
             f"HANDOFF_LANG={os.environ.get('HANDOFF_LANG')}")

    try:
        # Delegate to the shared boot logic (it will:
        #  - normalize 'fr'
        #  - build STT/TTs for FR
        #  - session.start(...)
        #  - write READY file: /tmp/{HANDOFF_ROOM}-READY-fr
        #  - greet in French
        await run_language_agent_entrypoint(ctx, "fr")
    except Exception as e:
        # Make any startup failure VERY visible
        log.exception("[fr] fatal error in entrypoint: %s", e)
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=None))