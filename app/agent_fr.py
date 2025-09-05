# app/agent_fr.py
import os, sys, logging, asyncio, traceback, inspect
from livekit.agents import JobContext, WorkerOptions, cli
from .agent_supervisor import Supervisor
from .agent_common import run_language_agent_entrypoint  # runner must NOT call connect()/shutdown()

log = logging.getLogger("agent_fr")

async def connect_compat(ctx, *, auto_subscribe=None, identity=None):
    """
    Call ctx.connect(...) using only the parameters supported by the installed
    livekit-agents version. Falls back to ctx.connect() with no args.
    """
    try:
        sig = inspect.signature(ctx.connect)
        kwargs = {}
        if "auto_subscribe" in sig.parameters and auto_subscribe is not None:
            kwargs["auto_subscribe"] = auto_subscribe
        if "identity" in sig.parameters and identity is not None:
            kwargs["identity"] = identity
        # older versions take no args at all
        if kwargs:
            return await ctx.connect(**kwargs)
        return await ctx.connect()
    except TypeError as e:
        log.warning("connect signature mismatch: %s; falling back to ctx.connect()", e)
        return await ctx.connect()

# simple console logs (avoid duplicate handlers)
if not logging.getLogger().handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("agent_fr")

def _mask(s: str | None, keep=3) -> str:
    if not s:
        return "<unset>"
    return s[:keep] + "…" if len(s) > keep else "…"

def _resolve_room_name(ctx: JobContext) -> str:
    # Prefer CLI-provided job fields
    job = getattr(ctx, "job", None)
    rn = getattr(job, "room_name", None)
    if isinstance(rn, str) and rn:
        return rn
    r = getattr(job, "room", None)
    if isinstance(r, str) and r:
        return r
    name = getattr(r, "name", None)
    if isinstance(name, str) and name:
        return name
    # Fallback to env
    env_room = os.getenv("HANDOFF_ROOM")
    if env_room:
        return env_room
    raise RuntimeError("agent_fr: no room provided (job.room_name / job.room / HANDOFF_ROOM)")

async def entrypoint(ctx: JobContext):
    print("***********FRENCH AGENT ENTRYPOINT***************", flush=True)
    log.info("argv=%r", sys.argv)
    log.info("env LIVEKIT_URL=%s API_KEY=%s SECRET=%s",
             _mask(os.getenv("LIVEKIT_URL"), 8),
             _mask(os.getenv("LIVEKIT_API_KEY"), 4),
             _mask(os.getenv("LIVEKIT_API_SECRET"), 4))

    # 0) Resolve room and export minimal env for downstream helpers
    try:
        room = _resolve_room_name(ctx)
    except Exception as e:
        log.error("room resolution failed: %s", e)
        log.error("HANDOFF_ROOM=%r job=%r", os.getenv("HANDOFF_ROOM"), getattr(ctx, "job", None))
        return

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["HANDOFF_ROOM"] = room
    os.environ["HANDOFF_LANG"] = "fr"

    # 1) CONNECT FIRST — if this fails, we log the full exception and exit
    identity = os.getenv("PARTICIPANT_IDENTITY", "agent_fr")
    auto_sub = os.getenv("LK_AUTO_SUBSCRIBE", "1") not in ("0", "false", "False")

    try:
        log.info("[fr] connecting… room=%s identity=%s auto_subscribe=%s", room, identity, auto_sub)
        await connect_compat(ctx, auto_subscribe=True, identity="agent_fr")
        room_name = getattr(getattr(ctx, "room", None), "name", None)
        log.info("[fr] CONNECTED ✓ room=%s", room_name)
    except Exception as e:
        log.error("[fr] CONNECT FAILED: %s", e)
        traceback.print_exc()
        # dump a couple env hints that commonly break connect
        log.error("HINT: verify LIVEKIT_URL/API_KEY/SECRET and that the room name matches your SIP ingress.")
        return

    # 2) Dump participants to verify we’re in the right room with the SIP ingress
    try:
        remotes = list(getattr(ctx.room, "remote_participants", {}).values())
        log.info("[fr] remote participants: %s", [getattr(p, "identity", "?") for p in remotes])
    except Exception:
        pass

    # 3) Supervisor (prevents double-spawn once we confirm FR is joining correctly)
    sup = Supervisor(room=room, ready_timeout=18.0, single_spawn=True)

    # 4) Run your agent logic. If your runner returns quickly, keep the job alive.
    try:
        await run_language_agent_entrypoint(ctx, "fr", supervisor=sup)
        log.warning("[fr] runner returned early; keeping job alive")
        await asyncio.Event().wait()
    finally:
        try:
            await ctx.shutdown()
            log.info("[fr] shutdown complete")
        except Exception as e:
            log.warning("[fr] shutdown error: %s", e)

if __name__ == "__main__":
    # Default to connect if no subcommand was given
    if len(sys.argv) == 1:
        room = os.getenv("HANDOFF_ROOM", "unknown_room")
        sys.argv += ["connect", "--room", room, "--participant-identity", "agent_fr"]
        print(f"[FR] no subcommand; defaulting to: {sys.argv!r}", flush=True)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="test"))

