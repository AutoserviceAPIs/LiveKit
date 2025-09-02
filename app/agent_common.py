# agent_common imports agent_base and app_common.
# agent_base should not import agent_common.
# agent_en/es/fr import agent_common (and not vice versa).
# supervisor doesn’t import agents; it spawns them via python -m app.agent_es etc.

import os, logging
from dotenv import load_dotenv
load_dotenv(".env")  # loads ELEVENLABS_API_KEY if you keep it in .env
log = logging.getLogger("agent_common")
if not os.getenv("ELEVENLABS_API_KEY"):
    log.error("ELEVENLABS_API_KEY is MISSING")
from .app_common import (start_recording, extract_phone_from_room_name, VOICE_BY_LANG, COMMON_PROMPT, register_transcript_writer)
from .agent_base import AutomotiveBookingAssistant
from livekit import agents, rtc, api
from livekit.agents import (
    NOT_GIVEN, Agent, UserStateChangedEvent, AgentFalseInterruptionEvent,
    AgentSession, JobContext, JobProcess, MetricsCollectedEvent,
    RoomInputOptions, RunContext, WorkerOptions, cli, metrics, get_job_context,)
from livekit.agents.llm import function_tool
from livekit.agents.metrics import TTSMetrics
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from livekit.plugins.elevenlabs import VoiceSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.protocol import room as room_msgs  # ✅ protocol types live here
from livekit.protocol.sip import TransferSIPParticipantRequest
import logging, aiohttp, json, os, asyncio, re, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional
from pathlib import Path

log = logging.getLogger("agent_common")

LLM_MODEL = "gpt-4o-mini"

PROMPT_BY_LANG = {
    "en": f"""You are a professional receptionist for Woodbine Toyota. Help customers book appointments in English only.
- If caller requests another language (Spanish, French, Hindi), call tool `set_language` with the requested language.
- After calling the tool, say "Switching to {{language}}…" and stop responding.
{COMMON_PROMPT}""",
    "es": f"""Usted es un recepcionista profesional para Woodbine Toyota. Responda únicamente en español.
{COMMON_PROMPT}""",
    "fr": f"""Vous êtes une réceptionniste professionnelle pour Woodbine Toyota. Répondez uniquement en français.
{COMMON_PROMPT}""",
}


GREETING_TEMPLATES = {
    "en": {
        "full":       "Hello {first}! Welcome back to Woodbine Toyota. My name is Sara. I see you’re calling to schedule an appointment. What service would you like for your {year} {model}?",
        "no_vehicle": "Hello {first}! Welcome back to Woodbine Toyota. My name is Sara. I see you’re calling to schedule an appointment. What is your car’s year, make, and model?",
        "no_name":    "Hello! You reached Woodbine Toyota Service. My name is Sara. I’ll be glad to help with your appointment. Who do I have the pleasure of speaking with?",
        "reschedule": "Hello {first}! I see you have an appointment coming up for your vehicle. Would you like to reschedule your appointment, or could I help you with something else?",
    },
    "es": {
        "full":       "¡Hola {first}! Bienvenido de nuevo a Woodbine Toyota. Me llamo Sara. Veo que desea programar una cita. ¿Qué servicio le gustaría para su {year} {model}?",
        "no_vehicle": "¡Hola {first}! Bienvenido de nuevo a Woodbine Toyota. Me llamo Sara. Veo que desea programar una cita. ¿Cuál es el año, la marca y el modelo de su auto?",
        "no_name":    "¡Hola! Ha llamado al servicio de Woodbine Toyota. Me llamo Sara. Con gusto le ayudo con su cita. ¿Con quién tengo el gusto de hablar?",
    },
    "fr": {
        "full":       "Bonjour {first} ! Bienvenue chez Woodbine Toyota. Je m’appelle Sara. Je vois que vous souhaitez prendre un rendez-vous. Quel service désirez-vous pour votre {year} {model} ?",
        "no_vehicle": "Bonjour {first} ! Bienvenue chez Woodbine Toyota. Je m’appelle Sara. Je vois que vous souhaitez prendre un rendez-vous. Quelle est l’année, la marque et le modèle de votre véhicule ?",
        "no_name":    "Bonjour ! Vous avez joint le service de Woodbine Toyota. Je m’appelle Sara. Je serai ravie de vous aider à prendre un rendez-vous. Avec qui ai-je le plaisir de parler ?",
    },
}


READY_FLAG = lambda room, lang: Path(f"/tmp/{room}-READY-{lang}")


async def warm_greeting(agent, lang: str):
    # short, language-specific greeting that uses handed-off state if present
    first = (agent.state.get("customer") or {}).get("first")
    if lang == "es":
        msg = f"Hola {first}, ¿prefiere continuar en español?" if first else "Hola, ¿prefiere continuar en español?"
    elif lang == "fr":
        msg = f"Bonjour {first}, préférez-vous continuer en français ?" if first else "Bonjour, préférez-vous continuer en français ?"
    else:  # en
        msg = f"Hi {first}, would you like to continue in English?" if first else "Hi, would you like to continue in English?"
    await agent.say(msg)


def _pick_state_objects(agent):
    # be tolerant to where you store things
    customer = {}
    vehicle  = {}
    for source in (getattr(agent, "state", {}), getattr(agent, "customer_data", {})):
        if not isinstance(source, dict):
            continue
        customer = source.get("customer", customer) or source  # some code stores names in root
        vehicle  = source.get("vehicle", vehicle) or vehicle
    return customer or {}, vehicle or {}

def build_dynamic_greeting_and_next(agent, lang: str):
    customer, vehicle = _pick_state_objects(agent)
    first = (customer or {}).get("first") or (customer or {}).get("first_name") or ""
    last  = (customer or {}).get("last")  or (customer or {}).get("last_name") or ""
    year  = (vehicle  or {}).get("year")  or (agent.customer_data if hasattr(agent, "customer_data") else {}).get("car_year")  or ""
    make  = (vehicle  or {}).get("make")  or (agent.customer_data if hasattr(agent, "customer_data") else {}).get("car_make")  or ""
    model = (vehicle  or {}).get("model") or (agent.customer_data if hasattr(agent, "customer_data") else {}).get("car_model") or ""

    has_name    = bool(first)
    has_vehicle = bool(year and model)
    has_existing_appointment = bool(agent.customer_data.get("has_existing_appointment"))
    tmpl = GREETING_TEMPLATES.get(lang, GREETING_TEMPLATES["en"])
    if has_existing_appointment:
        text = tmpl["reschedule"].format(first=first)
        next_state = "ask reschedule or cancel"
    elif has_name and has_vehicle:
        text = tmpl["full"].format(first=first, year=year, make=make, model=model)
        next_state = "get service"   # you already use this state name
    elif has_name and not has_vehicle:
        text = tmpl["no_vehicle"].format(first=first)
        next_state = "get car"
    else:
        text = tmpl["no_name"]
        next_state = "get name"
    return text, next_state


def _load_snapshot_from_env() -> dict:
    path = os.environ.get("HANDOFF_STATE_PATH")
    if not path or not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"[COMMON] failed to load snapshot: {e}")
        return {}

def _signal_ready(room: str, lang: str):
    try:
        READY_FLAG(room, lang).touch()
        log.info(f"[COMMON] READY sentinel created for {room}/{lang}")
    except Exception as e:
        log.warning(f"[COMMON] READY sentinel failed: {e}")

def _attach_common_handlers(session: "AgentSession", agent, tag: str):
    @session.on("transcription")
    def _tx(ev):
        try:
            lang = getattr(ev, "language", None)
            print(f"[{tag}] STT lang={lang} final={ev.is_final} text={ev.text!r}")
        except Exception:
            print(f"[{tag}] STT final={getattr(ev,'is_final',None)} text={getattr(ev,'text',None)}")

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        if ev.new_state == "speaking":
            agent.cancel_timeout()
            log.info(f"[{tag}] user speaking → cancel timeout")
        elif ev.new_state == "away":
            agent.start_timeout()
            log.info(f"[{tag}] user away → restart timeout")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        log.info(f"[{tag}] item: {ev.item.type} - {ev.item.content[0]}...")

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        log.info(f"[{tag}] user said: {ev.transcript}")

    @session.on("transcript")
    def _on_transcript(ev):
        if (getattr(ev, "text", "") or "").strip():
            agent.cancel_timeout()

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        log.info(f"[{tag}] false interruption → resume")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)


async def run_language_agent_entrypoint(ctx, lang: str, *, supervisor=None, tools=None):
    """
    Shared entrypoint used by agent_en/es/fr wrappers.
    - Builds a session with STT/TTS for `lang`
    - Creates AutomotiveBookingAssistant with the right prompt
    - Restores snapshot from supervisor
    - Signals READY for the supervisor
    - Greets once (language-aware) AFTER connect
    """
    log.info(f"run_language_agent_entrypoint - lang {lang}")

    # ---- normalize internal lang + map for STT ----
    _norm = {"en-US": "en", "en": "en", "es": "es", "fr": "fr"}
    lang = _norm.get(lang, lang)
    assert lang in ("en", "es", "fr"), f"Unsupported lang: {lang}"
    DEEPGRAM_LANG = {"en": "en-US", "es": "es", "fr": "fr"}

    ctx.log_context_fields = {"room": ctx.room.name}
    prompt = PROMPT_BY_LANG[lang]
    #log.info(f"[{lang}] INSTRUCTIONS set")

    # ---- session ----
    session = AgentSession(
        llm=openai.LLM(model=LLM_MODEL),
        stt=deepgram.STT(model="nova-3", language=DEEPGRAM_LANG[lang], interim_results=True),
        tts=elevenlabs.TTS(
            voice_id=VOICE_BY_LANG[lang],
            model="eleven_flash_v2_5",
            voice_settings=VoiceSettings(similarity_boost=0.4, speed=1, stability=0.3, style=1, use_speaker_boost=True),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    # ---- agent ----
    agent = AutomotiveBookingAssistant(
        session, ctx, None,
        instructions=prompt,
        lang=lang,
        supervisor=supervisor,       # <- store supervisor on agent (EN will pass one)
    )

    # common handlers
    _attach_common_handlers(session, agent, tag=lang.upper())

    # ---- metrics collector ----
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        if getattr(agent, "appointment_created", False) and isinstance(ev.metrics, TTSMetrics):
            dur = getattr(ev.metrics, "audio_duration", 0) or 0
            if dur > 0:
                asyncio.create_task(agent.delayed_hangup(dur + 0.5))

        if isinstance(ev.metrics, TTSMetrics):
            dur = getattr(ev.metrics, "audio_duration", 0) or 0
            if dur > 0:
                asyncio.create_task(agent.delayed_timeout_start(dur))

    async def log_usage():
        summary = usage_collector.get_summary()
        log.info(f"[{lang}] Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # transcript writer (your old write_transcript)
    register_transcript_writer(ctx, session, agent)

    # ---- restore state from supervisor snapshot ----
    agent.restore_state(_load_snapshot_from_env())

    # ---- recording + customer prefill ----
    await start_recording(ctx, agent)

    phone_number = extract_phone_from_room_name(ctx.room.name)
    if phone_number:
        log.info(f"[{lang}] looking up customer by phone: {phone_number}")
        agent._sip_participant_identity = f'sip_{phone_number}'
        # be defensive: only call if method exists
        find_fn = getattr(agent, "findCustomer", None)
        if callable(find_fn):
            try:
                found = await find_fn(phone_number)
                if found and hasattr(agent, "set_current_state"):
                    agent.set_current_state("get service")
            except Exception as e:
                log.warning(f"[{lang}] findCustomer failed: {e}")

    # ---- join room ----
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True
        ),
    )

    # (optional) expose tools after start (AgentSession(...) doesn’t accept tools=)
    if tools:
        if hasattr(session, "update"):
            await session.update({"instructions": prompt, "tools": tools, "tool_choice": "auto"})
        elif hasattr(session, "response") and hasattr(session.response, "session_update"):
            await session.response.session_update({"instructions": prompt, "tools": tools, "tool_choice": "auto"})

    # ---- signal READY to supervisor ----
    room = os.environ.get("HANDOFF_ROOM", ctx.room.name)
    _signal_ready(room, lang)

    # ---- connect, then greet once ----
    await ctx.connect()
    await agent.start_background(ctx.room, "office.mp3")

    greet_text, next_state = build_dynamic_greeting_and_next(agent, lang)
    if hasattr(agent, "set_current_state"):
        agent.set_current_state(next_state)
    else:
        agent.state["progress"] = next_state

    await agent.say(greet_text)