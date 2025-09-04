# agent_common imports agent_base and app_common.
# agent_base should not import agent_common.
# agent_en/es/fr import agent_common (and not vice versa).
# supervisor doesn’t import agents; it spawns them via python -m app.agent_es etc.

from __future__ import annotations  # optional, but nice to have in 3.11+
from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#from .agent_base import AutomotiveBookingAssistant  # type-only, no runtime import
from .agent_customerdata import (BUSINESSNAME, INSTRUCTIONS_RECALL, BUSINESSLOCATION, INSTRUCTIONS_CANCEL_RESCHEDULE, INSTRUCTIONS_PRICING, INSTRUCTIONS_WAITTIME, CARS_URL, CHECK_URL, BOOK_URL, HISTORY_URL)
import os, logging
from dotenv import load_dotenv
load_dotenv(".env")  # loads ELEVENLABS_API_KEY if you keep it in .env
log = logging.getLogger("agent_common")
if not os.getenv("ELEVENLABS_API_KEY"):
    log.error("ELEVENLABS_API_KEY is MISSING")
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
from collections import defaultdict

log = logging.getLogger("agent_common")

LLM_MODEL = "gpt-4o-mini"


COMMON_PROMPT = f"""You are a booking assistant. Help customers book appointments.

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer (We already have customer phone number): returns customer name, car details, or booking details.

## RULES:
- After collecting car year make and model: call save_customer_information
- After collecting services and transportation: call save_services_detail
- After booking: call create_appointment
- Do not say things like "Let me save your information" or "Please wait." Just proceed silently to next step
- For recall: {INSTRUCTIONS_RECALL}
- For cancel or reschedule appointment: {INSTRUCTIONS_CANCEL_RESCHEDULE}
- For speak with someone, customer service, or user is frustrated: call transfer_call
- Never say that you saved information

- For address: {BUSINESSLOCATION}
- For price: {INSTRUCTIONS_PRICING}
- For Wait time: {INSTRUCTIONS_WAITTIME}
- Only If user asks if you are a "human" a real person: say "I am actually a voice AI assistant to help you with your service appointment", then Repeat last question

## Follow this conversation flow:

Step 1. Gather First and Last Name

Step 2. Gather vehicle year make and model
- If first name or last name not captured: What is the spelling of your first name / last name?
- Once both first name and last name captured, Ask for the vehicle's year make and model? for example, 2025 Toyota Camry
- call save_customer_information

Step 3. Gather services
- Ask what services are needed for the vehicle
- Wait for services
  - If services has oil change, thank user and ask if user needs a cabin air filter replacement or a tire rotation
  - If services has maintenance, first service or general service: 
      thank user and ask if user is interested in adding wiper blades during the appointment
      Set is_maintenance to 1
  - If user does not know service or wants other services, help them find service, e.g. oil change, diagnostics or repairs
  - Confirm services before going to step 4

Step 4. Gather transportation
- After capture services, Ask if will be dropping off the vehicle or waiting while we do the work
- Wait for transportation
- Must go to Step 5 before Step 6

Step 5. Gather mileage
- call check_available_slots
- Once transportation captured, Ask what is the mileage
- Wait for mileage
    - If user does not know mileage, set mileage to 0
- call save_services_detail tool
Proceed to Step 6

Step 6. Offer first availability
- After services, transportation captured: Thank user, offer the first availability and ask if that will work, or if the user has a specific time

Step 7. Find availability
If first availability works for user, book it
Else:
    If user provides a period, ask for a date and time
    Once date and time captured:
    If found availability, book it
    Else:
        Offer 3 available times and repeat till user finds availability.
            If availability is found, confirm with: Just to be sure, you would like to book ...
    On book:
        Thank the user and Inform they will receive an email or text confirmation shortly. Have a great day and we will see you soon
        call create_appointment, after that, say goodbye and the call will automatically end.
"""


FOUND_APPT_PROMPT = f"""You are a booking assistant. Help customers book appointments.

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer (We already have customer phone number): returns customer name, car details, or booking details.
- next, call check_available_slots

## Follow this conversation flow:
Step 1. Reschedule or Cancel
- Ask user if they want to reschedule or cancel the appointment:
    If reschedule:
        Ask what is preferred date and time to reschedule to
        If user provides a period, ask for a date and time
        Once date and time captured:
        If found availability, reschedule it
        Else:
            Offer 3 available times and repeat till user finds availability.
                If availability is found, confirm with: Just to be sure, you would like to reschedule ...
        On reschedule:
            Thank the user and Inform they will receive an email or text confirmation shortly. Have a great day and we will see you soon
            call reschedule_appointment, after that, say goodbye and the call will automatically end.
    If cancel:
        Call cancel_appointment
        Thank the user and Inform they will receive an email or text confirmation shortly. Have a great day and we will see you soon
    Else:
        call transfer_call

"""


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
        "has_Appt":   "Hola {first_name}, veo que tienes una cita el {appointment_date} a las {appointment_time}. Quieres reprogramarla o cómo puedo ayudarte?",
    },
    "fr": {
        "full":       "Bonjour {first} ! Bienvenue chez Woodbine Toyota. Je m’appelle Sara. Je vois que vous souhaitez prendre un rendez-vous. Quel service désirez-vous pour votre {year} {model} ?",
        "no_vehicle": "Bonjour {first} ! Bienvenue chez Woodbine Toyota. Je m’appelle Sara. Je vois que vous souhaitez prendre un rendez-vous. Quelle est l’année, la marque et le modèle de votre véhicule ?",
        "no_name":    "Bonjour ! Vous avez joint le service de Woodbine Toyota. Je m’appelle Sara. Je serai ravie de vous aider à prendre un rendez-vous. Avec qui ai-je le plaisir de parler ?",
        "has_Appt":   "Bonjour {first_name}, je vois que vous avez un rendez-vous le {appointment_date} à {appointment_time}. Souhaitez-vous le reporter ou comment puis-je vous aider ?",
    },
}


#- If customer name and car details found: Hello {first_name}! welcome back to <business name> service. My name is Sara. I see you are calling to schedule an appointment. What service would you like for your {year} {model}?. Proceed to Step 3
#- If car details not found: Hello {first_name}! welcome back to <business name> service. My name is Sara. I see you are calling to schedule an appointment. What is your car's year, make, and model?. Proceed to Step 2
#- If customer name not found: Hello! You reached <business name> Service. My name is Sara. I'll be glad to help with your appointment. Who do I have the pleasure of speaking with?
#- Hello {first_name}, I see you have an appointment coming up on {appointment_date} at {appointment_time}. Would you like to reschedule your appointment, or could I help you with something else?


READY_FLAG = lambda room, lang: Path(f"/tmp/{room}-READY-{lang}")


VOICE_BY_LANG = { 
    "en" :   "zmcVlqmyk3Jpn5AVYcAL", #Arabella
    "fr" :   "zmcVlqmyk3Jpn5AVYcAL", #Arabella
    "es" :   "zmcVlqmyk3Jpn5AVYcAL", #Arabella
    "vi" :   "zmcVlqmyk3Jpn5AVYcAL", #Arabella
    "hi" :   "zmcVlqmyk3Jpn5AVYcAL", #Arabella
#    "en":    "zmcVlqmyk3Jpn5AVYcAL", Sapphire
#    "es":    "jB2lPb5DhAX6l1TLkKXy",
#    "fr":    "BewlJwjEWiFLWoXrbGMf",
#    "hi":    "CpLFIATEbkaZdJr01erZ",
#    "vi":    "8h6XlERYN1nW5v3TWkOQ",
#    "zh":    "bhJUNIXWQQ94l8eI2VUf",
#    "cn":    "bhJUNIXWQQ94l8eI2VUf",
#    uYXf8XasLslADfZ2MB4u Hope
#    aEO01A4wXwd1O8GPgGlF Arabeita
#    WAhoMTNdLdMoq1j3wf3I Hope - rough no
}


LANG_SYNONYMS = {
    "fr": ["french","francais","français","fr", "fransay","fransais","frances","en français","parlez français"],
    "es": ["spanish","espanol","español","es","en español","habla español","puedes hablar español"],
    "hi": ["hindi","hi","हिंदी"],
    "vi": ["vietnamese","viet","vi","tiếng việt","tieng viet"],
    "en-US": ["english","en","inglés","anglais"],
    "en": ["english","en","inglés","anglais"],
    # Chinese (Mandarin). Avoid non-standard 'cn' but accept it as alias.
    "zh": ["chinese", "mandarin", "zh", "zh-cn", "中文", "普通话", "国语", "cn"],
}


CONFIRM_BY_LANG = {
    "en-US": "Okay, I’ll continue in English.",
    "en":    "Okay, I’ll continue in English.",
    "es":    "De acuerdo, continuaré en español.",
    "fr":    "D’accord, je continue en français.",
    "hi":    "ठीक है, मैं हिंदी में जारी रखूँगा।",
    "vi":    "Được rồi, tôi sẽ tiếp tục bằng tiếng Việt.",
    "zh":    "好的，我会继续用中文交流。",
}


UNSUPPORTED_REPLY = "Sorry, that language isn’t supported yet. I can transfer you to a person if you’d like."


SUPPORTED_STT_LANGS = {"en-US","es","fr","hi","vi","vi-VN","zh","zh-CN"}


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
    log.info(f"build_dynamic_greeting_and_next")
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


def choose_prompt_by_flags(lang: str, found_cust: bool, found_appt: bool) -> str:
    # Use per-language if you have them; else fallback to globals.
    if found_appt and "FOUND_APPT_PROMPT" in globals():
        # If you maintain language-specific variants, use e.g. FOUND_APPT_PROMPT_BY_LANG[lang]
        return FOUND_APPT_PROMPT
    if found_cust and "COMMON_PROMPT" in globals():
        # Or COMMON_PROMPT_BY_LANG[lang]
        return COMMON_PROMPT
    return PROMPT_BY_LANG[lang]


def _singleflight_room(room_name: str) -> bool:
    """Return True if we acquired the lock (first process), False if duplicate."""
    p = Path(f"/tmp/once.{room_name}.lock")
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_room_lock(room_name: str) -> None:
    try:
        Path(f"/tmp/once.{room_name}.lock").unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


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

    room_obj   = getattr(ctx, "room", None)
    room_name  = getattr(room_obj, "name", None) or os.getenv("HANDOFF_ROOM") or "unknown_room"
    ctx.log_context_fields = {"room": room_name}

    # one process per room
    if not _singleflight_room(room_name):
        log.warning(f"[{lang}] duplicate entrypoint for room={room_name}; exiting")
        return

    try:
        log.info(f"run_language_agent_entrypoint - lang {lang}")
        # ... NOW do extract_phone_from_room_name(room_name)
        phone_number = extract_phone_from_room_name(room_name)
        log.info(f"extract_phone_from_room_name {room_name}")

        # do your singleflight lookup here (NOT before the lock)
        found_cust, found_appt, seed = await lookup_customer_singleflight(phone_number) if phone_number else (False, False, {})

        # ... build session, choose prompt, build agent, start, etc.
        # (rest of your existing code)

    finally:
        # release startup lock once we’ve finished initializing (or exited early with error)
        _release_room_lock(room_name)

    room_name = ctx.room.name
    if not _singleflight_room(room_name):
        log.warning(f"[{lang}] duplicate entrypoint for room={room_name}; exiting early")
        return

    # normalize + map for Deepgram
    _norm = {"en-US": "en", "en": "en", "es": "es", "fr": "fr"}
    lang = _norm.get(lang, lang)
    assert lang in ("en", "es", "fr"), f"Unsupported lang: {lang}"
    DEEPGRAM_LANG = {"en": "en-US", "es": "es", "fr": "fr"}

    ctx.log_context_fields = {"room": ctx.room.name}

    # 1) Build session
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

    # 2) Lookup BEFORE agent exists
    phone_number = extract_phone_from_room_name(ctx.room.name)
    found_cust = found_appt = False
    seed = {}
    if phone_number:
        try:
            found_cust, found_appt, seed = await lookup_customer_by_phone(phone=phone_number, cars_url=CARS_URL)
            log.info(f"[{lang}] lookup: found_cust={found_cust} found_appt={found_appt}")
        except Exception as e:
            log.warning(f"[{lang}] lookup failed: {e}")

    # 3) Choose prompt based on lookup
    final_prompt = choose_prompt_by_flags(lang, found_cust, found_appt)
    if found_appt:
        final_prompt = FOUND_APPT_PROMPT
    log.info(f"final_prompt = {final_prompt}")

    # 4) Build agent with that prompt, then seed its state

    from .agent_base import AutomotiveBookingAssistant
    agent = AutomotiveBookingAssistant(
        session, ctx, None,
        instructions=final_prompt,
        lang=lang,
        supervisor=supervisor,
    )

    if phone_number:
        agent._sip_participant_identity = f'sip_{phone_number}'

    # Seed customer_data if we got any
    if seed:
        agent.customer_data.update(seed)
        # Set a sensible starting state
        if found_appt and hasattr(agent, "set_current_state"):
            agent.set_current_state("cancel reschedule")
        elif found_cust and hasattr(agent, "set_current_state"):
            agent.set_current_state("get service")
        elif hasattr(agent, "set_current_state"):
            agent.set_current_state("get name")

    # Handlers / metrics / transcript
    _attach_common_handlers(session, agent, tag=lang.upper())

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

    register_transcript_writer(ctx, session, agent)

    # 5) Start recording & join room
    await start_recording(ctx, agent)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False
        ),
    )

    # Push tools (and final prompt again) BEFORE any generation
    try:
        upd = {"instructions": final_prompt}
        if tools:
            upd.update({"tools": tools, "tool_choice": "auto"})
        if hasattr(session, "update"):
            await session.update(upd)
        elif hasattr(session, "response") and hasattr(session.response, "session_update"):
            await session.response.session_update(upd)
    except Exception as e:
        log.warning(f"[{lang}] session update failed: {e}")

    # Signal READY to supervisor
    room = os.environ.get("HANDOFF_ROOM", ctx.room.name)
    _signal_ready(room, lang)

    # 6) Connect + greet once
    await ctx.connect()
    try:
        await agent.start_background(ctx.room, "office_48k_mono.wav")  # prefer WAV or use your MP3-safe loader
    except Exception as e:
        log.warning(f"[{lang}] start_background failed: {e}")

    greet_text, next_state = build_dynamic_greeting_and_next(agent, lang)
    if hasattr(agent, "set_current_state"):
        agent.set_current_state(next_state)
    else:
        agent.state["progress"] = next_state

    log.info(f"run_language_agent_entrypoint - Greet text: {greet_text}")

    await agent.say(greet_text)


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







# Extract phone number from room name
def extract_phone_from_room_name(room_name: str) -> str:
    """Extract phone number from room name format: call-_8055057710_uHsvtynDWWJN"""
    log.info(f"extract_phone_from_room_name {room_name}")
    pattern = r'call-_(\d+)_'
    match = re.search(pattern, room_name)
    return match.group(1) if match else ""


async def lookup_customer_by_phone(phone: str, *, cars_url: str | None = None):
    """
    Returns: (found_cust: bool, found_appt: bool, seed: dict)
      seed has keys suitable to prefill agent.customer_data
    """
    phone10 = _last10(phone)
    if not phone10:
        log.error(f"lookup: invalid phone: {phone!r}")
        return (False, False, {})

    url = cars_url or os.getenv("CARS_URL")
    if not url:
        log.warning("lookup: CARS_URL not set; skipping lookup")
        return (False, False, {})

    params = {"phone": phone10}
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(url, params=params) as resp:
                if resp.status != 200:
                    log.error(f"lookup: HTTP {resp.status}")
                    return (False, False, {})
                text = await resp.text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.error("lookup: bad JSON")
            return (False, False, {})

        cars = (data or {}).get("Cars") or []
        found_cust = bool(data) and not data.get("error") and bool(data.get("Found")) and len(cars) > 0
        if not found_cust:
            return (False, False, {})

        car = cars[0] or {}
        seed = {
            "phone":            phone10,
            "first_name":       car.get("fname", "") or "",
            "last_name":        car.get("lname", "") or "",
            "car_model":        car.get("model", "") or "",
            "car_make":         car.get("make", "") or "",
            "car_year":         car.get("year", "") or "",
            "appointment_id":   car.get("appointment_id"),
            "appointment_date": car.get("appointment_date", "") or "",
            "appointment_time": car.get("appointment_time", "") or "",
            ###TEST found_appt##
            #"appointment_id":   "1111",
            #"appointment_date": "09-09-2025",
            #"appointment_time": "11:00",
        }

        appt_raw = seed["appointment_id"]
        appt_num = int(appt_raw) if isinstance(appt_raw, str) and appt_raw.isdigit() else (appt_raw if isinstance(appt_raw, int) else None)
        found_appt = bool(appt_num and appt_num > 0) or bool(seed["appointment_date"] and seed["appointment_time"])
        ###TEST found_appt##
        #found_appt = True
        return (True, found_appt, seed)

    except aiohttp.ClientError as e:
        log.error(f"lookup: http error {e}")
        return (False, False, {})
    except Exception as e:
        log.error(f"lookup: unexpected {e}", exc_info=True)
        return (False, False, {})    


def normalize_lang(label: str) -> str:
    if not label:
        return ""
    k = label.strip().lower().replace("_", "-")
    # exact code match first
    for code in ("es", "fr", "en"):
        if k == code or k.startswith(code + "-"):
            return code
    # synonym match
    for code, names in LANG_SYNONYMS.items():
        if k in names:
            return code
    # last resort: two-letter if supported
    short = k[:2]
    return short if short in ("es", "fr", "en") else k


async def start_recording(ctx: JobContext, agent):
    log.info(f"start_recording")
    # 1) Make sure we are connected/joined
    await ctx.connect()

    # 2) (Optional) wait briefly until server lists the room
    #    This avoids a race on first connect
    for _ in range(10):  # up to ~1s
        rooms = await ctx.api.room.list_rooms(api.ListRoomsRequest(names=[ctx.room.name]))
        if rooms.rooms:
            break
        await asyncio.sleep(0.1)

    # 3) Start composite egress
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_key = f"{ctx.room.name}_{ts}.ogg"

    req = api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=True,
        file_outputs=[
            api.EncodedFileOutput(
                file_type=api.EncodedFileType.OGG,
                filepath=object_key,  # used as object key when S3 is set
                s3=api.S3Upload(
                    bucket=os.getenv("S3_BUCKET_NAME", "recording-autoservice"),
                    region=os.getenv("S3_REGION", "us-east-1"),
                    access_key=os.getenv("S3_ACCESS_KEY"),
                    secret=os.getenv("S3_SECRET_KEY"),
                ),
            )
        ],
    )

    # Store recording timestamp for transcript
    agent._recording_timestamp = ts
    await ctx.api.egress.start_room_composite_egress(req)


def prewarm(proc: JobProcess):
    log.info(f"prewarm")
    if "vad" not in proc.userdata:
        proc.userdata["vad"] = silero.VAD.load()
    if "llm" not in proc.userdata:
        proc.userdata["llm"] = openai.LLM(model=LLM_MODEL)


def register_transcript_writer(ctx, session, agent, *,
                               business_name=BUSINESSNAME,
                               project_id=BUSINESSNAME,
                               hours_location=BUSINESSLOCATION,
                               api_url=HISTORY_URL):
    async def write_transcript():
        try:
            current_date = getattr(agent, '_recording_timestamp', None) or datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./transcript_{ctx.room.name}_{current_date}.json"

            phone_number = extract_phone_from_room_name(ctx.room.name)
            formatted_phone = f"1{phone_number}" if phone_number else ""

            conversations = ["TELEPHONY_WELCOME"]
            if getattr(session, 'history', None) and session.history:
                hist = session.history.to_dict()
                for item in hist.get('items', []):
                    content = item.get('content')
                    if content:
                        content0 = content[0] if isinstance(content, list) else str(content)
                        conversations.append(content0)

            record_filename = f"{ctx.room.name}_{current_date}.ogg"
            record_url = f"https://recording-autoservice.s3.us-east-1.amazonaws.com/{record_filename}"

            payload = {
                "phone": formatted_phone,
                "agentId": 1,
                "conversations": conversations,
                "recordURL": record_url,
                "transfer": getattr(agent, "_call_already_transferred", False),
                "voicemail": False,
                "booked": getattr(agent, "appointment_created", False),
                "bookingintent": True,
                "perfect": True,
                "businessName": business_name,
                "customerName": f"{agent.customer_data.get('first_name','')} {agent.customer_data.get('last_name','')}".strip(),
                "serviceOriginal": ", ".join(agent.customer_data.get('services', [])),
                "carModel": agent.customer_data.get('car_model', ''),
                "carYear": agent.customer_data.get('car_year', ''),
                "carMake": agent.customer_data.get('car_make', ''),
                "advisorNumber": "",
                "havetimeslots": len(getattr(agent, "available_slots", [])) > 0,
                "project_id": project_id,
                "check_url": "",
                "book_url": "",
                "transportation_drop": agent.customer_data.get('transportation', ''),
                "fallback_service_default": "",
                "hoursLocation": hours_location,
            }

            conversation_data = {
                "room_name": ctx.room.name,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": session.history.to_dict() if getattr(session, 'history', None) else {},
                "customer_data": getattr(agent, "customer_data", {}),
                "appointment_created": getattr(agent, "appointment_created", False),
                "call_transferred": getattr(agent, "_call_already_transferred", False),
                "api_payload": payload,
            }

            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)

            log.info(f"Transcript saved to {filename}")

            try:
                async with aiohttp.ClientSession() as session_http:
                    async with session_http.post(api_url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                        if resp.status == 200:
                            log.info(f"History sent OK: {await resp.json()}")
                        else:
                            logger.error(f"History send failed: {resp.status} {await resp.text()}")
            except Exception as api_err:
                logger.error(f"Error sending to API: {api_err}")

        except Exception as e:
            logger.error(f"Failed to write transcript: {e}")

    ctx.add_shutdown_callback(write_transcript)


def _last10(phone: str) -> str | None:
    digits = ''.join(filter(str.isdigit, phone or ""))
    return digits[-10:] if len(digits) >= 10 else None


def _lookup_cache_paths(phone10: str):
    base = f"/tmp/lookup.{phone10}"
    return Path(base + ".lock"), Path(base + ".json")


async def lookup_customer_singleflight(phone_number: str):
    phone10 = _last10(phone_number)
    if not phone10:
        return (False, False, {})

    lock_path, cache_path = _lookup_cache_paths(phone10)

    # If someone already cached a result, use it
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return data["found_cust"], data["found_appt"], data["seed"]
        except Exception:
            pass

    # Try to become the lookup owner
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        i_am_owner = True
    except FileExistsError:
        i_am_owner = False

    if i_am_owner:
        # Do the real lookup once
        found_cust, found_appt, seed = await lookup_customer_by_phone(phone=phone_number, cars_url=CARS_URL)
        try:
            cache_path.write_text(json.dumps({
                "found_cust": found_cust, "found_appt": found_appt, "seed": seed
            }))
        except Exception:
            pass
        try:
            lock_path.unlink()
        except Exception:
            pass
        return found_cust, found_appt, seed
    else:
        # Wait briefly for the owner to finish and read the cache
        for _ in range(30):  # up to ~3s
            await asyncio.sleep(0.1)
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    return data["found_cust"], data["found_appt"], data["seed"]
                except Exception:
                    break
        # Fallback: perform our own lookup (rare)
        return await lookup_customer_by_phone(phone=phone_number, cars_url=CARS_URL)