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
import logging, aiohttp, json, os, asyncio, re, time, math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

log = logging.getLogger("agent_common")

LLM_MODEL = "gpt-4o-mini"

#Step 5. Gather mileage
#- Once transportation captured, Ask what is the mileage
#- call check_available_slots to get available dates
#- Wait for mileage
#    - If user does not know mileage, set mileage to 0
#Proceed to Step 6

#Step 6. Offer first availability
#Step6: - After services, transportation captured: Thank user, offer the first availability and ask if that will work, or if the user has a specific time
#If first availability works for user, book it
#Else:
# Get current date for the prompt
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")
current_date_readable = datetime.now().strftime("%A, %B %d, %Y")

COMMON_PROMPT = f"""You are a booking assistant. Help customers book appointments.

## CURRENT DATE INFORMATION:
- Today's date is: {current_date} ({current_date_readable})
- Use this information when customers ask about "today", "tomorrow", or specific dates
- if morning use time-7 AM
- if afternoon use time=2 PM
- if evening use time=5 PM
- If date not provided: use today's date
- If time not provided: use time=7 AM

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer (We already have customer phone number): returns customer name, car details, or booking details.

## RULES:
- After collecting car year make and model: call save_customer_information
- After collecting services and transportation: call validate_and_save_services (MANDATORY - must be called immediately after getting transportation)
- After collecting mileage: call check_available_slots
- After collecting date and time: call check_available_slots
- After booking: call create_appointment
- Do not say things like "Let me save your information" or "Please wait." Just proceed silently to next step
- For recall: {INSTRUCTIONS_RECALL}
- For cancel or reschedule appointment: {INSTRUCTIONS_CANCEL_RESCHEDULE}
- For speak with someone, customer service, or user is frustrated: call transfer_call
- Never say that you saved information
- CRITICAL: Always call validate_and_save_services immediately after getting transportation response

- For address: {BUSINESSLOCATION}
- For price: {INSTRUCTIONS_PRICING}
- For Wait time: {INSTRUCTIONS_WAITTIME}
- Only If user asks if you are a "human" a real person: say "I am actually a voice AI assistant to help you with your service appointment", then Repeat last question

## Follow this conversation flow:

Step 1. Gather First and Last Name
- If first name or last name not captured: What is the spelling of your first name / last name?

Step 2. Gather vehicle year make and model
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
  - After capture services, go to step 4

Step 4. Gather transportation
- After capture services, Ask if will be dropping off the vehicle or waiting while we do the work for <services>
- Wait for transportation
- IMMEDIATELY after getting transportation, call validate_and_save_services with services and transportation
- Must go to Step 5 before Step 6

Step 5. Ask date and time
- After transportation captured: Thank user, ask if they have a preferred date or time
- Wait for date or time

Step 6. Find availability
- call check_available_slots to get available dates 
- Offer 3 available times and repeat till user finds availability:
    If found availability, book it
    Else:
        call check_available_slots to get available dates 
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
    "en": f"""You are a professional receptionist for {BUSINESSNAME}. Help customers book appointments in English only.
- If caller requests another language (Spanish, French, Hindi), call tool `set_language` with the requested language.
{COMMON_PROMPT}""",
    "es": f"""Usted es un recepcionista profesional para {BUSINESSNAME}. Responda únicamente en español.
{COMMON_PROMPT}""",
    "fr": f"""Vous êtes une réceptionniste professionnelle pour {BUSINESSNAME}. Répondez uniquement en français.
{COMMON_PROMPT}""",
}


GREETING_TEMPLATES = {
    "en": {
        "full":       "Hello {first}! Welcome back to {BUSINESSNAME}. This is Sara. I see you’re calling to schedule an appointment. What service would you like for your {year} {model}?",
        "no_vehicle": "Hello {first}! Welcome back to {BUSINESSNAME}. This is Sara. I see you’re calling to schedule an appointment. What is your car’s year, make, and model?",
        "no_name":    "Hello! This is Sara from {BUSINESSNAME}. I’ll be glad to help with your appointment. Who do I have the pleasure of speaking with?",
        "reschedule": "Hello {first}! Welcome back to {BUSINESSNAME}. This is Sara. I see you have an appointment coming up for your vehicle. Would you like to reschedule your appointment, or could I help you with something else?",
    },
    "es": {
        "full":       "¡Hola {first}! Bienvenido de nuevo a {BUSINESSNAME}. Es Sara. Veo que desea programar una cita. ¿Qué servicio le gustaría para su {year} {model}?",
        "no_vehicle": "¡Hola {first}! Bienvenido de nuevo a {BUSINESSNAME}. Es Sara. Veo que desea programar una cita. ¿Cuál es el año, la marca y el modelo de su auto?",
        "no_name":    "¡Hola! Ha llamado al servicio de {BUSINESSNAME}. Me llamo Sara. Con gusto le ayudo con su cita. ¿Con quién tengo el gusto de hablar?",
        "has_Appt":   "Hola {first_name}! Bienvenido de nuevo a {BUSINESSNAME}. Es Sara. veo que tienes una cita el {appointment_date} a las {appointment_time}. Quieres reprogramarla o cómo puedo ayudarte?",
    },
    "fr": {
        "full":       "Bonjour {first}! Bienvenue chez {BUSINESSNAME}. c'est Sara. Je vois que vous souhaitez prendre un rendez-vous. Quel service désirez-vous pour votre {year} {model} ?",
        "no_vehicle": "Bonjour {first}! Bienvenue chez {BUSINESSNAME}. C'est] Sara. Je vois que vous souhaitez prendre un rendez-vous. Quelle est l’année, la marque et le modèle de votre véhicule ?",
        "no_name":    "Bonjour ! Vous avez joint {BUSINESSNAME}. C'est Sara. Je serai ravie de vous aider à prendre un rendez-vous. Avec qui ai-je le plaisir de parler ?",
        "has_Appt":   "Bonjour {first_name}! Bienvenue chez {BUSINESSNAME}. C'est Sara. je vois que vous avez un rendez-vous le {appointment_date} à {appointment_time}. Souhaitez-vous le reporter ou comment puis-je vous aider ?",
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


LANG_MAP = {
    "French": "fr",
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "Vietnamese": "vi",
    # add others as needed
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


SUPPORTED_STT_LANGS = {"en-US","es","fr","hi","vi","vi-VN","zh","zh-CN"}


UNSUPPORTED_REPLY = "Sorry, that language isn’t supported yet. I can transfer you to a person if you’d like."


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
    log.info("build_dynamic_greeting_and_next")

    customer, vehicle = _pick_state_objects(agent)

    first = (customer or {}).get("first") or (customer or {}).get("first_name") or ""
    last  = (customer or {}).get("last")  or (customer or {}).get("last_name")  or ""

    # vehicle fields with customer_data fallbacks
    cd    = getattr(agent, "customer_data", {}) or {}
    year  = (vehicle or {}).get("year")  or cd.get("car_year", "")
    make  = (vehicle or {}).get("make")  or cd.get("car_make", "")
    model = (vehicle or {}).get("model") or cd.get("car_model", "")

    # make sure we always have a business name
    business_name = (
        getattr(agent, "business_name", None)
        or globals().get("BUSINESSNAME")
        or "our service department"
    )

    has_name    = bool(first)
    has_vehicle = bool(year and model)
    has_existing_appointment = bool(cd.get("has_existing_appointment"))

    # template set with fallback to English
    tmpl = GREETING_TEMPLATES.get(lang) or GREETING_TEMPLATES.get("en", {})

    # small local formatter to avoid repeating kwargs
    def F(s: str) -> str:
        return s.format(
            first=first, last=last,
            year=year, make=make, model=model,
            BUSINESSNAME=business_name,   # supports {BUSINESSNAME}
            business_name=business_name,  # supports {business_name}
        )

    if has_existing_appointment and "reschedule" in tmpl:
        text = F(tmpl["reschedule"])
        next_state = "ask reschedule or cancel"

    elif has_name and has_vehicle and "full" in tmpl:
        text = F(tmpl["full"])
        next_state = "get service"

    elif has_name and "no_vehicle" in tmpl:
        text = F(tmpl["no_vehicle"])
        next_state = "get car"

    else:
        # fallback if "no_name" missing
        text = F(tmpl.get("no_name", "Hi{comma_first} welcome to {business_name}. May I have your first name, please?")
                   .replace("{comma_first}", f" {first}," if first else ","))
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

def _signal_ready(room_name: str, lang: str):
    # Matches your supervisor's wait file: /tmp/{room}-READY-{lang}
    try:
        Path(f"/tmp/{room_name}-READY-{lang}").touch(exist_ok=True)
        log.info(f"READY touched for room={room_name} lang={lang}")
    except Exception as e:
        log.warning(f"signal_ready failed: {e}")


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
    log.info(f"run_language_agent_entrypoint - lang {lang}")

    # Resolve room from ctx or env (supervisor sets HANDOFF_ROOM)
    room_obj  = getattr(ctx, "room", None)
    room_name = getattr(room_obj, "name", None) or os.getenv("HANDOFF_ROOM") or "unknown_room"
    ctx.log_context_fields = {"room": room_name}

    # Normalize language
    _norm = {"en-US": "en", "en": "en", "es": "es", "fr": "fr",
             "French": "fr", "English": "en", "Spanish": "es"}
    lang = _norm.get(lang, lang)
    assert lang in ("en", "es", "fr"), f"Unsupported lang: {lang}"
    DEEPGRAM_LANG = {"en": "en-US", "es": "es", "fr": "fr"}

    # --- singleflight with handoff bypass ---
    is_handoff_child = (os.getenv("HANDOFF_LANG") == lang)
    acquired_lock = False
    if not is_handoff_child:
        if not _singleflight_room(room_name):
            log.warning(f"[{lang}] duplicate entrypoint for room={room_name}; exiting")
            return
        acquired_lock = True
    else:
        log.info(f"[{lang}] handoff child detected; bypassing singleflight for room={room_name}")

    try:
        # Lookup seed by phone
        phone_number = extract_phone_from_room_name(room_name)
        found_cust = found_appt = False
        seed = {}
        if phone_number:
            try:
                found_cust, found_appt, seed = await lookup_customer_by_phone(
                    phone=phone_number, cars_url=CARS_URL
                )
                log.info(f"[{lang}] lookup: found_cust={found_cust} found_appt={found_appt}")
            except Exception as e:
                log.warning(f"[{lang}] lookup failed: {e}")

        # --------- robust VAD resolution (safe if missing) ----------
        vad_obj = None
        try:
            vad_obj = (getattr(getattr(ctx, "proc", None), "userdata", {}) or {}).get("vad")
            if not vad_obj:
                vad_obj = getattr(ctx, "userdata", {}).get("vad")
        except Exception:
            vad_obj = None
        if vad_obj:
            log.info(f"[{lang}] using provided VAD: {type(vad_obj).__name__}")
        else:
            log.info(f"[{lang}] no VAD provided; continuing without explicit VAD")

        # Build session
        session = AgentSession(
            llm=openai.LLM(model=LLM_MODEL),
            stt=deepgram.STT(model="nova-3", language=DEEPGRAM_LANG[lang], interim_results=True),
            tts=elevenlabs.TTS(
                voice_id=VOICE_BY_LANG[lang],
                model="eleven_flash_v2_5",
                voice_settings=VoiceSettings(
                    similarity_boost=0.75,
                    stability=0.5,
                    style=0.5,
                    use_speaker_boost=True
                ),
            ),
            turn_detection=MultilingualModel(),
            vad=vad_obj,  # may be None
            preemptive_generation=False,
        )

        install_speech_sanitizer(session)

        # --------- TTS sanitizer (times + "o’clock") installed globally ----------
        def _format_time_for_tts(text: str) -> str:
            """
            Normalize & remap times for TTS:
              - AM mapping:   12 -> 11, else hour -> hour + 1   (e.g., 7:00 AM -> 8 AM, 12:00 AM -> 11 AM)
              - PM mapping:   12 -> 12, else hour -> hour - 1   (e.g., 1:00 PM -> 12 PM, 12:00 PM -> 12 PM)
              - Remove 'o'clock' (including unicode apostrophes)
              - Collapse :00 minutes (keep non-zero minutes)
              - Normalize AM/PM variants (A.M., a.m., etc.)
              - Canonicalize 08 AM -> 8 AM

            Examples:
              "7:00 AM"        -> "8 AM"
              "12:00 AM"       -> "11 AM"
              "10:15 AM"       -> "11:15 AM"
              "1:00 PM"        -> "12 PM"
              "12:00 PM"       -> "12 PM"
              "8 o’clock a.m." -> "9 AM"  (AM rule: +1 hour, 8 -> 9)
            """
            if not isinstance(text, str):
                return text

            # --- Patterns ---
            AM_VARIANT = re.compile(r"\bA\s*\.?\s*M\.?\b", re.IGNORECASE)
            PM_VARIANT = re.compile(r"\bP\s*\.?\s*M\.?\b", re.IGNORECASE)
            AM_DOTTED  = re.compile(r"\b(AM|PM)\.", re.IGNORECASE)

            # e.g. "8 o'clock AM", "8 o’clock am", "8 o ’ clock PM"
            O_CLOCK    = re.compile(r"\b(\d{1,2})\s*o[’']?\s*clock\s*(AM|PM)\b", re.IGNORECASE)

            # e.g. "7:00 AM", "10:15 pm"
            TIME_AMPM  = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)\b", re.IGNORECASE)

            # e.g. "08 AM", "8AM"
            HOUR_AMPM  = re.compile(r"\b0?(\d{1,2})\s*(AM|PM)\b", re.IGNORECASE)

            # --- Helpers ---
            def _remap_hour(hour: int, ampm: str, minute: str | None) -> str:
                """Apply your mapping and format, dropping :00 minutes."""
                ampm_u = ampm.upper()
                h = hour

                if ampm_u == "AM":
                    if h == 12:
                        h = 11
                    else:
                        h = h + 1
                        if h == 13:
                            h = 1
                else:  # PM
                    if h == 12:
                        h = 12
                    else:
                        h = h - 1
                        if h == 0:
                            h = 12

                if minute is None or minute == "00":
                    return f"{h} {ampm_u}"
                return f"{h}:{minute} {ampm_u}"

            # --- 1) Normalize AM/PM variants ---
            text = AM_VARIANT.sub("AM", text)
            text = PM_VARIANT.sub("PM", text)
            text = AM_DOTTED.sub(lambda m: m.group(1).upper(), text)

            # --- 2) Handle "o'clock" hour-only cases first (they have no minutes) ---
            def _fix_oclock(m: re.Match) -> str:
                hour = int(m.group(1))
                ampm = m.group(2)
                return _remap_hour(hour, ampm, minute=None)
            text = O_CLOCK.sub(_fix_oclock, text)

            # --- 3) Handle hh:mm AM/PM (drop :00 after mapping; keep non-zero minutes) ---
            def _fix_hhmm(m: re.Match) -> str:
                hour = int(m.group(1))
                minute = m.group(2)
                ampm = m.group(3)
                return _remap_hour(hour, ampm, minute)
            text = TIME_AMPM.sub(_fix_hhmm, text)

            # --- 4) Handle bare hour + AM/PM (08 AM, 8AM) last, so we don't double-convert ---
            def _fix_hour(m: re.Match) -> str:
                hour = int(m.group(1))
                ampm = m.group(2)
                return _remap_hour(hour, ampm, minute=None)
            text = HOUR_AMPM.sub(_fix_hour, text)

            # --- 5) Tidy spaces ---
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            return text


        def _install_tts_sanitizer(sess):
            if getattr(sess, "_tts_sanitizer_installed", False):
                return
            sess._tts_sanitizer_installed = True

            def _sanitize_payload(p):
                if isinstance(p, str):
                    return _format_time_for_tts(p)
                if isinstance(p, dict):
                    if isinstance(p.get("instructions"), str):
                        p = dict(p)
                        p["instructions"] = _format_time_for_tts(p["instructions"])
                    r = p.get("response")
                    if isinstance(r, dict) and isinstance(r.get("instructions"), str):
                        p = dict(p)
                        rr = dict(r); rr["instructions"] = _format_time_for_tts(rr["instructions"])
                        p["response"] = rr
                return p

            # session.update
            if hasattr(sess, "update") and callable(sess.update):
                _orig_update = sess.update
                async def _patched_update(payload=None, **kw):
                    payload = _sanitize_payload(payload)
                    res = await _orig_update(payload, **kw)
                    # re-wrap response.create if SDK swaps it
                    resp_obj = getattr(sess, "response", None)
                    if resp_obj and hasattr(resp_obj, "create") and not getattr(resp_obj, "_tts_sanitizer_wrapped", False):
                        _wrap_response_create(sess)
                    return res
                sess.update = _patched_update  # type: ignore

            # session.say
            if hasattr(sess, "say") and callable(sess.say):
                _orig_say = sess.say
                async def _patched_say(text):
                    return await _orig_say(_format_time_for_tts(text))
                sess.say = _patched_say  # type: ignore

            # session.generate_reply
            if hasattr(sess, "generate_reply") and callable(sess.generate_reply):
                _orig_gen = sess.generate_reply
                async def _patched_gen(*args, **kw):
                    if args and isinstance(args[0], str):
                        args = (_format_time_for_tts(args[0]), *args[1:])
                    if "instructions" in kw and isinstance(kw["instructions"], str):
                        kw = dict(kw); kw["instructions"] = _format_time_for_tts(kw["instructions"])
                    return await _orig_gen(*args, **kw)
                sess.generate_reply = _patched_gen  # type: ignore

            # response.create
            def _wrap_response_create(s):
                robj = getattr(s, "response", None)
                if not robj or getattr(robj, "_tts_sanitizer_wrapped", False):
                    return
                _orig = robj.create
                async def _patched_create(payload):
                    return await _orig(_sanitize_payload(payload))
                robj.create = _patched_create  # type: ignore
                robj._tts_sanitizer_wrapped = True

            if getattr(sess, "response", None) and hasattr(sess.response, "create"):
                _wrap_response_create(sess)

        _install_tts_sanitizer(session)
        # --------- end sanitizer ----------

        # Choose prompt
        final_prompt = choose_prompt_by_flags(lang, found_cust, found_appt)
        if found_appt:
            final_prompt = FOUND_APPT_PROMPT
        log.info(f"final_prompt = {final_prompt}")

        # Build agent
        from .agent_base import AutomotiveBookingAssistant
        agent = AutomotiveBookingAssistant(
            session, ctx, None,
            instructions=final_prompt,
            lang=lang,
            supervisor=supervisor,
        )
        agent.lang = lang  # explicit
        if phone_number:
            agent._sip_participant_identity = f'sip_{phone_number}'
            agent.customer_data["phone"] = phone_number

        # Seed state
        if seed:
            agent.customer_data.update(seed)
            if found_appt and hasattr(agent, "set_current_state"):
                agent.set_current_state("cancel reschedule")
            elif found_cust and hasattr(agent, "set_current_state"):
                agent.set_current_state("get service")
            elif hasattr(agent, "set_current_state"):
                agent.set_current_state("get name")

        # Handlers/metrics/transcript
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

        # Start recording (safe to ignore failure pre-join)
        try:
            await start_recording(ctx, agent)
        except Exception as e:
            log.debug(f"[{lang}] start_recording pre-join failed (ok): {e}")

        # Join room
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
                close_on_disconnect=False
            ),
        )

        # Push tools/prompt after start
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

        # --- READY SIGNAL via HANDOFF_ROOM (robust) ---
        from pathlib import Path
        env_room = os.getenv("HANDOFF_ROOM") or room_name
        ready_path = Path(f"/tmp/{env_room}-READY-{lang}")
        log.info(f"[{lang}] touching READY flag: {ready_path}")

        ok = False
        for _ in range(5):
            try:
                ready_path.touch(exist_ok=True)
                if ready_path.exists():
                    ok = True
                    break
            except Exception as e:
                log.warning(f"[{lang}] READY touch failed: {e}")
            await asyncio.sleep(0.05)

        if not ok:
            log.error(f"[{lang}] FAILED to create READY flag at {ready_path}")
        else:
            log.info(f"[{lang}] READY flag created")

        # Ensure transport connected (idempotent)
        try:
            await ctx.connect()
        except Exception as e:
            log.warning(f"[{lang}] ctx.connect failed or already connected: {e}")

        # Greet (times already sanitized by the global wrapper)
        greet_text, next_state = build_dynamic_greeting_and_next(agent, lang)
        if hasattr(agent, "set_current_state"):
            agent.set_current_state(next_state)
        else:
            agent.state["progress"] = next_state

        log.info(f"run_language_agent_entrypoint - Greet text: {greet_text}")
        await agent.say(greet_text)

    finally:
        if acquired_lock:
            _release_room_lock(room_name)


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
        # Block any replies if agent is shutting down or handoff suppressed
        if getattr(agent, "_suppress_responses", False) or getattr(agent, "_is_shutting_down", False):
            log.info(f"[{tag}] false interruption ignored (handoff or shutdown in progress)")
            return

        log.info(f"[{tag}] false interruption → resume")
        try:
            session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)
        except Exception as e:
            log.warning(f"[{tag}] failed to resume after false interruption: {e}")


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
    log.info(f"lookup_customer_by_phone phone_number: {phone} url: {cars_url}")
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
    # await ctx.api.egress.start_room_composite_egress(req)


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
                            log.error(f"History send failed: {resp.status} {await resp.text()}")
            except Exception as api_err:
                log.error(f"Error sending to API: {api_err}")

        except Exception as e:
            log.error(f"Failed to write transcript: {e}")

    ctx.add_shutdown_callback(write_transcript)


def _last10(phone: str) -> str | None:
    digits = ''.join(filter(str.isdigit, phone or ""))
    return digits[-10:] if len(digits) >= 10 else None


def _lookup_cache_paths(phone10: str):
    base = f"/tmp/lookup.{phone10}"
    return Path(base + ".lock"), Path(base + ".json")


#FUNCTION DEPRECATED
async def lookup_customer_singleflight(phone_number: str):
    log.info(f"lookup_customer_singleflight phone_number: {phone_number}")
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



async def play_beeps(self, room: rtc.Room, *, count=2, freq=1000, duration=0.15, gap=0.15, volume=0.4):
    """
    Play short beeps into the room using the same pipeline as background audio.
    - count:   number of beeps
    - freq:    beep frequency in Hz
    - duration:seconds each beep lasts
    - gap:     seconds between beeps
    - volume:  0..1 amplitude
    """
    sample_rate = 48000
    samples_per_beep = int(sample_rate * duration)
    samples_gap = int(sample_rate * gap)
    frame_size = 960  # 20ms at 48kHz
    bytes_per_sample = 2

    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("beep", source)
    pub = await room.local_participant.publish_track(track)

    async def _beep_task():
        try:
            for _ in range(count):
                # Generate sine wave beep
                for n in range(0, samples_per_beep, frame_size):
                    frame_len = min(frame_size, samples_per_beep - n)
                    # 16-bit PCM samples
                    frame_data = bytearray()
                    for i in range(frame_len):
                        t = (n + i) / sample_rate
                        val = int(volume * 32767 * math.sin(2 * math.pi * freq * t))
                        frame_data += val.to_bytes(2, byteorder="little", signed=True)
                    frame = rtc.AudioFrame(
                        data=bytes(frame_data),
                        sample_rate=sample_rate,
                        num_channels=1,
                        samples_per_channel=frame_len,
                    )
                    await source.capture_frame(frame)
                    await asyncio.sleep(frame_len / sample_rate)
                # silence gap
                await asyncio.sleep(gap)
        finally:
            # unpublish when done
            await room.local_participant.unpublish_track(pub.sid)
            source.close()

    await _beep_task()


def install_speech_sanitizer(session):
    """Patch ALL outbound speech paths to normalize/shift times for TTS."""

    if getattr(session, "_speech_sanitizer_installed", False):
        return
    session._speech_sanitizer_installed = True

    # ---------- formatting logic ----------
    AM_VARIANT = re.compile(r"\bA\s*\.?\s*M\.?\b", re.IGNORECASE)
    PM_VARIANT = re.compile(r"\bP\s*\.?\s*M\.?\b", re.IGNORECASE)
    O_CLOCK    = re.compile(r"\b(\d{1,2})\s*o[’']?\s*clock\s*(AM|PM)\b", re.IGNORECASE)
    TIME_AMPM  = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)\b", re.IGNORECASE)
    HOUR_AMPM  = re.compile(r"\b0?(\d{1,2})\s*(AM|PM)\b", re.IGNORECASE)


    def _remap_hour(hour: int, ampm: str, minute: str | None) -> str:
        # Your requested mapping:
        # AM: 12 -> 11, else hour -> hour + 1
        # PM: 12 -> 12, else hour -> hour - 1
        ampm_u = ampm.upper()
        h = hour
        if ampm_u == "AM":
            if h == 12:
                h = 11
            else:
                h = h + 1
                if h == 13:
                    h = 1
        else:  # PM
            if h == 12:
                h = 12
            else:
                h = h - 1
                if h == 0:
                    h = 12
        if minute is None or minute == "00":
            return f"{h} {ampm_u}"
        return f"{h}:{minute} {ampm_u}"


    def _format_time_for_tts(text: str) -> str:
        """
        Normalize times so ElevenLabs doesn't speak 'o'clock':
          - '8:00 AM'  -> '8 AM'   (U+202F between hour and AM/PM)
          - '9:00 PM'  -> '9 PM'
          - '10:15 AM' -> '10:15 AM' (keeps minutes, inserts U+202F)
          - '8 o’clock AM'/'8 o'clock am' -> '8 AM'
          - 'A.M.'/'a.m.' → 'AM', 'P.M.'/'p.m.' → 'PM'
        """
        _HOUR_AMPM  = re.compile(r"\b0?(\d{1,2})\s*(AM|PM)\b", re.IGNORECASE)
        _TIME_AMPM = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b")
        _O_CLOCK   = re.compile(r"\b(\d{1,2})\s*o[’']?\s*clock\s*(AM|PM)\b", re.IGNORECASE)
        _AM_VARIANT= re.compile(r"\bA\s*\.?\s*M\.?\b", re.IGNORECASE)
        _PM_VARIANT= re.compile(r"\bP\s*\.?\s*M\.?\b", re.IGNORECASE)
        _AM_DOTTED = re.compile(r"\b(AM|PM)\.", re.IGNORECASE)

        if not isinstance(text, str):
            return text

        # 1) Normalize AM/PM variants
        text = _AM_VARIANT.sub("AM", text)
        text = _PM_VARIANT.sub("PM", text)

        # 2) Remove explicit "o'clock/o’clock" while preserving hour + AM/PM
        text = _O_CLOCK.sub(lambda m: f"{int(m.group(1))}\u202F{m.group(2).upper()}", text)

        # 3) On-the-hour: drop :00 and insert U+202F before AM/PM
        def _hhmm(m):
            h, mm, ap = int(m.group(1)), m.group(2), m.group(3).upper()
            if mm == "00":
                return f"{h}\u202F{ap}"
            return f"{h}:{mm}\u202F{ap}"
        text = _TIME_AMPM.sub(_hhmm, text)

        # 4) Bare hour + AM/PM (8 AM / 08AM) → insert U+202F
        text = _HOUR_AMPM.sub(lambda m: f"{int(m.group(1))}\u202F{m.group(2).upper()}", text)

        # 5) Clean regular spaces (do NOT touch U+202F)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        return text


    def _sanitize_any(obj: Any) -> Any:
        """Recursively sanitize strings in any dict/list/tuple payloads."""
        if isinstance(obj, str):
            out = _format_time_for_tts(obj)
            if out != obj:
                # Optional trace; comment out if too chatty
                # print(f"[tts-sanitize] '{obj}' -> '{out}'")
                pass
            return out
        if isinstance(obj, dict):
            return {k: _sanitize_any(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_any(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize_any(v) for v in obj)
        return obj

    # Helper to patch a callable attribute if present
    def _patch(obj, attr, wrapper_maker):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            orig = getattr(obj, attr)
            wrapped = wrapper_maker(orig)
            setattr(obj, attr, wrapped)
            return True
        return False

    # ---- Patch session.update ----
    _patch(session, "update", lambda orig: (lambda payload=None, **kw: orig(_sanitize_any(payload), **kw)))

    # ---- Patch session.say ----
    _patch(session, "say",    lambda orig: (lambda text: orig(_format_time_for_tts(text))))

    # ---- Patch session.generate_reply ----
    def _wrap_generate_reply(orig):
        async def _gen(*args, **kw):
            args = list(args)
            if args and isinstance(args[0], str):
                args[0] = _format_time_for_tts(args[0])
            if "instructions" in kw and isinstance(kw["instructions"], str):
                kw = dict(kw); kw["instructions"] = _format_time_for_tts(kw["instructions"])
            return await orig(*args, **kw)
        return _gen
    if hasattr(session, "generate_reply") and callable(session.generate_reply):
        session.generate_reply = _wrap_generate_reply(session.generate_reply)

    # ---- Patch response.create (common for Realtime/agents) ----
    def _wrap_response_create(sess):
        robj = getattr(sess, "response", None)
        if not robj or getattr(robj, "_tts_sanitizer_wrapped", False):
            return
        _orig = robj.create
        async def _create(payload):
            return await _orig(_sanitize_any(payload))
        robj.create = _create  # type: ignore
        robj._tts_sanitizer_wrapped = True
    if getattr(session, "response", None) and hasattr(session.response, "create"):
        _wrap_response_create(session)

    # ---- Patch session.send_event (many SDKs route response.create this way) ----
    def _wrap_send_event(orig):
        async def _send(payload):
            return await orig(_sanitize_any(payload))
        return _send
    if hasattr(session, "send_event") and callable(session.send_event):
        session.send_event = _wrap_send_event(session.send_event)

    # ---- Patch raw websocket JSON (last resort path) ----
    ws = getattr(session, "ws", None)
    if ws and hasattr(ws, "send_json") and callable(ws.send_json):
        orig = ws.send_json
        async def _send_json(payload):
            return await orig(_sanitize_any(payload))
        ws.send_json = _send_json  # type: ignore