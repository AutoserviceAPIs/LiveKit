# agent runtime wiring shared by all languages.
# builds the AgentSession (LLM/STT/TTS) for a given lang
# attaches common event handlers (timeouts, metrics, logging)
# reads handoff snapshot, signals READY, plays background, sends greeting
# exposes run_language_agent_entrypoint(ctx, lang)

from __future__ import annotations  # optional, but nice to have in 3.11+
from .agent_customerdata import (INSTRUCTIONS_RECALL, INSTRUCTIONS_CANCEL_RESCHEDULE, BUSINESSNAME, BUSINESSLOCATION, INSTRUCTIONS_PRICING, INSTRUCTIONS_WAITTIME, CARS_URL, CHECK_URL, BOOK_URL, HISTORY_URL)
from livekit import api
from livekit.agents import JobContext, AgentSession
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from datetime import datetime, timedelta
import json, logging, aiohttp, re, os
from dotenv import load_dotenv
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent_base import AutomotiveBookingAssistant

log = logging.getLogger("agent_common")
load_dotenv(".env")
  
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


#- If customer name and car details found: Hello {first_name}! welcome back to <business name> service. My name is Sara. I see you are calling to schedule an appointment. What service would you like for your {year} {model}?. Proceed to Step 3
#- If car details not found: Hello {first_name}! welcome back to <business name> service. My name is Sara. I see you are calling to schedule an appointment. What is your car's year, make, and model?. Proceed to Step 2
#- If customer name not found: Hello! You reached <business name> Service. My name is Sara. I'll be glad to help with your appointment. Who do I have the pleasure of speaking with?
#- Hello {first_name}, I see you have an appointment coming up on {appointment_date} at {appointment_time}. Would you like to reschedule your appointment, or could I help you with something else?

COMMON_PROMPT = f"""You are a booking assistant. Help customers book appointments.

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer (We already have customer phone number): returns customer name, car details, or booking details.

## RULES:
- After collecting car year make and model: call save_customer_information
- After collecting services and transportation: call save_services_detail
- After booking: call create_appointment
- Do not say things like "Let me save your information" or "Please wait." Just proceed silently to next step
- {INSTRUCTIONS_RECALL}
- {INSTRUCTIONS_CANCEL_RESCHEDULE}
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


LLM_MODEL    = "gpt-4o-mini"


LANG_TO_DG = {
    "english": "en-US", "en": "en-US",
    "french": "fr", "français": "fr", "fr": "fr",
    "spanish": "es", "español": "es", "es": "es",
    "hindi": "hi", "hi": "hi",
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

DG_CODE = {
    "en-US": "en-US",
    "en-US": "en",
    "es":    "es",
    "fr":    "fr",
    "hi":    "hi",
    "vi":    "vi",
    "zh":    "zh",
}


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
        found_cust, found_appt, seed = await lookup_customer_by_phone(phone_number)
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
        return await lookup_customer_by_phone(phone_number)
