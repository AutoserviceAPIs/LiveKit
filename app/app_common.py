# agent runtime wiring shared by all languages.
# builds the AgentSession (LLM/STT/TTS) for a given lang
# attaches common event handlers (timeouts, metrics, logging)
# reads handoff snapshot, signals READY, plays background, sends greeting
# exposes run_language_agent_entrypoint(ctx, lang)

from __future__ import annotations  # optional, but nice to have in 3.11+
from dotenv import load_dotenv
from livekit import api
from livekit.agents import JobContext, AgentSession
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from datetime import datetime, timedelta
import json, logging, aiohttp, re, os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent_base import AutomotiveBookingAssistant

log = logging.getLogger("agent_common")
load_dotenv(".env")
  
VOICE_BY_LANG = { 
    "en" :   "aEO01A4wXwd1O8GPgGlF", #Arabella
    "fr" :   "aEO01A4wXwd1O8GPgGlF", #Arabella
    "es" :   "aEO01A4wXwd1O8GPgGlF", #Arabella
    "vi" :   "aEO01A4wXwd1O8GPgGlF", #Arabella
    "hi" :   "aEO01A4wXwd1O8GPgGlF", #Arabella
#    "en":    "zmcVlqmyk3Jpn5AVYcAL",
#    "es":    "jB2lPb5DhAX6l1TLkKXy",
#    "fr":    "BewlJwjEWiFLWoXrbGMf",
#    "hi":    "CpLFIATEbkaZdJr01erZ",
#    "vi":    "8h6XlERYN1nW5v3TWkOQ",
#    "zh":    "bhJUNIXWQQ94l8eI2VUf",
#    "cn":    "bhJUNIXWQQ94l8eI2VUf",
}



COMMON_PROMPT = """You are a booking assistant. Help customers book appointments.

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer (We already have customer phone number): returns customer name, car details, or booking details.

## RULES:
- After collecting car year make and model: call save_customer_information
- After collecting services and transportation: call save_services_detail
- After booking: call create_appointment
- Do not say things like "Let me save your information" or "Please wait." Just proceed silently to next step
- For recall, reschedule appointment or cancel appointment: call transfer_call
- For speak with someone, customer service, or user is frustrated: call transfer_call

- For address: 80 Queens Plate Dr, Etobicoke
- For price: oil change starts at $130 plus tax
- For Wait time: 45 minutes to 1 hour
- Only If user asks if you are a "human" a real person: say "I am actually a voice AI assistant to help you with your service appointment", then Repeat last question

## Follow this conversation flow:

Step 1. Gather First and Last Name
- If customer name and car details found: Hello {first_name}! welcome back to Woodbine Toyota. My name is Sara. I see you are calling to schedule an appointment. What service would you like for your {year} {model}?. Proceed to Step 3
- If car details not found: Hello {first_name}! welcome back to Woodbine Toyota. My name is Sara. I see you are calling to schedule an appointment. What is your car's year, make, and model?. Proceed to Step 2
- If customer name not found: Hello! You reached Woodbine Toyota Service. My name is Sara. I'll be glad to help with your appointment. Who do I have the pleasure of speaking with?

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
- call check_available_slots tool
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
                               business_name="Woodbine Toyota",
                               project_id="woodbine_toyota",
                               hours_location="80 Queens Plate Dr, Etobicoke",
                               api_url="https://voxbackend1-cx37agkkhq-uc.a.run.app/add-history"):
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