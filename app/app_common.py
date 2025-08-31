#TODO:
#VAD: reduce sensitivity / languages
#Languages

#Fixed hangup
#User does not know service / are you real?
#Fixed timeout: transfer call on silence
#Moved transfer_to to self._transfer_to
#I set preemptive_generation=False,

from __future__ import annotations  # optional, but nice to have in 3.11+
import logging

from livekit.agents import JobContext, AgentSession
from dotenv import load_dotenv
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from livekit import api
from datetime import datetime, timedelta
import re
import os


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent_base import AutomotiveBookingAssistant

logger = logging.getLogger("agent")
load_dotenv(".env")
  
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
    # Chinese (Mandarin). Avoid non-standard 'cn' but accept it as alias.
    "zh": ["chinese", "mandarin", "zh", "zh-cn", "中文", "普通话", "国语", "cn"],
}

CONFIRM_BY_LANG = {
    "en-US": "Okay, I’ll continue in English.",
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
    "es":    "es",
    "fr":    "fr",
    "hi":    "hi",
    "vi":    "vi",
    "zh":    "zh",
}


VOICE_BY_LANG = {
    "en-US": "zmcVlqmyk3Jpn5AVYcAL",
    "es":    "jB2lPb5DhAX6l1TLkKXy",
    "fr":    "BewlJwjEWiFLWoXrbGMf",
    "hi":    "CpLFIATEbkaZdJr01erZ",

    # Vietnamese (standard + regional alias)
    "vi":    "8h6XlERYN1nW5v3TWkOQ",
    "vi-VN": "8h6XlERYN1nW5v3TWkOQ",

    # Chinese (standard + mainland alias). Avoid 'cn' (non-standard), but keep it as a fallback if you like.
    "zh":    "bhJUNIXWQQ94l8eI2VUf",
    "zh-CN": "bhJUNIXWQQ94l8eI2VUf",
    # Optional backward-compat alias:
    "cn":    "bhJUNIXWQQ94l8eI2VUf",
}


def build_session(ctx: JobContext, lang: str) -> AgentSession:
    llm = ctx.proc.userdata.get("llm") or openai.LLM(model=LLM_MODEL)
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    stt_lang = lang if lang in SUPPORTED_STT_LANGS else "en-US"
    stt = deepgram.STT(
        model="nova-3",
        language=stt_lang,
        interim_results=True,
        api_key=os.environ["DEEPGRAM_API_KEY"],
    )
    tts = elevenlabs.TTS(
        voice_id=VOICE_BY_LANG.get(stt_lang, VOICE_BY_LANG["en-US"]),
        model="eleven_flash_v2_5",
        api_key=os.environ["ELEVEN_API_KEY"],
        voice_settings=VoiceSettings(
            similarity_boost=0.4, speed=1.0, stability=0.3, style=1.0, use_speaker_boost=True
        ),
    )

    return AgentSession(
        llm=llm,
        stt=stt,
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=vad,
        preemptive_generation=False,
    )


# Extract phone number from room name
def extract_phone_from_room_name(room_name: str) -> str:
    """Extract phone number from room name format: call-_8055057710_uHsvtynDWWJN"""
    pattern = r'call-_(\d+)_'
    match = re.search(pattern, room_name)
    return match.group(1) if match else ""
            

def make_session(lang: str, ctx) -> AgentSession:
    return AgentSession(
        llm=ctx.llm,
        stt=deepgram.STT(
            model="nova-3",
            language=lang,
            interim_results=True,
            api_key=os.environ["DEEPGRAM_API_KEY"],
        ),
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice_id=VOICE_BY_LANG.get(lang, VOICE_BY_LANG["en-US"]),
            api_key=os.environ["ELEVEN_API_KEY"],
        ),
        turn_detection=ctx.turn_detection,
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        instructions="You are a voice AI assistant for Woodbine Toyota…",
    )


async def handle_user_text(self, text: str | None, is_final: bool = True):
    logger.info("STT is_final=%s text=%s", is_final, text)
    s = (text or "").strip()
    if not s:
        return

    # 1) Fast path: detect a language **request** (works on partials or finals)
    code = detect_requested_lang(s)               # returns e.g. "fr","es","hi","vi","zh","en-US" or None
    current = getattr(self, "_current_lang", "en-US")
    if code and code != current:
        await self.set_language(code)             # <-- pass the CODE
        self._current_lang = code
        return  # let the next utterance be recognized with the new STT

    # 2) On final transcripts, try a more permissive resolve (aliases, miss-hearings)
    if is_final:
        code2 = resolve_lang_code(s)              # same normalized codes or None
        if code2 and code2 != current:
            await self.set_language(code2)        # <-- pass the CODE
            self._current_lang = code2
            return

    # 3) …continue with your normal handling here…
    # await self.route_user_intent(s)


async def handle_user_text(self, text: str, is_final: bool = True):
    logger.info(f"handle_user_text - str: {str}")
    # 1) Detect requested language
    code = resolve_lang_code(text)             # <-- now 'code' is defined here
    if code and code != getattr(self, "_current_lang", "en-US"):
        await self.set_language(code)          # pass the CODE, not the phrase
        self._current_lang = code
        return  # let the next utterance use the new STT


async def lang_switch_loop(session_builder, agent, lang_q):
    logger.info(f"lang_switch_loop - lang_q: {lang_q}")
    current_session = await session_builder("en-US")
    agent.bind_session(current_session)
    while True:
        code = await lang_q.get()
        await current_session.aclose()                 # stop current pipelines
        current_session = await session_builder(code)  # build with new STT/TTS
        agent.bind_session(current_session)            # re-bind


async def start_recording(ctx: JobContext, agent):
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


def normalize_lang_code(text: str) -> str:
    s = (text or "").strip().lower()
    # direct tags first
    if s in ("en-us","en"): return "en-US"
    for code, words in LANG_SYNONYMS.items():
        if s in words:
            return "en-US" if code == "en" else code
    # crude keyword fallbacks
    if any(k in s for k in ["franc", "franç", "french"]): return "fr"
    if any(k in s for k in ["espa", "spani"]):            return "es"
    if "हिं" in s or "hindi" in s:                         return "hi"
    if "viet" in s or "việt" in s:                         return "vi"
    if "中文" in s or "mandarin" in s or "zh" in s:         return "zh"
    return "en-US"


def detect_requested_lang(text: str) -> str | None:
    logger.info(f"detect_requested_lang - str: {str}")
    s = (text or "").lower()
    for code, words in LANG_SYNONYMS.items():
        if any(w in s for w in words):
            return "en-US" if code == "en" else code
    return None


def resolve_lang_code(lang_phrase: str) -> str | None:
    p = (lang_phrase or "").strip().lower()
    for code, words in LANG_SYNONYMS.items():
        if p in words or any(w in p for w in words):
            return code
    return None


def prewarm(proc: JobProcess):
    if "vad" not in proc.userdata:
        proc.userdata["vad"] = silero.VAD.load()
    if "llm" not in proc.userdata:
        proc.userdata["llm"] = openai.LLM(model=LLM_MODEL)