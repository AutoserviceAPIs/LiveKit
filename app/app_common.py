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


def resolve_lang_code(text: str, default: str = "en-US") -> str:
    if not isinstance(text, str):
        return default
    t = text.lower()
    for code, words in LANG_SYNONYMS.items():
        if any(w in t for w in words):
            return code
    return default


def prewarm(proc: JobProcess):
    if "vad" not in proc.userdata:
        proc.userdata["vad"] = silero.VAD.load()
    if "llm" not in proc.userdata:
        proc.userdata["llm"] = openai.LLM(model=LLM_MODEL)