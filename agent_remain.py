#TODO:
#VAD: reduce sensitivity / languages
#Languages

#Fixed hangup
#User does not know service / are you real?
#Fixed timeout: transfer call on silence
#Moved transfer_to to self._transfer_to
#I set preemptive_generation=False,

import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN, Agent, UserStateChangedEvent, AgentFalseInterruptionEvent,
    AgentSession, JobContext, JobProcess, MetricsCollectedEvent,
    RoomInputOptions, RunContext, WorkerOptions, cli, metrics, get_job_context,)
from livekit import agents
from livekit import rtc
from livekit import api
from livekit.agents.llm import function_tool
from livekit.agents.metrics import TTSMetrics
from livekit.protocol import room as room_msgs  # ✅ protocol types live here
from livekit.protocol.sip import TransferSIPParticipantRequest
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from livekit.plugins.elevenlabs import VoiceSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import agents
logger = logging.getLogger("agent")
import os
load_dotenv(".env")
import asyncio
import time
import subprocess
import re

logger = logging.getLogger("agent")
load_dotenv(".env")


DEFAULT_LANG      = "en-US"   

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

DG_CODE = {
    "en-US": "en-US",
    "es":    "es",
    "fr":    "fr",
    "hi":    "hi",
    "vi":    "vi",
    "zh":    "zh",
}
      

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


def resolve_lang_code(text: str, default="en-US") -> str:
    t = (text or "").lower()
    for code, words in LANG_SYNONYMS.items():
        if any(w in t for w in words):
            return code
    return default


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



            

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    lang_switch_q = asyncio.Queue()
    # Create timestamp once to use for both recording and transcript
    recording_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up recording for audio
    try:
        req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[api.EncodedFileOutput(
                file_type=api.EncodedFileType.OGG,
                filepath=f"./{ctx.room.name}_{recording_timestamp}.ogg",
                s3=api.S3Upload(
                    bucket=os.getenv('S3_BUCKET_NAME', 'recording-autoservice'),
                    region=os.getenv('S3_REGION', 'us-east-1'),
                    access_key=os.getenv('S3_ACCESS_KEY'),
                    secret=os.getenv('S3_SECRET_KEY'),
                ),
            )],
        )

        livekit_url = os.getenv('LIVEKIT_URL')
        api_key = os.getenv('LIVEKIT_API_KEY')
        api_secret = os.getenv('LIVEKIT_API_SECRET')

        async with api.LiveKitAPI(
            url=livekit_url,
            api_key=api_key,
            api_secret=api_secret
        ) as lkapi:
            res = await lkapi.egress.start_room_composite_egress(req)
            logger.info(f"Recording started with egress ID: {res.egress_id}")
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language=DEFAULT_LANG, interim_results=True),
        #stt=deepgram.STT(model="nova-3", language=DEFAULT_LANG, interim_results=True, api_key=DEEPGRAM_API_KEY),
        #stt=deepgram.STT(model="nova-3", language="en-US", api_key="DEEPGRAM_API_KEY=fab0ddff7c9cff66683da19af03b5020d2ec7ad6"),
        tts=elevenlabs.TTS(
            voice_id=VOICE_BY_LANG[DEFAULT_LANG], #sapphire
            #voice_id="xcK84VTjd6MHGJo2JVfS",#cloned
            model="eleven_flash_v2_5",
            api_key="59bb59df13287e23ba2da37ea6e48724",
            voice_settings= VoiceSettings(
                similarity_boost=0.4,
                speed=1,
                stability=0.3,
                style=1,
                use_speaker_boost=True
            )
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=False,
    )


    @session.on("transcription")
    def _tx(ev):
        # ev has .text and .is_final in current SDKs
        try:
            lang = getattr(ev, "language", None)
            print(f"STT lang={lang} final={ev.is_final} text={ev.text!r}")
        except Exception:
            print(f"STT final={getattr(ev,'is_final',None)} text={getattr(ev,'text',None)}")


    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "speaking":
            agent.cancel_timeout()
            logger.info(f"User state changed: {ev.new_state} - Cancel Timeout")
        if ev.new_state == "away":  
            agent.start_timeout()
            logger.info(f"User state changed: {ev.new_state} - Restart Timeout")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        """Log when conversation items are added"""
        logger.info(f"Conversation item added: {ev.item.type} - {ev.item.content[0]}...")

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        """Log when user input is transcribed"""
        logger.info(f"User input transcribed: {ev.transcript}")

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("transcript")
    def _on_transcript(ev):
        txt = getattr(ev, "text", "") or ""
        if txt.strip():
            agent.cancel_timeout()

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        # Check if this is TTS metrics and appointment was created
        if agent.appointment_created and isinstance(ev.metrics, TTSMetrics):
            print("TTS metrics collected")
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                hangup_delay = audio_duration + 0.5  # Add 0.5s buffer
                logger.info(f"TTS audio duration: {audio_duration}s, scheduling hangup in {hangup_delay}s")
                asyncio.create_task(agent.delayed_hangup(hangup_delay))

        # Restart timeout after each TTS response (agent finished speaking)
        if isinstance(ev.metrics, TTSMetrics):
            # Start timeout after audio duration finishes
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                asyncio.create_task(agent.delayed_timeout_start(audio_duration))
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    async def write_transcript():
        """Save conversation history to JSON file and send to API"""
        try:
            # Use the same timestamp as recording
            current_date = agent._recording_timestamp if hasattr(agent, '_recording_timestamp') else datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./transcript_{ctx.room.name}_{current_date}.json"
            
            # Extract phone number from room name
            phone_number = extract_phone_from_room_name(ctx.room.name)
            formatted_phone = f"1{phone_number}" if phone_number else ""
            
            # Extract conversations from session history
            conversations = ["TELEPHONY_WELCOME"]
            if hasattr(session, 'history') and session.history:
                history_dict = session.history.to_dict()
                if 'items' in history_dict:
                    for item in history_dict['items']:
                        if 'content' in item and item['content']:
                            # Get the first content element (usually the text)
                            content = item['content'][0] if isinstance(item['content'], list) else str(item['content'])
                            conversations.append(content)
            
            # Create record URL
            record_filename = f"{ctx.room.name}_{current_date}.ogg"
            record_url = f"https://recording-autoservice.s3.us-east-1.amazonaws.com/{record_filename}"
            
            # Create payload for API
            payload = {
                "phone": formatted_phone,
                "agentId": 1,
                "conversations": conversations,
                "recordURL": record_url,
                "transfer": agent._call_already_transferred,
                "voicemail": False,  # You can modify this based on your logic
                "booked": agent.appointment_created,
                "bookingintent": True,  # You can modify this based on your logic
                "perfect": True,  # You can modify this based on your logic
                "businessName": "Woodbine Toyota",
                "customerName": f"{agent.customer_data.get('first_name', '')} {agent.customer_data.get('last_name', '')}".strip(),
                "serviceOriginal": ", ".join(agent.customer_data.get('services', [])),
                "carModel": agent.customer_data.get('car_model', ''),
                "carYear": agent.customer_data.get('car_year', ''),
                "carMake": agent.customer_data.get('car_make', ''),
                "advisorNumber": "",
                "havetimeslots": len(agent.available_slots) > 0,
                "project_id": "woodbine_toyota",
                "check_url": "",
                "book_url": "",
                "transportation_drop": agent.customer_data.get('transportation', ''),
                "fallback_service_default": "",
                "hoursLocation": "80 Queens Plate Dr, Etobicoke"
            }
            
            # Save local transcript file
            conversation_data = {
                "room_name": ctx.room.name,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": session.history.to_dict(),
                "customer_data": agent.customer_data,
                "appointment_created": agent.appointment_created,
                "call_transferred": agent._call_already_transferred,
                "api_payload": payload
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Transcript for {ctx.room.name} saved to {filename}")
            
            # Send to API
            try:
                async with aiohttp.ClientSession() as session_http:
                    async with session_http.post(
                        "https://voxbackend1-cx37agkkhq-uc.a.run.app/add-history",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"History sent to API successfully: {result}")
                        else:
                            logger.error(f"Failed to send history to API: {response.status}")
                            logger.error(f"Response: {await response.text()}")
            except Exception as api_error:
                logger.error(f"Error sending to API: {api_error}")
                    
        except Exception as e:
            logger.error(f"Failed to write transcript: {e}")

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(write_transcript)

    # Create agent instance
    agent = AutomotiveBookingAssistant(session, ctx, lang_switch_q)
    # Store context reference for shutdown
    agent._ctx = ctx
    # Store recording timestamp for transcript
    agent._recording_timestamp = recording_timestamp

    # Try to find customer by phone number if available
    phone_number = extract_phone_from_room_name(ctx.room.name)
    if phone_number:
        logger.info(f"Attempting to find customer with phone: {phone_number}")
        agent._sip_participant_identity = f'sip_{phone_number}'

        found = await agent.findCustomer(phone_number)
        if found:
            agent.set_current_state("get service")
        else:
            logger.info("Customer not found, will proceed with normal flow")
    
    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False  # Don't close immediately after transfer
        ),

    )
    await agent.start_background(ctx.room, "office.mp3")
    await session.generate_reply(
        instructions="Greet user"
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    #agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="my-telephony-agent"))