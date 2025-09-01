from .agent_common import build_session, prewarm, extract_phone_from_room_name, start_recording, VOICE_BY_LANG, COMMON_PROMPT
from .agent_base import AutomotiveBookingAssistant
import logging, aiohttp, json, os, asyncio
from datetime import datetime, timedelta, time, subprocess, re
from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN, Agent, UserStateChangedEvent, AgentFalseInterruptionEvent,
    AgentSession, JobContext, JobProcess, MetricsCollectedEvent,
    RoomInputOptions, RunContext, WorkerOptions, cli, metrics, get_job_context,)
from livekit import api
from livekit.agents.llm import function_tool
from livekit.agents.metrics import TTSMetrics
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from livekit.plugins.elevenlabs import VoiceSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import Dict, List, Optional
from livekit import agents, rtc
from livekit.protocol import room as room_msgs  # ✅ protocol types live here
from livekit.protocol.sip import TransferSIPParticipantRequest
from pathlib import Path

DEFAULT_LANG = "vi"   
LLM_MODEL    = "gpt-4o-mini"

logger = logging.getLogger("agent_vi")
load_dotenv(".env")

MY_PROMPT = f"""You are a professional receptionist for Woodbine Toyota. Answer only in Vietnamese. Help customers book appointments.
{COMMON_PROMPT}"""


async def warm_greeting(self):
    logger.info(f"warm_greeting")
    name = (self.state.get("customer") or {}).get("first")
    if name:
        await self.say(f"Xin chào {name}, bạn có muốn tiếp tục bằng tiếng Việt không?")
    else:
        await self.say("Bạn có muốn tiếp tục bằng tiếng Việt không")


async def entrypoint(ctx: JobContext):
    logger.info("entrypoint starting")
    # DO NOT create a new lang_switch_q here; the supervisor owns that.
    # If you need supervisor-passed data, read it from ctx.proc.userdata:
    init_state = ctx.proc.userdata.get("state_snapshot") if hasattr(ctx, "proc") else None

    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=openai.LLM(model=LLM_MODEL),
        stt=deepgram.STT(model="nova-3", language=DEFAULT_LANG, interim_results=True),  # <-- Spanish STT
        tts=elevenlabs.TTS(
            voice_id=VOICE_BY_LANG[DEFAULT_LANG],   # <-- map "es" to a Spanish voice
            model="eleven_flash_v2_5",
            voice_settings=VoiceSettings(similarity_boost=0.4, speed=1, stability=0.3, style=1, use_speaker_boost=True)
        ),
        turn_detection=MultilingualModel(),  # fine to keep multilingual
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    # --- logging hooks (unchanged) ---
    @session.on("transcription")
    def _tx(ev):
        try:
            lang = getattr(ev, "language", None)
            print(f"[ES] STT lang={lang} final={ev.is_final} text={ev.text!r}")
        except Exception:
            print(f"[ES] STT final={getattr(ev,'is_final',None)} text={getattr(ev,'text',None)}")

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "speaking":
            agent.cancel_timeout()
            logger.info("[ES] User is speaking - cancel timeout")
        if ev.new_state == "away":
            agent.start_timeout()
            logger.info("[ES] User is away - restart timeout")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        logger.info(f"[ES] Conversation item: {ev.item.type} - {ev.item.content[0]}...")

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        logger.info(f"[ES] User input transcribed: {ev.transcript}")

    @session.on("transcript")
    def _on_transcript(ev):
        txt = getattr(ev, "text", "") or ""
        if txt.strip():
            agent.cancel_timeout()

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("[ES] false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        if agent.appointment_created and isinstance(ev.metrics, TTSMetrics):
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                hangup_delay = audio_duration + 0.5
                logger.info(f"[ES] TTS audio duration: {audio_duration}s, scheduling hangup in {hangup_delay}s")
                asyncio.create_task(agent.delayed_hangup(hangup_delay))

        if isinstance(ev.metrics, TTSMetrics):
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                asyncio.create_task(agent.delayed_timeout_start(audio_duration))

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"[ES] Usage: {summary}")

    async def write_transcript():
        try:
            current_date = agent._recording_timestamp if hasattr(agent, '_recording_timestamp') else datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./transcript_{ctx.room.name}_{current_date}.json"

            phone_number = extract_phone_from_room_name(ctx.room.name)
            formatted_phone = f"1{phone_number}" if phone_number else ""

            conversations = ["TELEPHONY_WELCOME"]
            if hasattr(session, 'history') and session.history:
                history_dict = session.history.to_dict()
                if 'items' in history_dict:
                    for item in history_dict['items']:
                        if 'content' in item and item['content']:
                            content = item['content'][0] if isinstance(item['content'], list) else str(item['content'])
                            conversations.append(content)

            record_filename = f"{ctx.room.name}_{current_date}.ogg"
            record_url = f"https://recording-autoservice.s3.us-east-1.amazonaws.com/{record_filename}"

            payload = {
                "phone": formatted_phone,
                "agentId": 1,
                "conversations": conversations,
                "recordURL": record_url,
                "transfer": agent._call_already_transferred,
                "voicemail": False,
                "booked": agent.appointment_created,
                "bookingintent": True,
                "perfect": True,
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

            logger.info(f"[ES] Transcript for {ctx.room.name} saved to {filename}")

            try:
                async with aiohttp.ClientSession() as session_http:
                    async with session_http.post(
                        "https://voxbackend1-cx37agkkhq-uc.a.run.app/add-history",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"[ES] History sent to API successfully: {result}")
                        else:
                            logger.error(f"[ES] Failed to send history to API: {response.status}")
                            logger.error(f"[ES] Response: {await response.text()}")
            except Exception as api_error:
                logger.error(f"[ES] Error sending to API: {api_error}")

        except Exception as e:
            logger.error(f"[ES] Failed to write transcript: {e}")

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(write_transcript)

    # --- Create the agent with Spanish prompt + lang ---
    agent = AutomotiveBookingAssistant(session, ctx, None, instructions=ES_PROMPT, lang=DEFAULT_LANG)
    agent._ctx = ctx

    # Restore the handoff snapshot if provided
    try:
        agent.restore_state(init_state)
    except Exception as e:
        logger.warning(f"[ES] restore_state failed: {e}")

    await start_recording(ctx, agent)

    # Optional: SIP identity, same as EN
    phone_number = extract_phone_from_room_name(ctx.room.name)
    if phone_number:
        agent._sip_participant_identity = f'sip_{phone_number}'

    # Start the realtime session in this room
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False
        ),
    )

    # --- SIGNAL READY to the supervisor (choose ONE method you actually use) ---
    # A) Sentinel file:
    try:
        Path(f"/tmp/{ctx.room.name}-READY-es").touch()
        logger.info("[ES] READY sentinel created")
    except Exception as e:
        logger.warning(f"[ES] READY sentinel failed: {e}")

    # B) OR if your supervisor expects a datachannel/queue signal, send it here instead.

    # Play background ambience if needed
    await agent.start_background(ctx.room, "office.mp3")

    # Greet in Spanish (don’t send English prompt here)
    await agent.warm_greeting()

    # Connect/enter main loop
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))