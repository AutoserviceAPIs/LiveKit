from .app_common import build_session, prewarm, extract_phone_from_room_name, start_recording, VOICE_BY_LANG, COMMON_PROMPT
from .agent_base import AutomotiveBookingAssistant
import logging
import aiohttp
from datetime import datetime, timedelta
import json
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
import os
import asyncio
from typing import Dict, List, Optional
from livekit import agents
from livekit import rtc
from livekit.protocol import room as room_msgs  # ✅ protocol types live here
from livekit.protocol.sip import TransferSIPParticipantRequest
import time
import subprocess
import re

DEFAULT_LANG = "es"   
LLM_MODEL    = "gpt-4o-mini"

logger = logging.getLogger("agent_es")
load_dotenv(".env")

MY_PROMPT = f"""You are a professional receptionist for Woodbine Toyota. Answer only in Spanish. Help customers book appointments.
{COMMON_PROMPT}"""


async def warm_greeting(self):
    name = (self.state.get("customer") or {}).get("first")
    if name:
        await self.say(f"Hola {name}, ¿prefiere continuar en español?")
    else:
        await self.say("Hola, ¿prefiere continuar en español?")


async def entrypoint(ctx: JobContext):
    lang_switch_q = asyncio.Queue()
    
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=openai.LLM(model=LLM_MODEL),
        stt=deepgram.STT(model="nova-3", language=DEFAULT_LANG, interim_results=True),
        tts=elevenlabs.TTS(
            voice_id=VOICE_BY_LANG[DEFAULT_LANG], #sapphire
            #voice_id="xcK84VTjd6MHGJo2JVfS",#cloned
            model="eleven_flash_v2_5",
            voice_settings= VoiceSettings(similarity_boost=0.4, speed=1, stability=0.3, style=1, use_speaker_boost=True)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    @session.on("transcription")
    def _tx(ev):
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
        logger.info(f"Conversation item added: {ev.item.type} - {ev.item.content[0]}...")

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        logger.info(f"User input transcribed: {ev.transcript}")

    @session.on("transcript")
    def _on_transcript(ev):
        txt = getattr(ev, "text", "") or ""
        if txt.strip():
            agent.cancel_timeout()

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        if agent.appointment_created and isinstance(ev.metrics, TTSMetrics):
            print("TTS metrics collected")
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                hangup_delay = audio_duration + 0.5  # Add 0.5s buffer
                logger.info(f"TTS audio duration: {audio_duration}s, scheduling hangup in {hangup_delay}s")
                asyncio.create_task(agent.delayed_hangup(hangup_delay))

        if isinstance(ev.metrics, TTSMetrics):
            audio_duration = ev.metrics.audio_duration
            if audio_duration > 0:
                asyncio.create_task(agent.delayed_timeout_start(audio_duration))
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    async def write_transcript():
        try:
            # Use the same timestamp as recording
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

    agent = AutomotiveBookingAssistant(session, ctx, lang_switch_q, instructions=MY_PROMPT, lang=DEFAULT_LANG)
    agent._ctx = ctx

    await start_recording(ctx, agent)

    phone_number = extract_phone_from_room_name(ctx.room.name)
    if phone_number:
        logger.info(f"Attempting to find customer with phone: {phone_number}")
        agent._sip_participant_identity = f'sip_{phone_number}'

        found = await agent.findCustomer(phone_number)
        if found:
            agent.set_current_state("get service")
        else:
            logger.info("Customer not found, will proceed with normal flow")
    
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False  # Don't close immediately after transfer
        ),
    )

    await agent.start_background(ctx.room, "office.mp3")
    await session.generate_reply(
        instructions=MY_PROMPT
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))