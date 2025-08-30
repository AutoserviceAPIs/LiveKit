from .app_common import build_session, prewarm, extract_phone_from_room_name, start_recording, VOICE_BY_LANG
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

DEFAULT_LANG = "en-US"   
LLM_MODEL    = "gpt-4o-mini"

logger = logging.getLogger("agent_en")
load_dotenv(".env")

#- If caller requests another language (Spanish, French, Hindi), call tool set_language with the requested language and continue in that language
EN_PROMPT    = """You are an English receptionist for Woodbine Toyota. Help customers book appointments.

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


async def entrypoint(ctx: JobContext):
    lang_switch_q = asyncio.Queue()
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        llm=openai.LLM(model=LLM_MODEL),
        stt=deepgram.STT(model="nova-3", language=DEFAULT_LANG, interim_results=True),
        tts=elevenlabs.TTS(
            voice_id=VOICE_BY_LANG[DEFAULT_LANG], #sapphire
            #voice_id="xcK84VTjd6MHGJo2JVfS",#cloned
            model="eleven_flash_v2_5",
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

    await start_recording(ctx, agent)

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
        instructions=EN_PROMPT
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))