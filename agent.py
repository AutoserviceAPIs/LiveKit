#I set preemptive_generation=False,

import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    UserStateChangedEvent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    get_job_context,
)
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

logger = logging.getLogger("agent")
load_dotenv(".env")

# Extract phone number from room name
def extract_phone_from_room_name(room_name: str) -> str:
    """Extract phone number from room name format: call-_8055057710_uHsvtynDWWJN"""
    import re
    pattern = r'call-_(\d+)_'
    match = re.search(pattern, room_name)
    return match.group(1) if match else ""

# Add hangup_call function according to LiveKit documentation
async def hangup_call():
    """
    Disconnect the PSTN/SIP caller. Prefer removing the SIP participant;
    fall back to deleting the room (which disconnects everyone).
    """
    ctx = get_job_context()
    if not ctx:
        logger.warning("hangup_call(): no job context")
        return

    lk = ctx.api            # LiveKitAPI client provided by the job
    room = ctx.room

    # Log participants to make sure we target the SIP leg
    for p in room.remote_participants.values():
        logger.info("Remote participant: %s attrs=%s", p.identity, getattr(p, "attributes", {}))

    removed_any = False
    for p in list(room.remote_participants.values()):
        attrs = getattr(p, "attributes", {}) or {}
        if any(k.startswith("sip.") for k in attrs.keys()):
            try:
                await lk.room.remove_participant(
                    room_msgs.RoomParticipantIdentity(room=room.name, identity=p.identity)  # ✅ correct type
                )
                logger.info("Removed SIP participant: %s", p.identity)
                removed_any = True
            except Exception as e:
                logger.warning("remove_participant failed for %s: %s", p.identity, e)

    if not removed_any:
        try:
            await lk.room.delete_room(  # ✅ this disconnects all participants
                room_msgs.DeleteRoomRequest(room=room.name)
            )
            logger.info("Room deleted via delete_room()")
        except Exception as e:
            logger.warning("delete_room failed: %s", e)
            

class AutomotiveBookingAssistant(Agent):
    TIMEOUT_SECONDS = 10
    MAX_TIMEOUTS    = 3         # on the 3rd timeout -> transfer

    def __init__(self, session, ctx) -> None:
        super().__init__(
           instructions="""You are a receptionist for Woodbine Toyota. Help customers book appointments.

## CUSTOMER LOOKUP:
- At the beginning of the conversation, call lookup_customer function first (We already have customer phone number).
- lookup_customer returns customer name, car details, or booking details.

## RULES:
- After collecting car year make and model: trigger save_customer_information tool
- After collecting services and transportation: trigger save_services_detail tool
- After booking: trigger create_appointment
- Do not say things like "Let me save your information" or "Please wait." Just proceed silently to next step
- For recall, reschedule appointment or cancel appointment: trigger transfer_call tool
- For speak with someone, customer service, or user is frustrated: trigger transfer_call tool

- For address: 80 Queens Plate Dr, Etobicoke
- For price: oil change starts at $130 plus tax
- For Wait time: 45 minutes to 1 hour
- If ask are you a human or real person: "I am actually a voice AI assistant to help you with your service appointment". Repeat last question

## Follow this conversation flow:

Step 1. Gather First and Last Name
- If customer name and car details found: Hello {first_name}! welcome back to Woodbine Toyota. My name is Sara. I see you are calling to schedule an appointment. What service would you like for your {year} {model}?. Proceed to Step 3
- If car details not found: Hello {first_name}! welcome back to Woodbine Toyota. My name is Sara. I see you are calling to schedule an appointment. What is your car's year, make, and model?. Proceed to Step 2
- If customer name not found: Hello! You reached Woodbine Toyota Service. My name is Sara. I'll be glad to help with your appointment. Who do I have the pleasure of speaking with?

Step 2. Gather vehicle year make and model
- If first name or last name not captured: What is the spelling of your first name / last name?
- Once both first name and last name captured, Ask for the vehicle's year make and model? for example, 2025 Toyota Camry
- call save_customer_information tool

Step 3. Gather services
- Ask what services are needed for the vehicle, for example oil change, diagnostics, repairs
- Wait for services
  - If services has oil change, thank user and ask if user needs a cabin air filter replacement or a tire rotation
  - If services has maintenance, first service or general service: 
      thank user and ask if user is interested in adding wiper blades during the appointment
      Set is_maintenance to 1
- Confirm services

Step 4. Gather transportation
- After capture services, Ask if will be dropping off the vehicle or waiting while we do the work
- Wait for transportation
- After services and transportation captured, call save_services_detail tool
- Must go to Step 5 before Step 6

Step 5. Gather mileage
- Once transportation captured, Ask what is the mileage
    - Wait for response
        - If do not know, proceed to Step 6
- After transportation captured, call check_available_slots tool

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
        Trigger tool name create_appointment.
        After triggering create_appointment, say goodbye and the call will automatically end.""",
    )

        # Globals
        # Add timeout handler for user silence
        self._session_ref              = session      # keep session for say(), generate_reply(), etc
        self._ctx                      = ctx          # keep context for shutdown, logging, etc
        self._current_state            = getattr(self, "_current_state", None) or "get name"
        self._call_already_transferred = False        #Prevents multiple transfers
        self._num_timeouts             = 0            # Num of back-to-back timesouts
        self._timeout_task             = None
        self._timeout_gen              = 0            # increases every (re)start/cancel
        self._num_timeouts             = 0
        self._loop                     = None         # set on first async call
        self._sip_participant_identity = None         # Store SIP participant identity for transfer
        # Keep references so we can stop later
        self._background_state = {
            "process": None,
            "task": None,
            "track": None,
            "source": None,
        }

        # In-memory storage for customer data and appointments
        self.customer_data = {
            "first_name": "",
            "last_name": "",
            "car_make": "",
            "car_model": "",
            "car_year": "",
            "phone": "",
            "services": [],
            "transportation": None,
            "mileage": None,
            "selected_date": None,
            "selected_time": None,
            "services_transcript": "",
            "is_maintenance": 0
        }
        
        # Service mapping
        self.service_mapping = {
            "oil change": "2OILCHANGE",
            "maintenance": "1MAINTENANCE",
            "tires": "3TIRES",
            "flat": "4FLAT",
            "balance": "5BALANCE",
            "rotation": "6ROTATION",
            "alignment": "7ALIGNMENT",
            "diagnostics": "8DIAG",
            "battery": "9BATTERY",
            "ac": "10AC",
            "repair": "11REPAIR",
            "brakes": "12BRAKES",
            "key": "13KEY",
            "electrical": "14ELECTRICAL",
            "inspection": "15INSPECTION",
            "air filter": "16AIRFILTER",
            "glass": "17GLASS",
            "accessory": "18ACCESSORY",
            "software": "19SOFTWARE",
            "body shop": "20BODYSHOP",
            "car wash": "21CARWASH",
            "wipers": "22WIPERS",
            "differential": "23DIFFERENTIAL",
            "toyota care": "24TOYOTACARE",
            "recall": "RECALL"
        }
        
        # Transportation mapping
        self.transportation_mapping = {
            "DROP_OFF": "00000000-0000-0000-0000-000000000000",
            "WAITER": "11111111-1111-1111-1111-111111111111"
        }
        
        # Generate available time slots for next 2 weeks
        self.available_slots = self._generate_available_slots()
        
        # API URLs
        self.CHECK_URL = "https://api.example.com/check"  # Replace with your actual API URL
        self.CARS_URL = "https://fvpww7a95k.execute-api.us-east-1.amazonaws.com/infor/get"    # Replace with your actual API URL
        
        # Flag to track if appointment was created
        self.appointment_created = False


    def set_current_state(self, status):
        self._current_state = status
        logger.info(f"set_current_state - current_action: {status}")

    
    async def _ensure_loop(self):
        if not self._loop or not self._loop.is_running():
            self._loop = asyncio.get_running_loop()


    async def _get_lk(self) -> api.LiveKitAPI:
        if self._lk is None:
            # Reads LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET from env
            self._lk = api.LiveKitAPI(session=http_session())
        return self._lk

    
    async def delayed_timeout_start(self, audio_duration):
        """Start timeout after audio finishes playing"""
        await asyncio.sleep(audio_duration + 0.5)
        self.start_timeout()   # ✅ use self.start_timeout()
        logger.info(f"Timeout started after {audio_duration}s audio finished")

    
    def start_timeout(self):
        """Start a new timeout task"""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_gen += 1 
        gen = self._timeout_gen
        self._timeout_task = asyncio.create_task(self.timeout_handler(gen))
        logger.info(f"---Timeout started for {self.TIMEOUT_SECONDS}s (gen={gen})")

    
    async def delayed_hangup(self, delay: float):
        await asyncio.sleep(max(0.0, delay))
        logger.info("Hanging up call after appointment creation")
        try:
            await hangup_call()
            logger.info("SIP call hung up (participant removed / room deleted)")
        finally:
            logger.info("Shutting down agent")
            self._ctx.shutdown(reason="Appointment created successfully")        

    def cancel_timeout(self):
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None
        self._timeout_gen += 1       # invalidate any in-flight handler
        self._num_timeouts = 0
        logger.info("---Timeout canceled; counters reset")


    async def timeout_handler(self, gen):
        """Wait; if still idle, issue a state-aware reprompt or escalate."""
        # 1) actually wait
        await asyncio.sleep(self.TIMEOUT_SECONDS)

        # 2) was this timer canceled/restarted meanwhile?
        if gen != self._timeout_gen:
            return

        # 3) count this *real* timeout
        self._num_timeouts += 1
        logger.info(f"timeout_handler: count={self._num_timeouts}, state={self._current_state}")

        # 4) escalate on 3rd silence
        if self._num_timeouts >= self.MAX_TIMEOUTS:
            logger.info("Timeout escalation → transfer_to_number()")
            await self._session_ref.say("Let me connect you to a person.")
            await self.transfer_to_number("+16506905516")
            return

        # 5) state-aware reprompts (1st/2nd timeout)
        first_name = (self.customer_data or {}).get("first_name") or "there"
            
        if self._current_state == "get name":
            # 1st timeout
            reprompt = "Whenever you are ready, please tell me your first and last name."
            self._current_state = "get ymm"            # advance after reprompt
            await self._session_ref.say(reprompt)

        elif self._current_state == "get ymm":
            reprompt = f"Just checking in {first_name}. Could you tell me your car’s year, make, and model?"
            self._current_state = "get service"
            await self._session_ref.say(reprompt)

        elif self._current_state == "get service":
            reprompt = f"Are you still there {first_name}? Would you like an oil change?"
            self._current_state = "get transportation"
            await self._session_ref.say(reprompt)

        elif self._current_state == "get transportation":
            reprompt = (f"Are you still there {first_name}? Would you like to drop off your vehicle "
                        "or wait at the dealership?")
            self._current_state = "first availability"
            await self._session_ref.say(reprompt)

        elif self._current_state == "first availability":
            # For availability, you wanted LLM to “repeat availability”
            reprompt = "Ask if user is still there. Repeat availability."
            self._current_state = "check availability"
            await self._session_ref.generate_reply(instructions=reprompt)

        elif self._current_state == "check availability":
            reprompt = "Ask if user is still there. Repeat availability."
            # stay in "check availability"
            await self._session_ref.generate_reply(instructions=reprompt)

        else:
            # generic fallback
            await self._session_ref.generate_reply(instructions="Repeat please")

        # 6) arm the next timeout window (for next escalation)
        self.start_timeout()

    
    def _generate_available_slots(self) -> Dict[str, List[str]]:
        """Generate available time slots for the next 2 weeks"""
        slots = {}
        start_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
        for i in range(14):  # Next 2 weeks
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            day_name = current_date.strftime("%A")
            
            # Skip Sundays
            if day_name == "Sunday":
                continue
                
            # Saturday: 9 AM - 2 PM
            if day_name == "Saturday":
                slots[date_str] = [
                    "09:00", "10:00", "11:00", "12:00", "13:00"
                ]
            else:
                # Weekdays: 9 AM - 6 PM
                slots[date_str] = [
                    "09:00", "10:00", "11:00", "12:00", "13:00", 
                    "14:00", "15:00", "16:00", "17:00"
                ]
        
        return slots


    async def start_background(self, room: rtc.Room, file_path: str):
        # ffmpeg decode mp3 -> PCM 16-bit, 48kHz, mono
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", file_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            "pipe:1",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        logger.info("Start background sound")

        source = rtc.AudioSource(48000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("background", source)
        await room.local_participant.publish_track(track)

        async def read_audio():
            frame_size = 960  # 20ms @ 48kHz
            bytes_per_sample = 2
            chunk_size = frame_size * bytes_per_sample
            try:
                while True:
                    data = await process.stdout.read(chunk_size)
                    if not data:
                        break
                    frame = rtc.AudioFrame(
                        data=data,
                        sample_rate=48000,
                        num_channels=1,
                        samples_per_channel=frame_size,
                    )
                    await source.capture_frame(frame)
                    await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                # graceful exit if we cancel the task
                pass

        task = asyncio.create_task(read_audio())

        # store handles so we can stop later (e.g., in transfer_call)
        self._background_state.update(
            {"process": process, "task": task, "track": track, "source": source}
        )


    async def stop_background(self, room: rtc.Room):
        state = self._background_state
        if not state:
            return

        task = state.get("task")
        process = state.get("process")
        track = state.get("track")

        # stop reading frames
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # stop/cleanup track
        if track:
            try:
                await room.local_participant.unpublish_track(track)
            except Exception:
                pass
            try:
                track.stop()
            except Exception:
                pass

        # terminate ffmpeg
        if process and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                process.kill()

        self._background_state.clear()

    
    async def transfer_to_number(self):
        """Transfer the call to another number using LiveKit SIP transfer.
        
        Hard coded transfer number: +15105550123
        """
        if self._call_already_transferred:
            logger.info("Call already transferred, skipping")
            return
        
        try:
            # Use stored SIP participant identity
            if not self._sip_participant_identity:
                logger.error("SIP participant identity not found")
                return
            
            logger.info(f"Transferring participant: {self._sip_participant_identity}")
            livekit_url = os.getenv('LIVEKIT_URL')
            api_key = os.getenv('LIVEKIT_API_KEY')
            api_secret = os.getenv('LIVEKIT_API_SECRET')
            print("----------------")
            print(livekit_url)
            print(api_key)
            print(api_secret)

            async with api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret
            ) as livekit_api:
                transfer_to = '<sip:16616811200@x.autoserviceai.voximplant.com;user=phone>'
                job_ctx = get_job_context()
                room_name = job_ctx.room.name
                # Create transfer request
                transfer_request = TransferSIPParticipantRequest(
                    participant_identity=self._sip_participant_identity,
                    room_name=room_name,
                    transfer_to=transfer_to,
                    play_dialtone=True
                )
                logger.debug(f"Transfer request: {transfer_request}")

                # Transfer caller
                await livekit_api.sip.transfer_sip_participant(transfer_request)

                logger.info(f"Call transferred successfully to {transfer_to}")
                self._call_already_transferred = True
                
        except Exception as e:
            logger.error(f"Error transferring call: {e}")
            # Give the LLM context about the failure
            await self._session_ref.generate_reply(
                instructions="Inform the user that the transfer failed and offer to continue helping them."
            )



    @function_tool
    async def transfer_call(self, context: RunContext) -> str:
        """Transfer call to human agent when user requested.
        
        This function is called when the user wants to speak with a human agent.
        """
        logger.info("Transferring call to human agent")

        await self.transfer_to_number()
        
        return "I'm transferring you to a human agent. Please hold on."



    @function_tool
    async def save_customer_information(self, context: RunContext, first_name: str, last_name: str, 
                                      car_make: str, car_model: str, car_year: str) -> str:
        """Save customer and car information.
        Args:
            first_name: Customer's first name
            last_name: Customer's last name
            car_make: Car make (e.g., Toyota)
            car_model: Car model (e.g., Camry)
            car_year: Car year (e.g., 2020)
        """
        logger.info(f"Saving customer information: {first_name} {last_name}, {car_year} {car_make} {car_model}")
        
        self.customer_data["first_name"] = first_name
        self.customer_data["last_name"] = last_name
        self.customer_data["car_make"] = car_make
        self.customer_data["car_model"] = car_model
        self.customer_data["car_year"] = car_year
        self._current_state = "get service"
        logger.info(f"save_customer_information: current_action={self._current_state}")

        return "Customer information saved successfully."

    @function_tool
    async def save_services_detail(self, context: RunContext, services: List[str], 
                                 transportation: str, mileage: Optional[str] = None) -> str:
        """Save service details and preferences and get available timeslots.
        
        Args:
            services: List of services requested
            transportation: Drop off or wait (DROP_OFF or WAITER)
            mileage: Car mileage (optional)
        """
        logger.info(f"Saving services: {services}, transportation: {transportation}")
        
        self.customer_data["services"] = services
        self.customer_data["transportation"] = transportation
        self.customer_data["mileage"] = mileage or "0"
        
        # Determine if maintenance service
        maintenance_services = ["oil change", "maintenance", "tires", "flat", "balance", 
                              "rotation", "alignment", "air filter", "wipers", "differential", 
                              "toyota care"]
        
        is_maintenance = any(service.lower() in maintenance_services for service in services)
        self.customer_data["is_maintenance"] = 1 if is_maintenance else 0
        
        # Get available slots for the next few days
        available_slots = {}
        today = datetime.now().date()
        
        for i in range(7):  # Next 7 days
            check_date = today + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            
            if date_str in self.available_slots:
                available_slots[date_str] = self.available_slots[date_str]

        self._current_state = "first availability"
        logger.info(f"save_services_detail: current_action={self._current_state}")
                                     
        return f"Service details saved. Available slots for next 7 days: {json.dumps(available_slots)}"

    @function_tool
    async def check_available_slots(self, context: RunContext, preferred_date: Optional[str] = None) -> str:
        """Check available time slots for a specific date.
        
        Args:
            preferred_date: Preferred date in YYYY-MM-DD format (optional)
        """
        logger.info(f"Checking available slots for date: {preferred_date}")
        
        if preferred_date:
            if preferred_date in self.available_slots:
                slots = self.available_slots[preferred_date]
                return f"Available slots on {preferred_date}: {', '.join(slots)}"
            else:
                return f"No available slots on {preferred_date}"
        else:
            # Return next 3 available dates
            available_dates = list(self.available_slots.keys())[:3]
            result = {}
            for date in available_dates:
                result[date] = self.available_slots[date]

            self._current_state = "check availability"
            logger.info(f"check_available_slots: current_action={self._current_state}")

            return f"Next available dates: {json.dumps(result)}"

    @function_tool
    async def create_appointment(self, context: RunContext, first_name: str, last_name: str,
                               car_make: str, car_model: str, car_year: str, services: List[str],
                               transportation: str, services_transcript: str, is_maintenance: int,
                               selected_date: str, selected_time: str) -> str:
        """Create and finalize the booking appointment.
        
        Args:
            first_name: Customer's first name
            last_name: Customer's last name
            car_make: Car make
            car_model: Car model
            car_year: Car year
            services: List of services requested
            transportation: Drop off or wait
            services_transcript: Transcript of the request for services
            is_maintenance: Whether this is a maintenance service (0 or 1)
            selected_date: Selected appointment date in YYYY-MM-DD format
            selected_time: Selected appointment time in HH:mm:ss format
        """
        logger.info(f"Creating appointment for {first_name} {last_name} on {selected_date} at {selected_time}")
        

        # Create appointment ID
        appointment_id = f"APT_{selected_date}_{selected_time}_{first_name}_{last_name}"
        
        # Remove the slot from available slots
        # self.available_slots[selected_date].remove(selected_time)
        
        # Store appointment details
        appointment_data = {
            "appointment_id": appointment_id,
            "customer_name": f"{first_name} {last_name}",
            "car_info": f"{car_year} {car_make} {car_model}",
            "services": services,
            "services_transcript": services_transcript,
            "transportation": transportation,
            "is_maintenance": is_maintenance,
            "date": selected_date,
            "time": selected_time,
            "status": "confirmed",
            "created_at": datetime.now().isoformat()
        }
        
        # In a real implementation, you would save this to a database
        logger.info(f"Appointment created: {json.dumps(appointment_data)}")
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        return f"Appointment confirmed! Your appointment ID is {appointment_id}. You have service scheduled for {selected_date} at {selected_time}. We'll send a confirmation message shortly. Have a great day!"

    @function_tool
    async def lookup_customer(self, context: RunContext, phone: str) -> str:
        """Look up customer information by phone number.
        
        Args:
            phone: Customer's phone number
        """
        logger.info(f"Looking up customer in")
        
        # Check if customer data is already populated
        if (self.customer_data["first_name"] and 
            self.customer_data["last_name"] and 
            self.customer_data["car_make"] and 
            self.customer_data["car_model"] and 
            self.customer_data["car_year"]):
            
            # Customer found, create response from existing customer_data
            result = {
                "success": True,
                "firstName": self.customer_data["first_name"],
                "lastName": self.customer_data["last_name"],
                "make": self.customer_data["car_make"],
                "model": self.customer_data["car_model"],
                "year": self.customer_data["car_year"],
                "message": f"Found customer record with name: {self.customer_data['first_name']} {self.customer_data['last_name']} with car {self.customer_data['car_year']} {self.customer_data['car_make']} {self.customer_data['car_model']}"                
            }
            # Store the lookup result for later use
            return json.dumps(result)
        else:
            result = {
                "success": False,
                "message": "Customer not found in our system. Please ask customer information."
            }
            return json.dumps(result)  

    async def findCustomer(self, phone: str) -> bool:
        """Find customer by phone number and populate customer_data.
        
        This function works like the lookupCustomer in BOOKING_ASSISTANT.md
        - Calls API to find customer by phone number
        - Populates customer_data if found
        - Returns True if found, False otherwise
        
        Args:
            phone: Customer's phone number
        """
        logger.info(f"---FindCustomer with phone: {phone}")
        try:
            # Get last 10 digits of phone number
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                phone_10_digits = clean_phone[-10:]
            else:
                logger.error(f"Invalid phone number: {phone}")
                return False
            
            # Call API to find customer
            url = self.CARS_URL
            params = {"phone": phone_10_digits}
            logger.info(f"Calling API: {url} with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        # Get response text first
                        response_text = await response.text()
                        logger.info(f"API Response: {response_text}")
                        
                        try:
                            # Try to parse as JSON
                            result = json.loads(response_text)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON response: {response_text}")
                            return False
                        
                        # Check if customer found
                        if (result and not result.get("error") and 
                            result.get("Found") and 
                            result.get("Cars") and 
                            len(result["Cars"]) > 0):
                            
                            # Get first car data
                            car_data = result["Cars"][0]
                            
                            # Populate customer_data
                            self.customer_data["first_name"] = car_data.get("fname", "")
                            self.customer_data["last_name"] = car_data.get("lname", "")
                            self.customer_data["car_model"] = car_data.get("model", "")
                            self.customer_data["car_make"] = car_data.get("make", "")
                            self.customer_data["car_year"] = car_data.get("year", "")
                            self.customer_data["phone"] = phone_10_digits
                            logger.info(f"Customer found: {self.customer_data['first_name']} {self.customer_data['last_name']} with {self.customer_data['car_year']} {self.customer_data['car_make']} {self.customer_data['car_model']}")
                            return True
                        else:
                            logger.info(f"Customer not found for phone: {phone_10_digits}")
                            return False
                    else:
                        logger.error(f"API call failed with status {response.status}")
                        return False
                        
        except Exception as error:
            logger.error(f"Customer lookup error: {error}")
            return False
            

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
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

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="en-US", api_key="98e42a6de0d4660afec26ebcbda8499f42bd4b5d"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=elevenlabs.TTS(
            voice_id="zmcVlqmyk3Jpn5AVYcAL", #sapphire
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
    agent = AutomotiveBookingAssistant(session, ctx)
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
    # cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="my-telephony-agent"))
