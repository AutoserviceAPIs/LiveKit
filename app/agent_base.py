#TO BE CHANGED
#   findCustomer: If find appointment
#   reschedule_appointment
#   cancel_appointment

# your AutomotiveBookingAssistant base class/logic
from .agent_customerdata import CARS_URL
from .agent_common import CONFIRM_BY_LANG, LANG_MAP, normalize_lang
from .agent_supervisor import LanguageSupervisor
from livekit import agents, rtc, api, agents
from livekit.agents.metrics import TTSMetrics
from livekit.protocol import room as room_msgs  # ✅ protocol types live here
from livekit.protocol.sip import TransferSIPParticipantRequest
from livekit.plugins import elevenlabs, deepgram, noise_cancellation, openai, silero
from livekit.plugins.elevenlabs import VoiceSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.voice import Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.agents import (
    NOT_GIVEN, Agent, UserStateChangedEvent, AgentFalseInterruptionEvent,
    AgentSession, JobContext, JobProcess, MetricsCollectedEvent,
    RoomInputOptions, WorkerOptions, cli, metrics, get_job_context,)
import logging, aiohttp, json, os, asyncio, time, subprocess, re, wave
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

log = logging.getLogger("agent")
load_dotenv(".env")


class AutomotiveBookingAssistant(Agent):
    TIMEOUT_SECONDS = 10
    MAX_TIMEOUTS    = 3         # on the 3rd timeout -> transfer

    def __init__(self, session, ctx, lang_switch_q=None, *, instructions: str = "", lang: str = "en", supervisor: Optional[object] = None) -> None:
        super().__init__(instructions=instructions)

        # Globals
        # Add timeout handler for user silence
        self._transfer_to              = '<sip:15129007000@x.autoserviceai.voximplant.com;user=phone>'        
        self._session                  = session      # keep session for say(), generate_reply(), etc
        self._session_ref              = session      # keep session for say(), generate_reply(), etc
        self._ctx                      = ctx          # keep context for shutdown, logging, etc
        self._current_state            = getattr(self, "_current_state", None) or "get name"
        self._call_already_transferred = False        #Prevents multiple transfers
        self._num_timeouts             = 0            # Num of back-to-back timesouts
        self._timeout_task             = None
        self._timeout_gen              = 0            # increases every (re)start/cancel
        self._loop                     = None         # set on first async call
        self._sip_participant_identity = None         # Store SIP participant identity for transfer
        self._lang_switch_q            = lang_switch_q
        self._recording_timestamp	   = None
        self._lang                     = lang
        self._is_shutting_down         = False
        self._timeouts                 = 0
        self.supervisor                = supervisor or lang_switch_q
        self._suppress_responses       = False
        self._background_state         = {"task": None, "source": None}
        self._found_appt_id            = 0            #If appointment is found
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
            "is_maintenance": 0,
            "appointment_id": 0
        }        
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
        
        # Flag to track if appointment was created
        self.appointment_created = False


    def set_current_state(self, status):
        self._current_state = status
        log.info(f"set_current_state - current_action: {status}")

    
    def _safe_get(self, d, key):
        return d.get(key) if isinstance(d, dict) else None


    def _safe_copy(self, v):
        # avoid sharing mutable references across processes
        if isinstance(v, dict):
            return {k: self._safe_copy(v[k]) for k in v}
        if isinstance(v, list):
            return [self._safe_copy(x) for x in v]
        return v


    def snapshot_state(self) -> dict:
        """Return a small, serializable snapshot for language handoff."""
        s = getattr(self, "customer_data", {}) or {}
        snap = {
            "phone":                self._safe_copy(self._safe_get(s, "phone")),
            "first_name":           self._safe_copy(self._safe_get(s, "first_name")),
            "last_name":            self._safe_copy(self._safe_get(s, "last_name")),
            "car_year":             self._safe_copy(self._safe_get(s, "car_year")),
            "car_make":             self._safe_copy(self._safe_get(s, "car_make")),
            "car_model":            self._safe_copy(self._safe_get(s, "car_model")),
            "services":             self._safe_copy(self._safe_get(s, "services") or []),
            "transportation":       self._safe_copy(self._safe_get(s, "transportation")),
            "mileage":              self._safe_copy(self._safe_get(s, "mileage")),
            "selected_date":        self._safe_copy(self._safe_get(s, "selected_date")),
            "selected_time":        self._safe_copy(self._safe_get(s, "selected_time")),
            "services_transcript":  self._safe_copy(self._safe_get(s, "services_transcript")),
            "is_maintenance":       self._safe_copy(self._safe_get(s, "is_maintenance")),
            "appointment_id":       self._safe_copy(self._safe_get(s, "appointment_id")),
            "appointment_date":     self._safe_copy(self._safe_get(s, "appointment_date")),
            "appointment_time":     self._safe_copy(self._safe_get(s, "appointment_time")),
        }
        return snap


    def restore_state(self, snapshot: dict | None):
        """Merge a snapshot back into self.customer_data (used by ES/FR on startup)."""
        if not isinstance(snapshot, dict):
            return
        for k, v in snapshot.items():
            self.customer_data[k] = self._safe_copy(v)


    async def say(self, text: str):
        """Speak a short line via whatever the session supports."""
        if getattr(self, "_is_shutting_down", False):
            if hasattr(self.session, "history"):
                try:
                    self.session.history.add_assistant_message(text)  # or your equivalent
                except Exception:
                    pass
            return

        # 1) OpenAI Realtime-style helper (preferred if available)
        if hasattr(self.session, "response") and hasattr(self.session.response, "create"):
            return await self.session.response.create({"instructions": text})

        # 2) Generic event sender your wrapper might expose
        if hasattr(self.session, "send_event"):
            return await self.session.send_event({
                "type": "response.create",
                "response": {"instructions": text}
            })

        # 3) Direct websocket JSON (if your session has a .ws client)
        if hasattr(self.session, "ws") and hasattr(self.session.ws, "send_json"):
            return await self.session.ws.send_json({
                "type": "response.create",
                "response": {"instructions": text}
            })

        # 4) Your own convenience methods (many wrappers expose these)
        if hasattr(self.session, "say") and callable(self.session.say):
            return await self.session.say(text)

        if hasattr(self.session, "generate_reply") and callable(self.session.generate_reply):
            if getattr(self, "_suppress_responses", False):
                return
            return await self.session.generate_reply(text)

        # 5) Last resort: don’t crash
        print(f"[AI SAY fallback] {text}")
        return None



    async def shutdown(self):
        log.info(f"shutdown")
        self._is_shutting_down = True
        await self.bg_stop()                 # if you have bg audio
        if getattr(self, "_timeout_task", None):
            try:
                self._timeout_task.cancel()
            except Exception:
                pass
            self._timeout_task = None
        # detach listeners / close session if applicable


    def is_shutting_down(self) -> bool:
        log.info(f"is_shutting_down")
        return self._is_shutting_down


    async def on_user_message(self, text: str):
        if self._is_shutting_down:
            return


    async def on_tool_call(self, name, args):
        if self._is_shutting_down and name != "set_language":
            return


    async def _ensure_loop(self):
        if not self._loop or not self._loop.is_running():
            self._loop = asyncio.get_running_loop()


    async def _start_timeout(self):
        if self._is_shutting_down:
            return


    async def _timeout_handler(self):
        if self._is_shutting_down:
            return


    async def _get_lk(self) -> api.LiveKitAPI:
        """Return the LiveKitAPI from current job context."""
        ctx = get_job_context()
        return ctx.api if ctx else None

    
    async def delayed_timeout_start(self, audio_duration):
        """Start timeout after audio finishes playing"""
        await asyncio.sleep(audio_duration + 0.5)
        self.start_timeout()   # ✅ use self.start_timeout()
        log.info(f"Timeout started after {audio_duration}s audio finished")
    
    def start_timeout(self):
        """Start (or restart) the silence timeout."""
        if getattr(self, "_is_shutting_down", False):
            return
        # cancel existing
        self.cancel_timeout()
        self._timeout_gen = getattr(self, "_timeout_gen", 0) + 1
        gen = self._timeout_gen
        self._timeout_task = asyncio.create_task(self._timeout_loop(gen))
        log.info(f"---Timeout started for {self.TIMEOUT_SECONDS}s (gen={gen})")
    
    async def delayed_hangup(self, delay: float):
        await asyncio.sleep(max(0.0, delay))
        log.info("Hanging up call after appointment creation")
        try:
            await self.hangup_call()
            log.info("SIP call hung up (participant removed / room deleted)")
        finally:
            log.info("Shutting down agent")
            self._ctx.shutdown(reason="Appointment created successfully")        

    def cancel_timeout(self):
        """Stop any pending silence timeout."""
        t = getattr(self, "_timeout_task", None)
        if t and not t.done():
            t.cancel()
        self._timeout_task = None
        self._num_timeouts = 0
        self._timeout_gen += 1       # invalidate any in-flight handler
        log.info("---Timeout canceled; counters reset")

    async def _timeout_loop(self, gen: int):
        try:
            await asyncio.sleep(self.TIMEOUT_SECONDS)
            if getattr(self, "_is_shutting_down", False) or gen != self._timeout_gen:
                return
            self._num_timeouts = getattr(self, "_num_timeouts", 0) + 1
            await self.say("Are you still there?")
            if self._num_timeouts >= self.MAX_TIMEOUTS:
                await self.transfer_call()
            else:
                # arm next timeout
                self.start_timeout()
        except asyncio.CancelledError:
            pass

    async def timeout_handler(self, gen):
        """Wait; if still idle, issue a state-aware reprompt or escalate."""
        # 1) actually wait
        await asyncio.sleep(self.TIMEOUT_SECONDS)

        # 2) was this timer canceled/restarted meanwhile?
        if gen != self._timeout_gen:
            return

        # 3) count this *real* timeout
        self._num_timeouts += 1
        log.info(f"timeout_handler: count={self._num_timeouts}, state={self._current_state}")

        # 4) escalate on 3rd silence
        if self._num_timeouts == (self.MAX_TIMEOUTS - 1):
            log.info("Timeout skip cycle")
            return     
            
        if self._num_timeouts >= self.MAX_TIMEOUTS:
            log.info("Timeout escalation → transfer_to_number()")
            await self._session_ref.say("Let me connect you to a person.")
            await self.transfer_to_number()
            return

        # 5) state-aware reprompts (1st/2nd timeout)
        first_name = (self.customer_data or {}).get("first_name") or "there"
            
        if self._current_state == "get name":
            # 1st timeout
            reprompt = "take your time, let me know when u are ready"
            #reprompt = "Whenever you are ready, please tell me your first and last name."
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
            reprompt = (f"Are you still there {first_name}? Would you like to drop off your vehicle"
                        "or wait at the dealership?")
            self._current_state = "first availability"
            await self._session_ref.say(reprompt)

        elif self._current_state == "first availability":
            # For availability, you wanted LLM to “repeat availability”
            reprompt = "Ask if user is still there. Repeat availability"
            self._current_state = "check availability"
            await self._session_ref.generate_reply(instructions=reprompt)

        elif self._current_state == "check availability":
            reprompt = "Ask if user is still there. Repeat availability"
            # stay in "check availability"
            await self._session_ref.generate_reply(instructions=reprompt)

        elif self._current_state == "cancel reschedule":
            reprompt = "Ask if user would like to reschedule"
            # stay in "check availability"
            await self._session_ref.generate_reply(instructions=reprompt)

        else:
            # generic fallback
            await self._session_ref.generate_reply(instructions="Repeat please")

        # 6) arm the next timeout window (for next escalation)
        self.start_timeout()
        log.info(f"timeout_handler: state= {self._current_state} reprompt= {reprompt}")

    
    def _generate_available_slots(self) -> Dict[str, List[str]]:
        """Generate available time slots for the next 2 weeks"""
        log.info("_generate_available_slots for next 2 weeks")
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


    # --- REPLACE your start_background with this ---
    async def start_background(self, room: rtc.Room, file_path: str):
        # ffmpeg: decode to PCM 16-bit, 48 kHz, mono
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", file_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", "48000", "-ac", "1",
            "pipe:1",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        log.info("Start background sound")

        source = rtc.AudioSource(48000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("background", source)

        # Keep the publication SID so we can unpublish cleanly later
        publication = await room.local_participant.publish_track(track)
        pub_sid = getattr(publication, "sid", None)

        async def read_audio():
            frame_size = 960      # 20 ms @ 48kHz
            bytes_per_sample = 2
            chunk_size = frame_size * bytes_per_sample
            buffer = b""
            try:
                while True:
                    chunk = await process.stdout.read(1024)
                    if not chunk:
                        break
                    buffer += chunk
                    while len(buffer) >= chunk_size:
                        frame_data = buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        frame = rtc.AudioFrame(
                            data=frame_data,
                            sample_rate=48000,
                            num_channels=1,
                            samples_per_channel=frame_size,
                        )
                        await source.capture_frame(frame)
                        await asyncio.sleep(0.02)  # pacing: 20 ms
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.error(f"Error in read_audio: {e}")
                raise

        task = asyncio.create_task(read_audio())

        # store handles so we can stop later
        self._background_state = {
            "process": process,
            "task": task,
            "track": track,
            "source": source,
            "room": room,
            "publication_sid": pub_sid,
        }


    async def stop_background(self):
        """Stop background audio playback and cleanup resources."""
        bg = getattr(self, "_background_state", None) or {}
        task  = bg.get("task")
        track = bg.get("track")
        source = bg.get("source")
        room  = bg.get("room")
        pub_sid = bg.get("publication_sid")
        proc = bg.get("process")

        # 1) Stop feeding audio
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        # 2) Unpublish the local track (this actually stops others from hearing it)
        try:
            if room and pub_sid:
                await room.local_participant.unpublish_track(pub_sid)
            # Fallback: some SDKs accept a track object or track.sid
            elif room and hasattr(track, "sid"):
                await room.local_participant.unpublish_track(track.sid)
        except Exception as e:
            log.warning(f"unpublish_track failed: {e}")

        # 3) Close the audio source if supported
        try:
            if source and hasattr(source, "close"):
                source.close()
        except Exception:
            pass

        # 4) Terminate ffmpeg process
        if proc:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    proc.kill()
            except Exception:
                pass

        log.info("Background audio stopped and cleaned up")
        self._background_state = {}

    
    # Add hangup_call function according to LiveKit documentation
    async def hangup_call(self):
        """
        Disconnect the PSTN/SIP caller. Prefer removing the SIP participant;
        fall back to deleting the room (which disconnects everyone).
        """
        # Stop background audio if playing
        await self.stop_background()
        
        self.cancel_timeout()
        ctx = get_job_context()
        if not ctx:
            log.warning("hangup_call(): no job context")
            return

        lk = ctx.api            # LiveKitAPI client provided by the job
        room = ctx.room

        # Log participants to make sure we target the SIP leg
        for p in room.remote_participants.values():
            log.info("Remote participant: %s attrs=%s", p.identity, getattr(p, "attributes", {}))

        removed_any = False
        for p in list(room.remote_participants.values()):
            attrs = getattr(p, "attributes", {}) or {}
            if any(k.startswith("sip.") for k in attrs.keys()):
                try:
                    await lk.room.remove_participant(
                        room_msgs.RoomParticipantIdentity(room=room.name, identity=p.identity)  # ✅ correct type
                    )
                    log.info("Removed SIP participant: %s", p.identity)
                    removed_any = True
                except Exception as e:
                    log.warning("remove_participant failed for %s: %s", p.identity, e)

        if not removed_any:
            try:
                await lk.room.delete_room(  # ✅ this disconnects all participants
                    room_msgs.DeleteRoomRequest(room=room.name)
                )
                log.info("Room deleted via delete_room()")
            except Exception as e:
                log.warning("delete_room failed: %s", e)


    async def transfer_to_number(self):
        """Transfer the call to another number using LiveKit SIP transfer.
        
        Hard coded transfer number: +15105550123
        """
        if self._call_already_transferred:
            log.info("Call already transferred, skipping")
            return
        
        try:
            # Use stored SIP participant identity
            if not self._sip_participant_identity:
                log.error("SIP participant identity not found")
                return
            
            log.info(f"Transferring participant: {self._sip_participant_identity}")
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
                transfer_to = self._transfer_to
                job_ctx = get_job_context()
                room_name = job_ctx.room.name
                # Create transfer request
                transfer_request = TransferSIPParticipantRequest(
                    participant_identity=self._sip_participant_identity,
                    room_name=room_name,
                    transfer_to=transfer_to,
                    play_dialtone=True
                )
                log.debug(f"Transfer request: {transfer_request}")

                # Transfer caller
                await livekit_api.sip.transfer_sip_participant(transfer_request)

                log.info(f"Call transferred successfully to {transfer_to}")
                self._call_already_transferred = True
                
        except Exception as e:
            log.error(f"Error transferring call: {e}")
            # Give the LLM context about the failure
            await self._session_ref.generate_reply(
                instructions="Inform the user that the transfer failed and offer to continue helping them."
            )


    @function_tool
    async def transfer_call(self, context: RunContext) -> str:
        """Transfer call to human agent when user requested.
        
        This function is called when the user wants to speak with a human agent.
        """
        log.info("Transferring call to human agent")

        await self.transfer_to_number()
        
        return "I'm transferring you to a human agent. Please hold on."


    @function_tool
    async def set_language(self, lang: str) -> str:
        if getattr(self, "_handoff_inflight", False):
            return "BUSY"

        canon = normalize_lang(lang)  # e.g., "French" -> "fr"
        log.info(f"set_language lang = {canon}")

        SUPPORTED = {"es", "fr", "hi", "vi"}
        if canon not in SUPPORTED:
            await self.say("Sorry, that language isn’t available right now.")
            return "ERR_UNSUPPORTED_LANG"

        # No-op if already in this language (no beeps)
        if canon == getattr(self, "lang", "en"):
            return "OK"

        # 1) Speak confirmation in the user's requested language (or fallback)
        key = LANG_MAP.get(lang, lang)  # handles "French" -> "fr", etc.
        await self.say(CONFIRM_BY_LANG.get(key, CONFIRM_BY_LANG["en"]))

        self._handoff_inflight = True
        self._is_shutting_down = True
        try:
            # Cancel any timers
            try:
                self.cancel_timeout()
            except Exception:
                pass

            # Stop any background audio started by this agent
            try:
                if hasattr(self, "stop_background"):
                    await self.stop_background()
            except Exception:
                pass

            # Ensure we have a supervisor (room-scoped)
            if not getattr(self, "supervisor", None):
                room = getattr(getattr(self._ctx, "room", None), "name", None)
                if not room:
                    self._is_shutting_down = False
                    return "ERR_NO_SUPERVISOR"
                self.supervisor = LanguageSupervisor(room=room, ready_timeout=12.0)

            # Snapshot current state for the new agent
            try:
                snapshot = self.snapshot_state()
            except Exception:
                snapshot = {
                    "state": getattr(self, "state", {}),
                    "customer_data": getattr(self, "customer_data", {}),
                    "available_slots": getattr(self, "available_slots", []),
                }

            # --- Immediately mute this agent so it won't talk over the new one ---
            self._suppress_responses = True
            try:
                await self.session.update(input_audio_enabled=False)
            except Exception:
                pass

            # 2) Start beeping DURING handoff; stop when READY or on timeout
            stop_evt = asyncio.Event()

            async def _handoff_beeper():
                # Non-blocking loop: emits two short beeps every ~0.35s until stop_evt is set
                try:
                    while not stop_evt.is_set():
                        try:
                            if hasattr(self, "play_beeps"):
                                await self.play_beeps(
                                    self.session.room,
                                    count=2, freq=1000, duration=0.12, gap=0.12, volume=0.38
                                )
                        except Exception as e:
                            log.debug(f"handoff beeper suppressed: {e}")
                            await asyncio.sleep(0.3)
                        # brief spacing between beep pairs unless stopped
                        try:
                            await asyncio.wait_for(stop_evt.wait(), timeout=0.35)
                        except asyncio.TimeoutError:
                            pass
                except Exception as e:
                    log.warning(f"handoff beeper error: {e}")

            beep_task = asyncio.create_task(_handoff_beeper())

            # 3) Spawn new language agent and wait for READY
            try:
                await self.supervisor.handoff(canon, state_snapshot=snapshot)
            except TimeoutError:
                # Keep this agent muted; stop beeps; let FR/other agent greet when ready
                stop_evt.set()
                try:
                    await asyncio.wait_for(beep_task, timeout=0.3)
                except Exception:
                    beep_task.cancel()
                return "PENDING"

            # READY received → stop beeps
            stop_evt.set()
            try:
                await asyncio.wait_for(beep_task, timeout=0.3)
            except Exception:
                beep_task.cancel()

            # 4) Leave the room as the old-language participant (do not tear down the room)
            try:
                await self.session.close()
            except Exception:
                pass

            return "OK"

        except Exception as e:
            # Hard failure: allow this agent to resume (optional: you might choose to keep it muted)
            self._is_shutting_down = False
            try:
                self._suppress_responses = False
                await self.session.update(input_audio_enabled=True)
            except Exception:
                pass
            await self.say("I couldn’t switch languages right now. We can continue here or I can transfer you to a person.")
            return f"ERR_HANDOFF_FAILED:{e}"

        finally:
            self._handoff_inflight = False



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
        log.info(f"Saving customer information: {first_name} {last_name}, {car_year} {car_make} {car_model}")
        
        self.customer_data["first_name"] = first_name
        self.customer_data["last_name"] = last_name
        self.customer_data["car_make"] = car_make
        self.customer_data["car_model"] = car_model
        self.customer_data["car_year"] = car_year
        self._current_state = "get service"
        log.info(f"save_customer_information: current_action={self._current_state}")

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
        log.info(f"Saving services: {services}, transportation: {transportation}")
        
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
        log.info(f"save_services_detail: current_action={self._current_state}")
                                     
        return f"Service details saved. Available slots for next 7 days: {json.dumps(available_slots)}"


    @function_tool
    async def check_available_slots(self, context: RunContext, preferred_date: Optional[str] = None) -> str:
        """Check available time slots for a specific date.
        
        Args:
            preferred_date: Preferred date in YYYY-MM-DD format (optional)
        """
        log.info(f"Checking available slots for date: {preferred_date}")
        
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
            log.info(f"check_available_slots: current_action={self._current_state}")

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
        log.info(f"Creating appointment for {first_name} {last_name} on {selected_date} at {selected_time}")
        

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
        log.info(f"Appointment created: {json.dumps(appointment_data)}")
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        return f"Appointment confirmed! Your appointment ID is {appointment_id}. You have service scheduled for {selected_date} at {selected_time}. We'll send a confirmation message shortly. Have a great day!"


    #TO BE CHANGED
    @function_tool
    async def reschedule_appointment(self, context: RunContext, appointment_id: str, first_name: str, last_name: str,
                               car_make: str, car_model: str, car_year: str, services: List[str],
                               transportation: str, services_transcript: str, is_maintenance: int,
                               selected_date: str, selected_time: str) -> str:
        """reschedule an booking appointment.
        
        Args:
            appointment_id
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
        log.info(f"Rescheduling appointment for {first_name} {last_name} on {selected_date} at {selected_time}")
        

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
        log.info(f"Appointment created: {json.dumps(appointment_data)}")
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        # Set flag to indicate appointment was created - this will trigger hangup after goodbye
        self.appointment_created = True
        
        return f"Appointment rescheduled! Your service is rescheduled for {selected_date} at {selected_time}. We'll send a confirmation message shortly. Have a great day!"


    #TO BE CHANGED
    @function_tool
    async def cancel_appointment(self, context: RunContext, appointment_id: str) -> str:
        """cancel appointment.
        
        Args:
            appointment_id
        """
        log.info(f"cancel_appointment for {first_name} {last_name} on {selected_date} at {selected_time}")        
        return f"Appointment cancelled! We'll send a confirmation message shortly. Have a great day!"


    @function_tool(name="lookup_customer", description="Look up customer by phone number already on file.")
    async def lookup_customer(self, context: RunContext, phone: str) -> str:
        """Look up customer information by phone number.
        
        Args:
            phone: Customer's phone number
        """
        log.info(f"Looking up customer in")
        
        # Check if customer data is already populated
        if (self.customer_data.get("first_name") and 
            self.customer_data.get("last_name") and 
            self.customer_data.get("car_make") and 
            self.customer_data.get("car_model") and 
            self.customer_data.get("car_year")):

            # Derive existing appointment info from customer_data if present
            has_existing = bool(self.customer_data.get("found_appt_id") and 
                                self.customer_data.get("selected_date") and 
                                self.customer_data.get("selected_time")) or \
                           bool(self.customer_data.get("has_existing_appointment"))

            result = {
                "success": True,
                "firstName": self.customer_data.get("first_name", ""),
                "lastName": self.customer_data.get("last_name", ""),
                "make": self.customer_data.get("car_make", ""),
                "model": self.customer_data.get("car_model", ""),
                "year": self.customer_data.get("car_year", ""),
                "services": self.customer_data.get("services", []),
                "transportation": self.customer_data.get("transportation", ""),
                "hasExistingAppointment": has_existing,
                "found_appt_id": self.customer_data.get("found_appt_id", ""),
                "appt_date": self.customer_data.get("selected_date", ""),
                "appt_time": self.customer_data.get("selected_time", ""),
                "message": f"Found customer record with name: {self.customer_data.get('first_name','')} {self.customer_data.get('last_name','')} with car {self.customer_data.get('car_year','')} {self.customer_data.get('car_make','')} {self.customer_data.get('car_model','')}"
            }
            return json.dumps(result)
        else:
            result = {
                "success": False,
                "hasExistingAppointment": False,
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
        log.info(f"---FindCustomer with phone: {phone}")
        try:
            # Get last 10 digits of phone number
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                phone_10_digits = clean_phone[-10:]
            else:
                log.error(f"Invalid phone number: {phone}")
                return False
            
            # Call API to find customer
            url = CARS_URL
            params = {"phone": phone_10_digits}
            log.info(f"Calling API: {url} with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        # Get response text first
                        response_text = await response.text()
                        log.info(f"API Response: {response_text}")
                        
                        try:
                            # Try to parse as JSON
                            result = json.loads(response_text)
                        except json.JSONDecodeError:
                            log.error(f"Failed to parse JSON response: {response_text}")
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

                            # Optional additional fields from response
                            ## TODO: Remove this after testing
                            found_appt_id = "123"
                            appt_date = "2025-09-09"
                            appt_time = "10:00:00"
                            services_raw = (result.get("services") or "").strip()
                            transportation_raw = (result.get("transportation") or "").strip()

                            # Store services as list if provided
                            if services_raw:
                                try:
                                    services_list = [s.strip() for s in services_raw.split(',') if s.strip()]
                                except Exception:
                                    services_list = [services_raw]
                                self.customer_data["services"] = services_list

                            # Store transportation (keep raw and a simple mapped flag)
                            if transportation_raw != "":
                                self.customer_data["transportation"] = transportation_raw

                            # If there is an existing appointment, capture and pivot flow
                            if found_appt_id and appt_date and appt_time:
                                self.customer_data["found_appt_id"] = found_appt_id
                                self.customer_data["selected_date"] = appt_date
                                self.customer_data["selected_time"] = appt_time
                                self.customer_data["has_existing_appointment"] = True

                                # Switch conversation to reschedule/cancel flow (prompt is handled via COMMON_PROMPT)
                                self._current_state = "ask reschedule or cancel"

                            log.info(f"Customer found: {self.customer_data['first_name']} {self.customer_data['last_name']} with {self.customer_data['car_year']} {self.customer_data['car_make']} {self.customer_data['car_model']} and has existing appointment: {self.customer_data['has_existing_appointment']}")
                            return True
                        else:
                            log.info(f"Customer not found for phone: {phone_10_digits}")
                            return False
                    else:
                        log.error(f"API call failed with status {response.status}")
                        return False
                        
        except Exception as error:
            log.error(f"Customer lookup error: {error}")
            return False

    @function_tool
    async def cancel_appointment(self, context: RunContext, appointment_id: str) -> str:
        """Cancel an existing appointment."""
        try:
            payload = {"appointment_id": appointment_id}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.CANCEL_URL, json=payload) as resp:
                    self.customer_data["has_existing_appointment"] = False
                    return "Appointment cancelled successfully."
        except Exception as e:
            log.error(f"cancel_appointment error: {e}")
            return "Failed to cancel appointment."

    @function_tool
    async def reschedule_appointment(self, context: RunContext, appointment_id: str, new_date: str, new_time: str) -> str:
        """Reschedule an existing appointment to a new date and time."""
        try:
            payload = {"appointment_id": appointment_id, "date": new_date, "time": new_time}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.RESCHEDULE_URL, json=payload) as resp:
                    self.customer_data["selected_date"] = new_date
                    self.customer_data["selected_time"] = new_time
                    return "Appointment rescheduled successfully."
    
        except Exception as e:
            log.error(f"reschedule_appointment error: {e}")
            return "Failed to reschedule appointment."