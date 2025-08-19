#vad = silero.VAD.load(threshold=0.6)
#Not completed: assemblyai test
#Not needed: elevenlabs speaking rate
#session.generate_reply instructions
#super().__init__(instructions


from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
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
)
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    elevenlabs,
    deepgram,
    silero
)

load_dotenv()
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import asyncio
import subprocess
from livekit import rtc


async def publish_background(room: rtc.Room, file_path: str):
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

    # tạo audio source + track
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("background", source)
    await room.local_participant.publish_track(track)

    async def read_audio():
        frame_size = 960  # 20ms @ 48kHz
        bytes_per_sample = 2
        chunk_size = frame_size * bytes_per_sample

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

    asyncio.create_task(read_audio())

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful booking assistant.
		Capture first name and last name:
			If last name is not captured, ask again
			If first name is not a valid name, ask for spelling
			If last name is not a valid last name, ask for spelling
		Once both first name and last name captured, Ask for the vehicle's year make and model, for example, 2025 Toyota Camry:
			model needs to be a valid car model.
		Once the Year and the Make and the Model are captured, Ask for services needed for the vehicle, e.g. oil change, diagnostics, repairs:
                        If service is oil change, ask if user needs a cabin air filter replacement or a tire rotation.
                Once that is captured: Ask if user will be dropping off the vehicle or waiting while we do the work.
                Once that is captured, Ask what is the mileage.
                Once that is capture or user does not know the mileage, Offer the first availability:
			ask if that will work, or if the user has a specific time and wait for a reply:
				If the user requested date and time is available, book it and confirm the booking.
                        	Else If the user requested date and time is not available:
                                	Offer 3 timeslots and repeat till the user finds a suitable time.
                                	If user provides a period, ask for an approximate date and time.
					If a match is found, confirm the user selection for example: Just to be sure, you would like to book ...
                		When booked, inform the user they will receive an email or text confirmation shortly.
            Response English only.""",
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="en-US", api_key="98e42a6de0d4660afec26ebcbda8499f42bd4b5d"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=elevenlabs.TTS(
            voice_id="xcK84VTjd6MHGJo2JVfS",
            model="eleven_flash_v2_5",
            api_key="59bb59df13287e23ba2da37ea6e48724",
            #voice_settings= {
            #     "similarity_boost": 0.4,
            #     "speed": 1.1,
            #     "stability": 0.3,
            #     "style": 1,
            #     "use_speaker_boost": True,
            #	  "speaking_rate": 0.8
            # }
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )
    await publish_background(ctx.room, "office.mp3")
    await session.generate_reply(
        instructions="say: Hello. Welcome to Toyota One. This is Serra and i will be happy to schedule your service appointment. Who do I have the pleasure of speaking with?"
    )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        # agent_name is required for explicit dispatch
        # agent_name="my-telephony-agent"
    ))
