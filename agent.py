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
        super().__init__(instructions="You are a helpful voice AI assistant. Help customer book service appointment at Autoservice AI. Response english only")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini", api_key="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"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="en-US", api_key="98e42a6de0d4660afec26ebcbda8499f42bd4b5d"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=elevenlabs.TTS(
            voice_id="xcK84VTjd6MHGJo2JVfS",
            model="eleven_flash_v2_5",
            api_key="59bb59df13287e23ba2da37ea6e48724",
            # voice_settings= {
            #     "similarity_boost": 0.4,
            #     "speed": 1.1,
            #     "stability": 0.3,
            #     "style": 1,
            #     "use_speaker_boost": True
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
        instructions="Greet the user and Help customer book service appointment at Autoservice AI. English"
    )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        # agent_name is required for explicit dispatch
        agent_name="my-telephony-agent"
    ))