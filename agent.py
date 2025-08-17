from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
)

load_dotenv()

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
            # ✅ cần await
            await source.capture_frame(frame)
            await asyncio.sleep(0.02)

    asyncio.create_task(read_audio())

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. Help customer book service appointment at Autoservice AI. Response english only")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
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


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,

        # agent_name is required for explicit dispatch
        agent_name="my-telephony-agent"
    ))