import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    ToolError,
    UserInputTranscribedEvent,
    function_tool,
    room_io,
)
from livekit.plugins import noise_cancellation, openai
from openai.types.beta.realtime.session import TurnDetection

from voice_inbox.repository import VoiceInboxRepository

load_dotenv(".env.local")

INSTRUCTIONS = (
    "You are Voice Inbox, a concise voice agent which speaks english and that helps the user make sense of "
    "messy spoken thoughts. When the user clearly states something they intend to do, "
    "call create_task. Do not create tasks for completed actions, ideas, notes, reminder "
    "requests, or tentative thinking aloud. You cannot save ideas or schedule reminders "
    "yet, so never claim that those were saved. Only say a task was captured after "
    "create_task succeeds. Keep spoken responses brief and natural."
)

repository = VoiceInboxRepository(Path(__file__).with_name("voice_inbox.db"))


@dataclass
class SessionState:
    source_transcript: str = ""
    transcript_ready: asyncio.Event = field(default_factory=asyncio.Event)


@function_tool
async def create_task(context: RunContext[SessionState], title: str) -> str:
    """Save a task that the user clearly intends to do."""
    try:
        async with asyncio.timeout(3):
            await context.userdata.transcript_ready.wait()
    except TimeoutError as error:
        raise ToolError("The final user transcript is not available yet.") from error

    task = repository.create_task(
        title=title,
        source_transcript=context.userdata.source_transcript,
    )
    return f"Created task {task.id}: {task.title}"


def initialize_process(_: object) -> None:
    repository.initialize()


server = AgentServer(setup_fnc=initialize_process)


@server.rtc_session(agent_name="voice-inbox-agent")
async def voice_inbox_agent(ctx: agents.JobContext) -> None:
    await ctx.connect()
    # event-driven coordinator, does not itself understand the language
    session = AgentSession[SessionState](
        userdata=SessionState(),

        llm=openai.realtime.RealtimeModel(
            voice=os.getenv("OPENAI_VOICE", "coral"),
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.6,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True,
                interrupt_response=True,
            ),
        )
    )

    @session.on("user_input_transcribed")
    def record_user_transcript(event: UserInputTranscribedEvent) -> None:
        if event.is_final:
            session.userdata.source_transcript = event.transcript
            session.userdata.transcript_ready.set()
        else:
            session.userdata.transcript_ready.clear()

    await session.start(
        room=ctx.room,
        agent=Agent(instructions=INSTRUCTIONS, tools=[create_task]),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind
                == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    await session.generate_reply(
        instructions="Greet the user briefly in English and ask what is on their mind."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
