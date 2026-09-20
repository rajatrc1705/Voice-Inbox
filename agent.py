import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

repository = VoiceInboxRepository(Path(__file__).with_name("voice_inbox.db"))


@dataclass
class SessionState:
    source_transcript: str = ""
    transcript_ready: asyncio.Event = field(default_factory=asyncio.Event)


def build_instructions() -> str:
    timezone = ZoneInfo(os.getenv("VOICE_INBOX_TIMEZONE", "Europe/Berlin"))
    local_now = datetime.now(timezone)
    return (
        "You are Voice Inbox, a concise English-speaking voice agent that turns messy "
        "spoken thoughts into useful items. Use create_task for a concrete action the "
        "user is clearly committed or obligated to perform. Use create_idea for a "
        "concept, possibility, experiment, or topic they want to retain without a clear "
        "commitment. If exploratory language is ambiguous, prefer an idea. Use "
        "create_reminder only when the user explicitly asks to be notified and provides "
        "enough date and time information to calculate a concrete timestamp. Ask only "
        "for missing reminder timing instead of calling the tool. A reminder request is "
        "not also a task. Do not capture completed actions. One utterance may require "
        "multiple tool calls. Only claim an item was captured after its tool succeeds. "
        "Reminder delivery is not active yet, so say a reminder was recorded, not that "
        f"the user will be notified. The current local time is {local_now.isoformat()} "
        f"in {timezone.key}. Keep spoken responses brief and natural."
    )


async def source_transcript(context: RunContext[SessionState]) -> str:
    try:
        async with asyncio.timeout(3):
            await context.userdata.transcript_ready.wait()
    except TimeoutError as error:
        raise ToolError("The final user transcript is not available yet.") from error
    return context.userdata.source_transcript


@function_tool
async def create_task(context: RunContext[SessionState], title: str) -> str:
    """Save a concrete action the user clearly intends to perform.

    Args:
        title: A short action-oriented title for the task.
    """

    task = repository.create_task(
        title=title,
        source_transcript=await source_transcript(context),
    )
    return f"Created task {task.id}: {task.title}"


@function_tool
async def create_idea(context: RunContext[SessionState], text: str) -> str:
    """Save an idea, possibility, experiment, or topic without making it a task.

    Args:
        text: A concise description of the idea.
    """

    idea = repository.create_idea(
        text=text,
        source_transcript=await source_transcript(context),
    )
    return f"Created idea {idea.id}: {idea.text}"


@function_tool
async def create_reminder(
    context: RunContext[SessionState], title: str, trigger_at: str
) -> str:
    """Record an explicit reminder request with a concrete notification time.

    Args:
        title: A short description of what the user wants to be reminded about.
        trigger_at: The reminder time as an ISO 8601 timestamp with UTC offset.
    """

    try:
        trigger_time = datetime.fromisoformat(trigger_at)
    except ValueError as error:
        raise ToolError("trigger_at must be an ISO 8601 timestamp.") from error

    reminder = repository.create_reminder(
        title=title,
        trigger_at=trigger_time,
        source_transcript=await source_transcript(context),
    )
    return f"Recorded reminder {reminder.id}: {reminder.title} at {trigger_at}"


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
        agent=Agent(
            instructions=build_instructions(),
            tools=[create_task, create_idea, create_reminder],
        ),
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
