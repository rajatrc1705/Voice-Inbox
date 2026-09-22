import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    RunContext,
    ToolError,
    UserInputTranscribedEvent,
    function_tool,
    room_io,
)
from livekit.plugins import noise_cancellation, openai
from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad

from voice_inbox.repository import VoiceInboxRepository
from voice_inbox.tracing import ToolCallTrace, TraceWriter, TurnTrace

load_dotenv(".env.local")

repository = VoiceInboxRepository(Path(__file__).with_name("voice_inbox.db"))
trace_writer = TraceWriter(Path(__file__).with_name("voice_inbox_traces.jsonl"))


@dataclass
class SessionState:
    repository: VoiceInboxRepository
    session_id: str = field(default_factory=lambda: str(uuid4()))
    source_transcript: str = ""
    transcript_ready: asyncio.Event = field(default_factory=asyncio.Event)
    active_turn: TurnTrace | None = None


def start_turn(state: SessionState, transcript: str, started_at: float) -> None:
    state.active_turn = TurnTrace(
        session_id=state.session_id,
        turn_id=str(uuid4()),
        user_transcript=transcript,
        started_at=started_at,
    )


def record_tool_calls(
    state: SessionState, event: FunctionToolsExecutedEvent
) -> None:
    if state.active_turn is None:
        return

    for function_call, output in event.zipped():
        result = output.output if output is not None and not output.is_error else None
        error = output.output if output is not None and output.is_error else None
        state.active_turn.tool_calls.append(
            ToolCallTrace(
                name=function_call.name,
                arguments=json.loads(function_call.arguments),
                result=result,
                error=error,
                started_at=function_call.created_at,
                completed_at=output.created_at if output is not None else event.created_at,
            )
        )


def complete_turn(
    state: SessionState, assistant_response: str, completed_at: float
) -> None:
    if state.active_turn is None:
        return

    state.active_turn.assistant_response = assistant_response
    state.active_turn.completed_at = completed_at
    trace_writer.write(state.active_turn)
    state.active_turn = None


def build_instructions(now: datetime | None = None) -> str:
    timezone = ZoneInfo(os.getenv("VOICE_INBOX_TIMEZONE", "Europe/Berlin"))
    local_now = now.astimezone(timezone) if now is not None else datetime.now(timezone)
    return (
        "You are Voice Inbox, a concise English-speaking voice agent that turns messy "
        "spoken thoughts into useful items. Use create_task for a concrete action the "
        "user is clearly committed or obligated to perform. Use create_idea for a "
        "concept, possibility, experiment, or topic they want to retain without a clear "
        "commitment. If exploratory language is ambiguous, prefer an idea. Use "
        "create_reminder only when the user explicitly asks to be notified and provides "
        "enough date and time information to calculate a concrete timestamp. Ask only "
        "for missing reminder timing instead of calling the tool. A reminder request is "
        "not also a task. Do not capture completed actions. When one utterance contains "
        "several independent items, call the appropriate tool once for each item. "
        "Only claim an item was captured after its tool succeeds. "
        "Reminder delivery is not active yet. After create_reminder succeeds, say "
        "'Recorded your reminder. Notifications are not active yet.' You may include "
        "the title and time. Never say the reminder is set or scheduled, or promise "
        "to remind, notify, or alert the user. "
        f"The current local time is {local_now.isoformat()} "
        f"in {timezone.key}. Keep spoken responses brief and natural."
    )


def build_agent(now: datetime | None = None) -> Agent:
    return Agent(
        instructions=build_instructions(now),
        tools=[create_task, create_idea, create_reminder],
    )

def build_realtime_model() -> openai.realtime.RealtimeModel:
    return openai.realtime.RealtimeModel(
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        voice=os.getenv("OPENAI_VOICE", "coral"),
        turn_detection=SemanticVad(
            type="semantic_vad",
            eagerness="low",
            create_response=True,
            interrupt_response=True,
        ),
    )


def build_session(
    session_repository: VoiceInboxRepository,
    realtime_model: openai.realtime.RealtimeModel | None = None,
) -> AgentSession[SessionState]:
    return AgentSession[SessionState](
        userdata=SessionState(repository=session_repository),
        llm=realtime_model or build_realtime_model(),
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

    task = context.userdata.repository.create_task(
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

    idea = context.userdata.repository.create_idea(
        text=text,
        source_transcript=await source_transcript(context),
    )
    return f"Created idea {idea.id}: {idea.text}"


@function_tool
async def create_reminder(
    context: RunContext[SessionState], title: str, trigger_at: str
) -> str:
    """Store a reminder request and its requested time; no notification is scheduled.

    Delivery is inactive. Confirm only that the reminder was recorded and explain
    that notifications are not active yet.

    Args:
        title: A short description of what the user wants to be reminded about.
        trigger_at: The reminder time as an ISO 8601 timestamp with UTC offset.
    """

    try:
        trigger_time = datetime.fromisoformat(trigger_at)
    except ValueError as error:
        raise ToolError("trigger_at must be an ISO 8601 timestamp.") from error

    reminder = context.userdata.repository.create_reminder(
        title=title,
        trigger_at=trigger_time,
        source_transcript=await source_transcript(context),
    )
    return (
        f"Recorded reminder {reminder.id}: {reminder.title} at {trigger_at}. "
        "Notifications are not active yet; no notification has been scheduled."
    )


def initialize_process(_: object) -> None:
    repository.initialize()


server = AgentServer(setup_fnc=initialize_process)


@server.rtc_session(agent_name="voice-inbox-agent")
async def voice_inbox_agent(ctx: agents.JobContext) -> None:
    await ctx.connect()
    # event-driven coordinator, does not itself understand the language
    session = build_session(repository)

    @session.on("user_input_transcribed")
    def record_user_transcript(event: UserInputTranscribedEvent) -> None:
        if event.is_final:
            session.userdata.source_transcript = event.transcript
            session.userdata.transcript_ready.set()
            start_turn(session.userdata, event.transcript, event.created_at)
        else:
            session.userdata.transcript_ready.clear()

    @session.on("function_tools_executed")
    def record_executed_tools(event: FunctionToolsExecutedEvent) -> None:
        record_tool_calls(session.userdata, event)

    @session.on("conversation_item_added")
    def record_conversation_item(event: ConversationItemAddedEvent) -> None:
        if event.item.role != "assistant":
            return
        response = event.item.text_content
        if response is not None:
            complete_turn(session.userdata, response, event.created_at)

    await session.start(
        room=ctx.room,
        agent=build_agent(),
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
