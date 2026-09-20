import os

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, room_io
from livekit.plugins import noise_cancellation, openai
from openai.types.beta.realtime.session import TurnDetection

load_dotenv(".env.local")

INSTRUCTIONS = (
    "You are Voice Inbox, a concise voice agent that helps the user make sense of "
    "messy spoken thoughts. Identify tasks, ideas, and explicit reminder requests, "
    "including multiple items in one utterance, and summarize them clearly. Distinguish "
    "between thinking aloud and asking for an action. You do not have tools yet, so "
    "never claim that anything was saved or scheduled. Keep spoken responses brief "
    "and natural."
)

server = AgentServer()


@server.rtc_session(agent_name="voice-inbox-agent")
async def voice_inbox_agent(ctx: agents.JobContext) -> None:
    await ctx.connect()
    # event-driven coordinator, does not itself understand the language
    session = AgentSession(

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
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=INSTRUCTIONS),
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
        instructions="Greet the user briefly and ask what is on their mind."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
