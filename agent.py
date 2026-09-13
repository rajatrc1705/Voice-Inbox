import os

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, room_io
from livekit.plugins import noise_cancellation, openai
from openai.types.beta.realtime.session import TurnDetection

load_dotenv(".env.local")

INSTRUCTIONS = (
    "You are a concise underwriting voice agent. Help the user talk through a loan "
    "application and identify information an underwriter would need. Ask focused "
    "questions one at a time, adapting to what the user says. Cover relevant topics "
    "such as loan purpose and amount, repayment source, financial performance, "
    "existing obligations, collateral, and material risks. Do not promise approval, "
    "terms, pricing, or policy exceptions. Keep spoken responses brief and natural."
)

server = AgentServer()


@server.rtc_session(agent_name="underwriting-agent")
async def underwriting_agent(ctx: agents.JobContext) -> None:
    await ctx.connect()
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
        instructions="Greet the user briefly and ask how you can help with their loan application."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
