import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, CreateAgentDispatchRequest, LiveKitAPI, VideoGrants

load_dotenv(".env.local")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

app = FastAPI(title="Underwriting Voice Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session")
async def create_session() -> dict[str, str]:
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(status_code=503, detail="LiveKit is not configured")

    session_id = uuid4().hex
    room_name = f"voice-{session_id}"
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(f"user-{session_id}")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    livekit = LiveKitAPI()
    try:
        await livekit.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name="underwriting-agent",
                room=room_name,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent dispatch failed") from exc
    finally:
        await livekit.aclose()

    return {"livekit_url": LIVEKIT_URL, "token": token}
