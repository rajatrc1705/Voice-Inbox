# Voice Inbox

Open the page, press the button, speak naturally, and see the live transcript.

## Architecture

The single Next.js page calls `POST /session` on the small FastAPI service. That endpoint creates a private LiveKit room, dispatches `agent.py`, and returns a short-lived browser token. LiveKit carries microphone audio, agent audio, transcription events, and agent state. `agent.py` uses OpenAI Realtime for speech-to-speech inference and exposes tools that store tasks, ideas, and reminders in SQLite. FastAPI serves those items to the page, while the visible conversation transcript remains browser-only.

## Run locally

```bash
cp .env.example .env.local
uv sync
uv run uvicorn api.main:app --reload
```

In two more terminals:

```bash
uv run python agent.py dev
cd web && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and press **Start conversation**.
