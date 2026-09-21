# Voice Inbox

Open the page, press the button, speak naturally, and see the live transcript.

Tasks, ideas, and reminder requests are saved locally. Reminders record the requested
time, but notification delivery is not active yet.

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

## Run agent evaluations

The eval runner sends scripted text turns through the same agent instructions, model,
and tools without using microphone audio, transcription, or speech playback. Each case
uses a temporary database.

```bash
uv run python -m evals.run
```

Run one case or repeat every case:

```bash
uv run python -m evals.run --case clear_task
uv run python -m evals.run --repetitions 3
```

Reports are written to the ignored `eval_results/` directory.
These runs call the configured OpenAI Realtime model and incur API usage. They send
the scripted case text and agent instructions, using temporary databases rather than
your saved inbox items. They do not verify microphone capture or audio playback.
