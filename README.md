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

To check several items in one turn, run:

```bash
uv run python -m evals.run --case multiple_items_in_one_turn
```

For a voice check, say “I need to send the invoice, I want to explore prefix caching,
and remind me tomorrow at eleven to call Alex.” The page should show one new task,
one new idea, and one new reminder. The reminder should be recorded without promising
notification delivery. The latest record in `voice_inbox_traces.jsonl` should contain
all three tool calls. Pause for about a second between the task, idea, and reminder
while speaking: the agent should wait for the complete thought before replying.

## Check conversational corrections

The agent can correct the most recently created task, idea, or reminder of each type
within the current conversation. A correction updates the existing row.

```bash
uv run python -m evals.run --case correct_recent_task
uv run python -m evals.run --case correct_recent_idea
uv run python -m evals.run --case correct_recent_reminder
```

For a voice check, say “Remind me tomorrow at eleven to call Alex,” wait for the
response, then say “Actually make that four in the afternoon.” The page should
show one reminder for 4 PM. Refresh the page to confirm the correction persisted.
Recent-item references currently last for one conversation session.
