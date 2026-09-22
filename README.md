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

## Ask about local files

Put `.txt`, `.md`, or text-based `.pdf` files in `workspace/`, or set
`AGENT_WORKSPACE_DIR` in `.env.local` to another directory. The agent can list,
search, and read files in that directory only. Files in `workspace/` are ignored by Git.
Scanned PDFs need text extraction before the agent can use them.

For a basic voice check, add `workspace/resume.md` containing
`In 2024 I worked at Acme Robotics.` and ask: “Search my local files for my
resume. Where did I work in 2024? Cite the file.” The reply should name Acme
Robotics and `resume.md`. The last entry in `voice_inbox_traces.jsonl` should show
`search_files` followed by `read_file`, both successful.

The quantitative text eval uses its own synthetic files and checks tool order,
successful reads, the answer, and the file citation:

```bash
uv run python -m evals.run --case local_file_resume_lookup --repetitions 3
```

Check `case_success` in the printed report; `3/3` means all three runs met every
check. Text evals do not test microphone capture or speech playback.

## Compare a webpage with a local file

Before starting a conversation, paste a public HTML URL into the **Webpage URL to
discuss** field. The URL belongs to that conversation. The agent reads the page
only when you ask about it, extracts its main text, and sends at most 12,000
characters to the model. Private network URLs and non-HTML pages are rejected.
JavaScript-only pages may not contain readable text in their initial HTML.

For a quick voice test, keep `workspace/resume.md` with
`In 2024 I worked at Acme Robotics.`, paste `https://example.com` into the URL
field, start a new conversation, and ask: “Read the webpage I supplied and search
my local files for my resume. What is this domain for, and where did I work in 2024? Cite both
sources.” The answer should say the domain is for documentation examples, name
Acme Robotics, and cite both `example.com` and `resume.md`. The last record in
`voice_inbox_traces.jsonl` should contain successful `read_page`, `search_files`,
and `read_file` calls.

The quantitative comparison eval uses a fixed synthetic job posting and resume:

```bash
uv run python -m evals.run --case compare_web_posting_with_resume --repetitions 3
```

It passes when the agent reads both sources, identifies Kubernetes as a posting
requirement not mentioned in the resume, cites both, and makes no inbox items.
The eval supplies fixed page HTML, so it does not depend on a live website.

If a pause splits the voice request into two turns, ask the follow-up “What is
this domain for and where did I work in 2024? Cite both sources.” The agent
should use the page and resume it read in the first turn, without fetching them
again. Automatic turn detection can still treat a pause as the end of a turn.

The follow-up behavior has its own two-turn text eval:

```bash
uv run python -m evals.run --case web_question_continues_after_source_request --repetitions 3
```

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
