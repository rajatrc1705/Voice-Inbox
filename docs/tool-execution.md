# Tool execution with guardrails

The six mutation tools now return structured JSON outcomes from
`voice_inbox/execution.py`. Live voice and text evaluations call the same adapters
in `agent.py`; direct executor tests require neither LiveKit nor a model session.
Read-only tools retain their current implementation.

## Contract

An attempt returns `action_id`, `action`, `status`, `reason`, `next_step`, `result`,
`effect`, and `recording_error`.

- `executed`: the operation committed; `result` contains the persisted item.
- `blocked`: validation or a precondition prevented mutation; `effect` is `none`.
- `failed`: an unexpected execution error occurred. An attempted side effect is
  marked `unknown`, not assumed safe to retry.

Reasons include `invalid_arguments`, `empty_content`, `invalid_timestamp`,
`missing_timezone`, `context_unavailable`, `target_not_found`, `target_inactive`,
`no_changes`, `unknown_action`, and `execution_error`.
`next_step` distinguishes clarification from correcting a proposal or doing nothing.
A recording failure preserves the actual action outcome and adds
`recording_error: recording_failed`; it must not trigger a duplicate operation.

The model supplies only declared arguments. Recent target IDs and source
transcripts come from runtime context. Unknown fields are rejected, including
attempts to supply a target ID or override the transcript. Required strings are
nonblank and trimmed. Timestamp inputs must parse with an explicit UTC offset.
We do not impose future-time rules or infer whether the user supplied enough
information. Missing recent targets require clarification; identical changes
return `no_changes`.

The repository acquires a SQLite write transaction before checking update target
existence, status, and current values. It issues no mutation SQL for rejected
updates. The transaction closes on every path. This is session-scoped target
selection, not multi-user authorization or natural-language reference resolution.

## Observability and evaluation

Live actions append one completed-attempt record to the ignored local file
`voice_operator_actions.jsonl`. It includes the proposal, outcome, available
session/turn/tool-call IDs, and validation/repository durations. Repository time
includes transactional precondition checks. This timing excludes transcript
waiting, model inference, audio, and record writing.

These records do not depend on completion of the existing turn trace. The
premature turn-completion bug is still a separate issue. A process crash before
an attempt finishes can leave no completed record: this is not a durable audit
log. Correlation during overlapping/interrupted voice turns still needs the
planned lifecycle work.

Text eval reports include `action_events` for completed turns. If a later model or
session failure occurs, the report retains completed turns and stores records
from the unfinished turn in `incomplete_action_events`, alongside `execution_error`.
Startup failures also close the session; cleanup errors remain execution errors.
Expected calls can
assert `outcome.status`, `outcome.reason`, `outcome.next_step`, and `outcome.effect`.
For structured outcomes, `succeeds` means the action executed, rather than merely
that LiveKit returned a non-error tool response. Missing or malformed mutation
outcomes cannot count as successful execution. Existing mutation cases now
require `executed / ok / committed`; no existing assertions were weakened.

## Free local validation

```bash
.venv/bin/python -m unittest discover -s tests
```

For a direct manual check against a disposable database:

```bash
.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path
from voice_inbox.execution import ActionContext, execute_action
from voice_inbox.repository import VoiceInboxRepository

with tempfile.TemporaryDirectory() as directory:
    repo = VoiceInboxRepository(Path(directory) / 'check.db')
    repo.initialize()
    ctx = ActionContext('manual-session', {}, 'Capture a demo task')
    created = execute_action('create_task', {'title': 'Demo task'}, ctx, repo)
    before = repo.list_tasks()
    blocked = execute_action('update_recent_task', {'title': 'Changed'}, ctx, repo)
    assert created.status == 'executed'
    assert blocked.reason == 'target_not_found'
    assert repo.list_tasks() == before
    print(created.to_dict())
    print(blocked.to_dict())
PY
```

Optional voice check (uses paid inference): start the existing backend, agent,
and frontend as described in the README. In a fresh conversation ask to capture
“Send the demo invoice”, then correct it to “Send the demo report”. Verify two
`executed` action records and the same persisted item ID. Ask to correct an idea
before creating one: the model may clarify without calling a tool, or its tool
request will return `blocked / target_not_found`. Neither should create an idea.
Use the direct check above to exercise blocking deterministically.

## Extending the feature

The current registry describes the six mutation actions. Add an input schema and
handler to reuse outcomes and recording. The schema validator intentionally
supports only the current string/timestamp types; extend it explicitly for new
argument types. Keep side-effect-specific preconditions close to their atomic
execution boundary. Browser/Mac capabilities will need their own target context
and policies, while reusing the outcome/recording contract.

No new framework, model call, retry deduplication, full trace replay, load test,
or model backend abstraction is introduced here. LiveKit can reject malformed
calls before the adapter runs; those remain framework errors rather than executor
outcomes. Existing synchronous SQLite calls still run on the session event loop;
concurrency and latency limits remain to be measured and addressed.
