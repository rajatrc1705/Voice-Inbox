"use client";

import { useEffect, useRef, useState } from "react";
import {
  Room,
  RoomEvent,
  Track,
  type Participant,
  type RemoteTrack,
  type TranscriptionSegment,
} from "livekit-client";

type VoiceState = "idle" | "listening" | "processing" | "speaking" | "error";
type TranscriptTurn = {
  id: string;
  speaker: "user" | "agent";
  text: string;
  final: boolean;
};
type TaskItem = {
  id: string;
  title: string;
  status: string;
};
type IdeaItem = {
  id: string;
  text: string;
};
type ReminderItem = {
  id: string;
  title: string;
  trigger_at: string;
  status: string;
};
type StoredItems = {
  tasks: TaskItem[];
  ideas: IdeaItem[];
  reminders: ReminderItem[];
};

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function createSession(pageUrl: string): Promise<{ livekit_url: string; token: string }> {
  const response = await fetch(`${API_URL}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ page_url: pageUrl || null }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function getTasks(): Promise<TaskItem[]> {
  const response = await fetch(`${API_URL}/tasks`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function getIdeas(): Promise<IdeaItem[]> {
  const response = await fetch(`${API_URL}/ideas`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function getReminders(): Promise<ReminderItem[]> {
  const response = await fetch(`${API_URL}/reminders`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function getStoredItems(): Promise<StoredItems> {
  const [tasks, ideas, reminders] = await Promise.all([
    getTasks(),
    getIdeas(),
    getReminders(),
  ]);
  return { tasks, ideas, reminders };
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function agentState(value?: string): VoiceState | null {
  if (value === "speaking") return "speaking";
  if (value === "listening" || value === "idle") return "listening";
  if (value === "thinking" || value === "initializing") return "processing";
  return null;
}

const stateLabel: Record<VoiceState, string> = {
  idle: "Ready",
  listening: "Listening…",
  processing: "Thinking…",
  speaking: "Speaking…",
  error: "Something went wrong",
};

export default function Home() {
  const [pageUrl, setPageUrl] = useState("");
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [transcript, setTranscript] = useState<TranscriptTurn[]>([]);
  const [items, setItems] = useState<StoredItems>({
    tasks: [],
    ideas: [],
    reminders: [],
  });
  const [error, setError] = useState<string | null>(null);
  const roomRef = useRef<Room | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let mounted = true;
    void getStoredItems()
      .then((storedItems) => {
        if (mounted) setItems(storedItems);
      })
      .catch(() => undefined);

    return () => {
      mounted = false;
      void roomRef.current?.disconnect();
    };
  }, []);

  const endConversation = async () => {
    const room = roomRef.current;
    roomRef.current = null;
    if (room) await room.disconnect();
    setVoiceState("idle");
  };

  const startConversation = async () => {
    setError(null);
    setTranscript([]);
    setVoiceState("processing");

    const room = new Room();
    roomRef.current = room;

    room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
      if (track.kind === Track.Kind.Audio && audioRef.current) {
        track.attach(audioRef.current);
      }
    });

    room.on(
      RoomEvent.TranscriptionReceived,
      (segments: TranscriptionSegment[], participant?: Participant) => {
        const speaker: "user" | "agent" =
          participant?.identity === room.localParticipant.identity ? "user" : "agent";
        setTranscript((current) => {
          const next = [...current];
          for (const segment of segments) {
            if (!segment.text.trim()) continue;
            const id = `${speaker}-${segment.id}`;
            const turn = { id, speaker, text: segment.text, final: segment.final };
            const index = next.findIndex((item) => item.id === id);
            if (index === -1) next.push(turn);
            else next[index] = turn;
          }
          return next;
        });
      },
    );

    room.on(RoomEvent.ParticipantAttributesChanged, (attributes) => {
      const nextState = agentState(attributes["lk.agent.state"]);
      if (nextState) {
        setVoiceState(nextState);
        if (nextState === "listening") {
          void getStoredItems().then(setItems).catch(() => undefined);
        }
      }
    });

    room.on(RoomEvent.Disconnected, () => {
      if (roomRef.current === room) roomRef.current = null;
      setVoiceState("idle");
    });

    try {
      const session = await createSession(pageUrl.trim());
      await room.connect(session.livekit_url, session.token);
      await room.startAudio();
      await room.localParticipant.setMicrophoneEnabled(true);
      setVoiceState("listening");
    } catch (cause) {
      await room.disconnect();
      roomRef.current = null;
      setError(cause instanceof Error ? cause.message : "Unable to start conversation");
      setVoiceState("error");
    }
  };

  const active = voiceState !== "idle" && voiceState !== "error";

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col px-6 py-16 text-zinc-900">
      <header className="text-center">
        <h1 className="text-2xl font-semibold">Voice Inbox</h1>
        <p className="mt-2 text-sm text-zinc-500">
          Speak naturally and untangle what is on your mind.
        </p>
      </header>

      <label className="mt-8 flex flex-col gap-2 text-sm font-medium text-zinc-700">
        Public webpage URL to discuss (optional)
        <input
          type="url"
          value={pageUrl}
          onChange={(event) => setPageUrl(event.target.value)}
          disabled={voiceState !== "idle" && voiceState !== "error"}
          placeholder="https://example.com/job-posting"
          className="rounded-lg border border-zinc-300 bg-white px-3 py-2 font-normal text-zinc-900 disabled:bg-zinc-100"
        />
      </label>

      <section className="flex flex-col items-center py-16">
        <button
          type="button"
          aria-label={active ? "End conversation" : "Start conversation"}
          onClick={active ? endConversation : startConversation}
          className={`flex h-44 w-44 items-center justify-center rounded-full text-white shadow-lg transition hover:scale-[1.02] active:scale-95 ${
            voiceState === "error"
              ? "bg-red-600"
              : active
                ? "bg-zinc-900"
                : "bg-blue-600"
          } ${voiceState === "listening" ? "animate-pulse" : ""}`}
        >
          {voiceState === "speaking" ? (
            <span className="flex h-12 items-center gap-2" aria-hidden="true">
              <span className="voice-bar" />
              <span className="voice-bar [animation-delay:120ms]" />
              <span className="voice-bar [animation-delay:240ms]" />
              <span className="voice-bar [animation-delay:360ms]" />
            </span>
          ) : active ? (
            <span className="h-8 w-8 rounded-sm bg-white" aria-hidden="true" />
          ) : (
            <span className="px-7 text-center text-base font-semibold">
              {voiceState === "error" ? "Try again" : "Start conversation"}
            </span>
          )}
        </button>
        <p className="mt-6 text-sm font-medium text-zinc-600">{stateLabel[voiceState]}</p>
        {error && <p className="mt-2 max-w-md text-center text-sm text-red-600">{error}</p>}
        <audio ref={audioRef} autoPlay />
      </section>

      <section className="grid gap-8 border-t border-zinc-200 py-8 sm:grid-cols-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
            Tasks
          </h2>
          {items.tasks.length === 0 ? (
            <p className="mt-4 text-sm text-zinc-400">No tasks yet.</p>
          ) : (
            <ul className="mt-4 space-y-4">
              {items.tasks.map((task) => (
                <li key={task.id}>
                  <p className="text-sm font-medium">{task.title}</p>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
            Ideas
          </h2>
          {items.ideas.length === 0 ? (
            <p className="mt-4 text-sm text-zinc-400">No ideas yet.</p>
          ) : (
            <ul className="mt-4 space-y-4">
              {items.ideas.map((idea) => (
                <li key={idea.id}>
                  <p className="text-sm font-medium">{idea.text}</p>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
            Reminders
          </h2>
          {items.reminders.length === 0 ? (
            <p className="mt-4 text-sm text-zinc-400">No reminders yet.</p>
          ) : (
            <ul className="mt-4 space-y-4">
              {items.reminders.map((reminder) => (
                <li key={reminder.id}>
                  <p className="text-sm font-medium">{reminder.title}</p>
                  <p className="mt-1 text-xs text-zinc-500">
                    {formatDate(reminder.trigger_at)}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section aria-live="polite" className="border-t border-zinc-200 pt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
          Conversation
        </h2>
        {transcript.length === 0 ? (
          <p className="py-10 text-sm text-zinc-400">
            Your conversation will appear here.
          </p>
        ) : (
          <div className="mt-6 space-y-6">
            {transcript.map((turn) => (
              <div key={turn.id} className={turn.final ? "" : "opacity-60"}>
                <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  {turn.speaker === "user" ? "You" : "Agent"}
                </p>
                <p className="mt-1 whitespace-pre-wrap leading-7">{turn.text}</p>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
