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

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function createSession(): Promise<{ livekit_url: string; token: string }> {
  const response = await fetch(`${API_URL}/session`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
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
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [transcript, setTranscript] = useState<TranscriptTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const roomRef = useRef<Room | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
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
      if (nextState) setVoiceState(nextState);
    });

    room.on(RoomEvent.Disconnected, () => {
      if (roomRef.current === room) roomRef.current = null;
      setVoiceState("idle");
    });

    try {
      const session = await createSession();
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
        <h1 className="text-2xl font-semibold">Underwriting Voice Agent</h1>
        <p className="mt-2 text-sm text-zinc-500">
          Start a conversation and speak naturally.
        </p>
      </header>

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
