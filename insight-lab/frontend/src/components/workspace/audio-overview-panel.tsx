"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Pause, Play, Volume2 } from "lucide-react";
import { apiFetch, getApiUrl, type AudioOverviewResponse } from "@/lib/api";
import { downloadTextFile } from "@/lib/export-utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type AudioOverviewPanelProps = {
  documentId: string;
  accessToken: string | null;
  ready: boolean;
  overview?: AudioOverviewResponse | null;
  onGenerated?: (overview: AudioOverviewResponse) => void;
  onPlayingChange?: (playing: boolean, overview: AudioOverviewResponse | null) => void;
  onControlsReady?: (controls: { playPause: () => void; stop: () => void }) => void;
};

export function AudioOverviewPanel({
  documentId,
  accessToken,
  ready,
  overview,
  onGenerated,
  onPlayingChange,
  onControlsReady,
}: AudioOverviewPanelProps) {
  const [localOverview, setLocalOverview] = useState<AudioOverviewResponse | null>(overview ?? null);
  const [generating, setGenerating] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    setLocalOverview(overview ?? null);
  }, [overview]);

  const revokeAudioUrl = useCallback(() => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  }, [audioUrl]);

  const loadMp3 = useCallback(async () => {
    if (!accessToken || !localOverview?.has_audio) {
      return;
    }
    revokeAudioUrl();
    const response = await fetch(getApiUrl(`/documents/${documentId}/audio-overview/mp3`), {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) {
      return;
    }
    const blob = await response.blob();
    setAudioUrl(URL.createObjectURL(blob));
  }, [accessToken, documentId, localOverview?.has_audio, revokeAudioUrl]);

  useEffect(() => {
    void loadMp3();
    return () => revokeAudioUrl();
  }, [loadMp3, revokeAudioUrl]);

  const loadExisting = useCallback(async () => {
    if (!accessToken || !ready) {
      return;
    }
    const response = await apiFetch(`/documents/${documentId}/audio-overview`, accessToken);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    if (data.audio_overview) {
      setLocalOverview(data.audio_overview as AudioOverviewResponse);
    }
  }, [accessToken, documentId, ready]);

  useEffect(() => {
    void loadExisting();
  }, [loadExisting]);

  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
      revokeAudioUrl();
    };
  }, [revokeAudioUrl]);

  async function handleGenerate() {
    if (!accessToken || !ready) {
      return;
    }
    setGenerating(true);
    setError(null);
    try {
      const response = await apiFetch(`/documents/${documentId}/audio-overview/generate`, accessToken, {
        method: "POST",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || "Could not generate audio overview.");
        return;
      }
      const data = await response.json();
      const next = {
        document_id: documentId,
        title: data.title,
        script: data.script,
        estimated_minutes: data.estimated_minutes,
        has_audio: data.has_audio,
        overview_id: data.overview_id,
      } as AudioOverviewResponse;
      setLocalOverview(next);
      onGenerated?.(next);
    } finally {
      setGenerating(false);
    }
  }

  const stopAll = useCallback(() => {
    audioRef.current?.pause();
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }
    window.speechSynthesis?.cancel();
    setPlaying(false);
    onPlayingChange?.(false, localOverview);
  }, [localOverview, onPlayingChange]);

  const togglePlay = useCallback(() => {
    if (audioUrl && audioRef.current) {
      if (playing) {
        audioRef.current.pause();
        setPlaying(false);
        onPlayingChange?.(false, localOverview);
      } else {
        void audioRef.current.play();
        setPlaying(true);
        onPlayingChange?.(true, localOverview);
      }
      return;
    }

    if (!localOverview?.script || typeof window === "undefined" || !window.speechSynthesis) {
      return;
    }
    if (playing) {
      window.speechSynthesis.cancel();
      setPlaying(false);
      onPlayingChange?.(false, localOverview);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(localOverview.script);
    utterance.onend = () => {
      setPlaying(false);
      onPlayingChange?.(false, localOverview);
    };
    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    setPlaying(true);
    onPlayingChange?.(true, localOverview);
  }, [audioUrl, localOverview, onPlayingChange, playing]);

  useEffect(() => {
    onControlsReady?.({ playPause: togglePlay, stop: stopAll });
  }, [onControlsReady, togglePlay, stopAll]);

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Audio overview</CardTitle>
        <CardDescription>
          {localOverview?.has_audio
            ? "Pre-generated MP3 narration from your document summary."
            : "Generate a narrated summary. MP3 is created when TTS is available on the server."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!localOverview ? (
          <Button type="button" disabled={!ready || generating} onClick={() => void handleGenerate()}>
            {generating ? "Generating…" : "Generate audio overview"}
          </Button>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="default" onClick={togglePlay}>
                {playing ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                {playing ? "Pause" : "Play"}
              </Button>
              <Button type="button" variant="outline" onClick={stopAll}>
                Stop
              </Button>
              {localOverview.script ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => downloadTextFile(localOverview.script, "audio-overview-script.txt")}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Script
                </Button>
              ) : null}
            </div>
            {audioUrl ? (
              <audio
                ref={audioRef}
                src={audioUrl}
                className="w-full"
                controls
                onPlay={() => {
                  setPlaying(true);
                  onPlayingChange?.(true, localOverview);
                }}
                onPause={() => {
                  setPlaying(false);
                  onPlayingChange?.(false, localOverview);
                }}
                onEnded={() => {
                  setPlaying(false);
                  onPlayingChange?.(false, localOverview);
                }}
              />
            ) : (
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Volume2 className="h-4 w-4" />
                Using browser voice — regenerate on server with TTS for MP3 quality.
              </p>
            )}
            {localOverview.estimated_minutes ? (
              <p className="text-xs text-muted-foreground">~{localOverview.estimated_minutes} min listen</p>
            ) : null}
            <div className="max-h-48 overflow-y-auto rounded-md border bg-muted/30 p-3 text-sm text-muted-foreground whitespace-pre-wrap">
              {localOverview.script}
            </div>
          </>
        )}
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
      </CardContent>
    </Card>
  );
}
