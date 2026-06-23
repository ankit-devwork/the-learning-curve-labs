"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Pause, Play, Volume2 } from "lucide-react";
import { apiFetch, type AudioOverviewResponse } from "@/lib/api";
import { downloadTextFile } from "@/lib/export-utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type AudioOverviewPanelProps = {
  documentId: string;
  accessToken: string | null;
  ready: boolean;
  overview?: AudioOverviewResponse | null;
  onGenerated?: (overview: AudioOverviewResponse) => void;
};

export function AudioOverviewPanel({
  documentId,
  accessToken,
  ready,
  overview,
  onGenerated,
}: AudioOverviewPanelProps) {
  const [localOverview, setLocalOverview] = useState<AudioOverviewResponse | null>(overview ?? null);
  const [generating, setGenerating] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    setLocalOverview(overview ?? null);
  }, [overview]);

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
    };
  }, []);

  async function handleGenerate() {
    if (!accessToken) {
      return;
    }
    setGenerating(true);
    setError(null);
    try {
      const response = await apiFetch(
        `/documents/${documentId}/audio-overview/generate`,
        accessToken,
        { method: "POST" },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || "Failed to generate audio overview");
        return;
      }
      const data = (await response.json()) as AudioOverviewResponse;
      setLocalOverview(data);
      onGenerated?.(data);
    } finally {
      setGenerating(false);
    }
  }

  function handlePlayPause() {
    if (!localOverview?.script || typeof window === "undefined" || !window.speechSynthesis) {
      return;
    }

    if (playing) {
      window.speechSynthesis.cancel();
      setPlaying(false);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(localOverview.script);
    utterance.rate = 1;
    utterance.onend = () => setPlaying(false);
    utterance.onerror = () => setPlaying(false);
    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    setPlaying(true);
  }

  function handleStop() {
    window.speechSynthesis?.cancel();
    setPlaying(false);
  }

  return (
    <Card className="shadow-sm" data-tour="audio-overview">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Volume2 className="h-4 w-4 text-primary" aria-hidden />
          <CardTitle className="text-base">Audio overview</CardTitle>
        </div>
        <CardDescription>
          Listen to a narrated summary of this document using your browser&apos;s text-to-speech.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {!localOverview ? (
          <Button type="button" disabled={!ready || generating} onClick={() => void handleGenerate()}>
            {generating ? "Generating script…" : "Generate audio overview"}
          </Button>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="default" className="gap-2" onClick={handlePlayPause}>
                {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {playing ? "Pause" : "Play overview"}
              </Button>
              {playing ? (
                <Button type="button" variant="outline" onClick={handleStop}>
                  Stop
                </Button>
              ) : null}
              <Button
                type="button"
                variant="outline"
                className="gap-2"
                onClick={() =>
                  downloadTextFile(localOverview.script, `${documentId}-audio-overview.txt`)
                }
              >
                <Download className="h-4 w-4" aria-hidden />
                Download script
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={generating}
                onClick={() => void handleGenerate()}
              >
                Regenerate
              </Button>
            </div>
            {localOverview.estimated_minutes ? (
              <p className="text-xs text-muted-foreground">
                ~{localOverview.estimated_minutes} min listen
                {localOverview.cached ? " · cached" : ""}
              </p>
            ) : null}
            <div className="max-h-48 overflow-y-auto rounded-md border bg-muted/20 p-3 text-sm leading-relaxed text-muted-foreground">
              {localOverview.script}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
