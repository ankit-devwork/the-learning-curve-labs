"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, type ClassroomAnalytics } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type ClassroomAnalyticsPanelProps = {
  setId: string;
  accessToken: string | null;
  canManage: boolean;
  embedded?: boolean;
};

function formatPercent(value: number | null | undefined): string {
  return value != null ? `${value}%` : "—";
}

function formatDate(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  return new Date(value).toLocaleString();
}

export function ClassroomAnalyticsPanel({
  setId,
  accessToken,
  canManage,
  embedded = false,
}: ClassroomAnalyticsPanelProps) {
  const [analytics, setAnalytics] = useState<ClassroomAnalytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!accessToken || !canManage) {
      return;
    }
    setLoading(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/classroom/analytics`, accessToken);
    setLoading(false);
    if (!response.ok) {
      setError("Could not load classroom analytics.");
      return;
    }
    setAnalytics((await response.json()) as ClassroomAnalytics);
  }, [accessToken, canManage, setId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (!canManage) {
    return null;
  }

  if (loading && !analytics) {
    return <p className="text-sm text-muted-foreground">Loading classroom analytics…</p>;
  }

  if (error) {
    return <p className="text-sm text-destructive">{error}</p>;
  }

  if (!analytics) {
    return null;
  }

  const content = (
    <div className="space-y-4" data-tour={embedded ? "classroom-analytics" : undefined}>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <div className="rounded-lg border bg-muted/20 px-3 py-2">
          <dt className="text-xs text-muted-foreground">Members</dt>
          <dd className="mt-0.5 text-lg font-semibold">{analytics.member_count}</dd>
        </div>
        <div className="rounded-lg border bg-muted/20 px-3 py-2">
          <dt className="text-xs text-muted-foreground">Class avg quiz</dt>
          <dd className="mt-0.5 text-lg font-semibold">
            {formatPercent(analytics.class_avg_quiz_percent)}
          </dd>
        </div>
        <div className="rounded-lg border bg-muted/20 px-3 py-2">
          <dt className="text-xs text-muted-foreground">Public quiz attempts</dt>
          <dd className="mt-0.5 text-lg font-semibold">{analytics.public_quiz_attempts}</dd>
        </div>
        <div className="rounded-lg border bg-muted/20 px-3 py-2">
          <dt className="text-xs text-muted-foreground">Public quiz avg</dt>
          <dd className="mt-0.5 text-lg font-semibold">
            {formatPercent(analytics.public_quiz_avg_percent)}
          </dd>
        </div>
      </dl>

      <div className="overflow-x-auto rounded-xl border">
        <table className="min-w-full text-sm">
          <thead className="border-b bg-muted/30 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">Member</th>
              <th className="px-3 py-2 font-medium">Role</th>
              <th className="px-3 py-2 font-medium">Quizzes</th>
              <th className="px-3 py-2 font-medium">Avg score</th>
              <th className="px-3 py-2 font-medium">Flashcards</th>
              <th className="px-3 py-2 font-medium">Mastery</th>
              <th className="px-3 py-2 font-medium">Weak topics</th>
              <th className="px-3 py-2 font-medium">Last active</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {analytics.members.map((member) => (
              <tr key={member.user_id}>
                <td className="px-3 py-2">
                  <p className="font-medium">{member.full_name || member.email || member.user_id}</p>
                  {member.email ? (
                    <p className="text-xs text-muted-foreground">{member.email}</p>
                  ) : null}
                </td>
                <td className="px-3 py-2 capitalize text-muted-foreground">{member.role}</td>
                <td className="px-3 py-2">{member.quiz_attempts}</td>
                <td className="px-3 py-2">{formatPercent(member.avg_quiz_percent)}</td>
                <td className="px-3 py-2">
                  {member.flashcard_reviews}
                  {member.flashcard_knew_percent != null ? (
                    <span className="text-xs text-muted-foreground">
                      {" "}
                      · {member.flashcard_knew_percent}% knew
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-2">{formatPercent(member.mastery_avg_percent)}</td>
                <td className="px-3 py-2">{member.weak_topic_count}</td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {formatDate(member.last_activity_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {analytics.members.length === 0 ? (
        <p className="text-sm text-muted-foreground">No members yet — invite classmates to see progress here.</p>
      ) : null}
    </div>
  );

  if (embedded) {
    return content;
  }

  return (
    <Card className="shadow-sm" data-tour="classroom-analytics">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Classroom analytics</CardTitle>
        <CardDescription>
          Per-member progress on this study sheet — quiz scores, flashcards, and weak topics.
        </CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
