"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type MeResponse = {
  user_id: string;
  email: string | null;
  role: string | null;
  authenticated: boolean;
  correlation_id?: string;
  observability?: unknown;
};

export function BackendMeCard() {
  const [data, setData] = useState<MeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadMe() {
      try {
        const supabase = createClient();
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (!session?.access_token) {
          setError("No Supabase session token found");
          setLoading(false);
          return;
        }

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const response = await fetch(`${apiUrl}/me`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        });

        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          setError(body.error || `Backend returned ${response.status}`);
          setLoading(false);
          return;
        }

        setData(await response.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to reach backend");
      } finally {
        setLoading(false);
      }
    }

    loadMe();
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Backend connection</CardTitle>
        <CardDescription>
          JWT verified by FastAPI using pycorekit tracing (`GET /me`)
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading && <p className="text-sm text-muted-foreground">Checking backend...</p>}
        {error && <p className="text-sm text-destructive">{error}</p>}
        {data && (
          <dl className="space-y-2 text-sm">
            <div>
              <dt className="font-medium">User ID</dt>
              <dd className="break-all text-muted-foreground">{data.user_id}</dd>
            </div>
            <div>
              <dt className="font-medium">Email</dt>
              <dd className="text-muted-foreground">{data.email}</dd>
            </div>
            <div>
              <dt className="font-medium">Correlation ID</dt>
              <dd className="break-all text-muted-foreground">{data.correlation_id}</dd>
            </div>
          </dl>
        )}
      </CardContent>
    </Card>
  );
}
