import { InviteAcceptClient } from "@/components/workspace/invite-accept-client";

export default async function InvitePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  return (
    <div className="flex min-h-screen items-center justify-center bg-[hsl(var(--shell))] p-6">
      <InviteAcceptClient token={token} />
    </div>
  );
}
