import { PublicQuizClient } from "@/components/workspace/public-quiz-client";

type PublicQuizPageProps = {
  params: Promise<{ token: string }>;
};

export default async function PublicQuizPage({ params }: PublicQuizPageProps) {
  const { token } = await params;
  return (
    <div className="mx-auto max-w-2xl space-y-4 p-6">
      <h1 className="font-display text-2xl font-semibold">InsightLab Quiz</h1>
      <PublicQuizClient shareToken={token} />
    </div>
  );
}
