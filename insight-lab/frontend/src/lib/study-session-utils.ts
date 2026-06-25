import type { StudySessionRecord } from "@/lib/api";

const QUIZ_STEP_TYPES = new Set(["set_quiz", "adaptive_quiz"]);

/** Step id to attach when submitting a set-wide quiz during a tracked session. */
export function resolveActiveQuizStepId(session: StudySessionRecord | null | undefined): string | null {
  if (!session || session.status !== "active") {
    return null;
  }

  const inProgress = session.steps.find(
    (step) => QUIZ_STEP_TYPES.has(step.step_type) && step.status === "in_progress",
  );
  if (inProgress) {
    return inProgress.id;
  }

  const current = session.steps.find(
    (step) => step.step_index === session.current_step_index && QUIZ_STEP_TYPES.has(step.step_type),
  );
  if (current && current.status !== "completed" && current.status !== "skipped") {
    return current.id;
  }

  return null;
}
