export const ONBOARDING_STORAGE_KEY = "insightlab_onboarding_done";

export function isOnboardingComplete(): boolean {
  if (typeof window === "undefined") {
    return true;
  }
  return window.localStorage.getItem(ONBOARDING_STORAGE_KEY) === "true";
}

export function markOnboardingComplete(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(ONBOARDING_STORAGE_KEY, "true");
}

export function resetOnboarding(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(ONBOARDING_STORAGE_KEY);
}
