export const ONBOARDING_VERSION = "phase8-sharing-lifecycle-v1";
export const ONBOARDING_STORAGE_KEY = "insightlab_onboarding_done";

export function isOnboardingComplete(): boolean {
  if (typeof window === "undefined") {
    return true;
  }
  return window.localStorage.getItem(ONBOARDING_STORAGE_KEY) === ONBOARDING_VERSION;
}

export function markOnboardingComplete(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(ONBOARDING_STORAGE_KEY, ONBOARDING_VERSION);
}

export function resetOnboarding(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(ONBOARDING_STORAGE_KEY);
}

export const TOUR_RESTART_EVENT = "insightlab:restart-tour";

export function requestTourRestart(): void {
  if (typeof window === "undefined") {
    return;
  }
  resetOnboarding();
  window.dispatchEvent(new CustomEvent(TOUR_RESTART_EVENT));
}
