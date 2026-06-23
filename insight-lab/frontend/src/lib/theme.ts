export const THEME_STORAGE_KEY = "insightlab-theme";

export type ThemeMode = "light" | "dark";

export function getStoredTheme(): ThemeMode | null {
  if (typeof window === "undefined") {
    return null;
  }
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "dark" || stored === "light" ? stored : null;
}

export function applyTheme(mode: ThemeMode) {
  document.documentElement.classList.toggle("dark", mode === "dark");
  window.localStorage.setItem(THEME_STORAGE_KEY, mode);
}

export function initTheme() {
  const stored = getStoredTheme();
  if (stored) {
    applyTheme(stored);
    return;
  }
  if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    applyTheme("dark");
  }
}

export function toggleTheme(): ThemeMode {
  const next: ThemeMode = document.documentElement.classList.contains("dark") ? "light" : "dark";
  applyTheme(next);
  return next;
}
