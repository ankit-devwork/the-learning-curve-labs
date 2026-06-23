"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { getStoredTheme, toggleTheme, type ThemeMode } from "@/lib/theme";

export function ThemeToggle({ className }: { className?: string }) {
  const [mode, setMode] = useState<ThemeMode>("light");

  useEffect(() => {
    const stored = getStoredTheme();
    if (stored) {
      setMode(stored);
      return;
    }
    setMode(document.documentElement.classList.contains("dark") ? "dark" : "light");
  }, []);

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={className}
      aria-label={mode === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => setMode(toggleTheme())}
    >
      {mode === "dark" ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
    </Button>
  );
}
