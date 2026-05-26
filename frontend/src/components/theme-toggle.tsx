"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useHydration } from "@/hooks/use-hydration";
import { cn } from "@/lib/utils";

export function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme();
  const hydrated = useHydration();
  const isDark = resolvedTheme === "dark";

  if (!hydrated) {
    return <div className="h-6 w-11 shrink-0 rounded-full bg-muted" />;
  }

  return (
    <div className="flex items-center gap-2">
      <Sun
        className={cn(
          "h-4 w-4 transition-colors",
          isDark ? "text-muted-foreground" : "text-foreground",
        )}
        aria-hidden
      />
      <button
        type="button"
        role="switch"
        aria-checked={isDark}
        aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        onClick={() => setTheme(isDark ? "light" : "dark")}
        className={cn(
          "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full",
          "border-2 border-transparent transition-colors focus-visible:outline-none",
          "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          "focus-visible:ring-offset-background",
          isDark ? "bg-primary" : "bg-muted",
        )}
      >
        <span
          className={cn(
            "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-md",
            "ring-0 transition-transform",
            isDark ? "translate-x-5" : "translate-x-0",
          )}
        />
      </button>
      <Moon
        className={cn(
          "h-4 w-4 transition-colors",
          isDark ? "text-foreground" : "text-muted-foreground",
        )}
        aria-hidden
      />
    </div>
  );
}
