"use client";

import { Suspense, useState } from "react";
import { Leaderboard, type LeaderboardData } from "@/src/index";
import { DICT, type Locale } from "@/lib/dict";
import { ThemeToggle } from "@/components/theme-toggle";
import { Logo } from "@/components/logo";
import raw from "@/data/leaderboard.json";

const data = raw as unknown as LeaderboardData;

// Rendered in a stable locale so the static HTML is deterministic.
const updated = new Date(data.generated_at).toISOString().slice(0, 16).replace("T", " ");

const THEME_LABELS: Record<Locale, { toLight: string; toDark: string }> = {
  en: { toLight: "Switch to light mode", toDark: "Switch to dark mode" },
  zh: { toLight: "切换到白天模式", toDark: "切换到暗夜模式" },
};

export default function Page() {
  const [locale, setLocale] = useState<Locale>("en");
  const copy = DICT[locale];

  return (
    <>
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-border bg-paper/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5 sm:px-8">
          <a href="https://diaugeia.ai" aria-label="Diaugeia.AI — home">
            <Logo />
          </a>
          <div className="flex items-center gap-3">
            <div className="inline-flex items-center rounded-full border border-border p-0.5 text-[0.8rem]">
              {(["en", "zh"] as const).map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => setLocale(l)}
                  aria-current={locale === l ? "true" : undefined}
                  className={`rounded-full px-2.5 py-1 leading-none transition-colors ${
                    locale === l ? "bg-ink text-paper" : "text-muted hover:text-ink"
                  }`}
                >
                  {l === "en" ? "EN" : "中文"}
                </button>
              ))}
            </div>
            <ThemeToggle labels={THEME_LABELS[locale]} />
            <a
              href="https://github.com/Diaugeia/TSFLab"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub repository"
              title="GitHub"
              className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-border text-muted transition-colors hover:border-border-strong hover:text-ink"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className="h-[1.05rem] w-[1.05rem]">
                <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58 0-.29-.01-1.04-.02-2.05-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.09 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.34-5.47-5.95 0-1.31.47-2.39 1.24-3.23-.13-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.24 1.92 1.24 3.23 0 4.62-2.81 5.64-5.49 5.94.43.37.82 1.1.82 2.22 0 1.6-.02 2.89-.02 3.29 0 .32.22.7.83.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z" />
              </svg>
            </a>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="border-b border-border">
        <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
          <h1 className="font-serif text-4xl tracking-[-0.02em] text-ink sm:text-6xl">{copy.title}</h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">{copy.lede}</p>
          <p className="mt-6 text-sm text-faint">
            {copy.updated} <span className="text-accent">{updated} UTC</span>
            {" · "}
            <span className="text-accent">{data.n_submissions}</span> {copy.submissions}
            {" · "}
            {copy.rankedBy} {data.primary_metric.toUpperCase()}
          </p>
        </div>
      </section>

      {/* Leaderboard */}
      <section className="mx-auto w-full max-w-6xl px-5 py-12 sm:px-8 sm:py-16">
        <Suspense fallback={null}>
          <Leaderboard data={data} copy={copy} locale={locale} />
        </Suspense>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-8 text-sm text-faint sm:px-8">
          <span>
            διαύγεια · open infrastructure for AI research
          </span>
          <span className="flex gap-4">
            <a className="hover:text-ink" href="https://github.com/Diaugeia/TSFLab">GitHub</a>
            <a className="hover:text-ink" href="https://github.com/Diaugeia/TSFLab/blob/main/apps/web/SUBMITTING.md">Submit</a>
            <a className="hover:text-ink" href="https://huggingface.co/datasets/Diaugeia/TSFLab-Datasets">Datasets</a>
          </span>
        </div>
      </footer>
    </>
  );
}
