"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import type { LeaderboardDict } from "./types";
import type { LeaderboardData } from "./lib/leaderboard-data";
import { Seg } from "./ui/seg";
import { datasetRank } from "./lib/datasets";
import { EvolutionChart } from "./evolution-chart";
import { DatasetCard } from "./dataset-card";
import { RollingRounds } from "./rolling-rounds";

// Re-exports for back-compat with existing importers (src/index.ts, etc.).
export { Seg } from "./ui/seg";
export type { LeaderboardData, LeaderRow } from "./lib/leaderboard-data";

const STATIC_TRACKS = ["time_series", "spatiotemporal", "covariate"];
const REALTIME_TRACKS = ["stock", "traffic", "air_quality"];

export function Leaderboard({ data, copy, locale = "en" }: { data: LeaderboardData; copy: LeaderboardDict; locale?: "en" | "zh" }) {
  // Visualization data (stock trends) is fetched client-side from /public.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [visualizationData, setVisualizationData] = useState<any>(null);
  const hasStock = !!data.tracks?.["stock"];
  useEffect(() => {
    if (!hasStock) return; // the quant charts are the only reader
    fetch("/visualization_data.json")
      .then((res) => res.json())
      .then(setVisualizationData)
      .catch((err) => console.error("Failed to load visualization data:", err));
  }, [hasStock]);

  // True when no track has a single ranked row (the board before the first results).
  const boardEmpty = !Object.values(data.tracks ?? {}).some((block) =>
    Object.values(block?.datasets ?? {}).some(
      (ds) => Object.values(ds?.horizons ?? {}).some((rows) => rows.length > 0) ||
        Object.values(ds?.quant ?? {}).some((rows) => rows.length > 0),
    ),
  );

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const trackHasData = (t: string) =>
    !!(data.tracks?.[t] && Object.keys(data.tracks[t]?.datasets ?? {}).length > 0);

  // View state is restored from the URL query so it survives a language switch /
  // refresh and is deep-linkable (?cat=realtime&track=air_quality&view=quant).
  const initialCategory: "static" | "realtime" =
    searchParams.get("cat") === "realtime" ? "realtime" : "static";
  const [category, setCategory] = useState<"static" | "realtime">(initialCategory);
  const [query, setQuery] = useState("");

  const stockDatasets = data.tracks["stock"] ? Object.keys(data.tracks["stock"].datasets) : [];
  const [selectedStockDataset, setSelectedStockDataset] = useState(() => {
    const ds = searchParams.get("ds");
    return ds && stockDatasets.includes(ds) ? ds : stockDatasets[0] || "Stock-HS300";
  });

  const [view, setView] = useState<"regression" | "quant">(
    searchParams.get("view") === "regression" ? "regression" : "quant",
  );

  const [quantConfig, setQuantConfig] = useState<"conservative" | "balanced" | "aggressive">("conservative");

  const [track, setTrack] = useState(() => {
    const tracks = initialCategory === "static" ? STATIC_TRACKS : REALTIME_TRACKS;
    const t = searchParams.get("track");
    if (t && tracks.includes(t) && trackHasData(t)) return t;
    return tracks.find(trackHasData) ?? tracks[0];
  });

  // Mirror state back into the URL (replace = no history spam, scroll preserved).
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("cat", category);
    params.set("track", track);
    if (track === "stock") params.set("view", view);
    else params.delete("view");
    if (track === "stock" && stockDatasets.length > 1) params.set("ds", selectedStockDataset);
    else params.delete("ds");
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [category, track, view, selectedStockDataset]);

  // Populated tracks first so empty ones (e.g. traffic) sink to the end.
  const currentTracks = (category === "static" ? STATIC_TRACKS : REALTIME_TRACKS)
    .slice()
    .sort((a, b) => Number(trackHasData(b)) - Number(trackHasData(a)));

  const current = data.tracks[track];
  const datasets = current
    ? Object.entries(current.datasets).sort(([a], [b]) => datasetRank(a) - datasetRank(b) || a.localeCompare(b))
    : [];

  const pickTrack = (cat: "static" | "realtime") => {
    const tracks = cat === "static" ? STATIC_TRACKS : REALTIME_TRACKS;
    return tracks.find((t) => data.tracks[t]) ?? tracks[0];
  };

  return (
    <>
      {/* Primary category */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Seg size="lg" active={category === "static"} onClick={() => { setCategory("static"); setTrack(pickTrack("static")); }}>
          {copy.categories?.commonStatic ?? "Common Static Dataset"}
        </Seg>
        <Seg size="lg" active={category === "realtime"} onClick={() => { setCategory("realtime"); setTrack(pickTrack("realtime")); }}>
          {copy.categories?.realtime ?? "Real-Time Dataset"}
        </Seg>
      </div>

      {/* Track tabs */}
      <div className="flex flex-wrap items-center gap-2">
        {currentTracks.map((t) => (
          <Seg key={t} size="md" active={t === track} disabled={!trackHasData(t)} onClick={() => setTrack(t)}>
            {copy.tracks[t] ?? t}
          </Seg>
        ))}
      </div>

      {/* Dataset selector for stock — only when there's a real choice */}
      {track === "stock" && stockDatasets.length > 1 && (
        <div className="mt-5 flex items-center gap-2">
          {stockDatasets.map((d) => (
            <Seg key={d} size="md" active={selectedStockDataset === d} onClick={() => setSelectedStockDataset(d)}>
              {d}
            </Seg>
          ))}
        </div>
      )}

      {/* Model filter + view toggle */}
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={copy.searchPlaceholder}
          aria-label={copy.searchPlaceholder}
          className="w-full max-w-sm rounded-lg border border-border bg-paper-2 px-3.5 py-2 text-sm text-ink placeholder:text-faint focus:border-accent focus:outline-none"
        />
        {track === "stock" && (
          <div className="flex items-center gap-1.5">
            <Seg size="md" active={view === "quant"} onClick={() => setView("quant")}>
              {copy.quant.quant}
            </Seg>
            <Seg size="md" active={view === "regression"} onClick={() => setView("regression")}>
              {copy.quant.regression}
            </Seg>
          </div>
        )}
      </div>

      {/* Method evolution chart at the top of the static section */}
      {category === "static" && current && datasets.length > 0 && (
        <div className="mt-6">
          <EvolutionChart datasets={current.datasets} copy={copy} />
        </div>
      )}

      {datasets.length === 0 ? (
        <p className="mt-10 rounded-2xl border border-border bg-surface px-6 py-16 text-center text-muted">
          {boardEmpty && copy.emptyBoard ? copy.emptyBoard : copy.emptyTrack}
        </p>
      ) : track === "stock" ? (
        <div className="mt-6">
          {current?.datasets[selectedStockDataset] && (
            <DatasetCard
              track={track}
              name={selectedStockDataset}
              ds={current.datasets[selectedStockDataset]}
              copy={copy}
              query={query}
              hideTitle
              view={view}
              quantConfig={quantConfig}
              onQuantConfigChange={setQuantConfig}
              locale={locale}
              visualizationData={visualizationData}
            />
          )}
        </div>
      ) : (
        <div className="mt-6 space-y-6">
          {datasets.map(([name, ds]) => (
            <DatasetCard key={`${track}/${name}`} track={track} name={name} ds={ds} copy={copy} query={query} locale={locale} />
          ))}
        </div>
      )}

      {category === "realtime" && copy.rolling && (
        <RollingRounds data={data.realtime ?? {}} tracks={data.realtime_tracks} copy={copy.rolling} domainLabels={copy.tracks} />
      )}
    </>
  );
}
