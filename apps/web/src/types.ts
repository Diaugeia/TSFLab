// Localized copy the leaderboard UI needs. Structurally identical to the host
// site's `Dictionary["tsflab"]`, so the host can pass `dict.tsflab` directly
// without this package depending on the site's dictionary module.
//
// Metric abbreviations (MSE/MAE/Sharpe/Win Rate…) are intentionally NOT here —
// they render as fixed English labels in the components.
import type { RollingCopy } from "./rolling-rounds";

export interface LeaderboardDict {
  rolling?: RollingCopy;
  title: string;
  lede: string;
  updated: string;
  submissions: string;
  rankedBy: string;
  horizon: string;
  emptyTrack: string;
  // Shown when the whole board has no rows yet (e.g. before the first release results).
  emptyBoard?: string;
  categories?: {
    commonStatic: string;
    realtime: string;
  };
  tracks: Record<string, string>;
  cols: {
    model: string;
    mse: string;
    mae: string;
    rmse: string;
    corr: string;
    fit_time: string;
    inference_time: string;
    runs: string;
  };
  searchPlaceholder: string;
  showing: string;
  noMatch: string;
  quant: {
    regression: string;
    quant: string;
    modelType: string;
    strategy: string;
    pollutant: string;
    baselineTag: string;
    buyAndHold: string;
    lastValueCopy: string;
    configs: { conservative: string; balanced: string; aggressive: string };
    rankTip: string;
    catRankTip: string;
  };
  viz: {
    title: string;
    logReturn: string;
    cumulativeReturn: string;
    configuration: string;
    selectModels: string;
    maxSelected: string;
    searchModels: string;
    addModel: string;
    showAll: string;
    showLess: string;
    groundTruth: string;
    buyHold: string;
    // Strategy P&L (Quant view) + Prediction-accuracy scatter (Regression view).
    cumulativeTitle: string;
    cumulativeCaption: string;
    cumulativeFootnote: string;
    scatterTitle: string;
    scatterCaption: string;
    scatterFootnote: string;
    predicted: string;
    actual: string;
    perfectLine: string;
  };
  strategy: {
    heading: string;
    intro: string;
    coreTitle: string;
    core: { label: string; value: string }[];
    scoringLabel: string;
    scoringFormula: string;
    items: { title: string; body: string }[];
    backtestPeriod: string;
    marketCondition: string;
    initialCapital: string;
    disclaimerLabel: string;
    disclaimer: string;
  };
  overview: {
    title: string;
    caption: string;
    dataset: string;
    xAxis: string;
    frontier: string;
    clickHint: string;
    scale: string;
    scaleOverview: string;
    scaleZoom: string;
  };
}
