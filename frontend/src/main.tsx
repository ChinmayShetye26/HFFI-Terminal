import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  analyzePortfolio,
  fetchAssets,
  fetchChart,
  fetchMarket,
  runBacktest,
} from "./api";
import { clamp, money, number, percent, signedNumber, signedPercent } from "./format";
import type {
  AnalysisResponse,
  Asset,
  AssetCategory,
  BacktestCurveRow,
  BacktestResponse,
  ChartRow,
  HoldingInput,
  HouseholdInput,
  MarketSnapshotRow,
} from "./types";
import "./styles.css";

const DEFAULT_HOUSEHOLD: HouseholdInput = {
  monthlyIncome: 6000,
  monthlyEssentialExpenses: 3500,
  monthlyTotalExpenses: 4500,
  liquidSavings: 8000,
  totalDebt: 22000,
  monthlyDebtPayment: 800,
  portfolioVolatility: 0.16,
  expectedDrawdown: 0.2,
  rateSensitivity: 0.5,
  employmentType: "full_time",
  dependents: 0,
  portfolioWeights: { equity: 0.6, bond: 0.3, cash: 0.1 },
};

const DEFAULT_HOLDINGS: HoldingInput[] = [
  { id: crypto.randomUUID(), category: "equity", ticker: "AAPL", name: "Apple", units: 10, buyPrice: 150 },
  { id: crypto.randomUUID(), category: "bond", ticker: "AGG", name: "US Aggregate Bonds", units: 20, buyPrice: 95 },
];

type AllocationKey = "equity" | "bond" | "cash";

type AllocationBreakdown = {
  amounts: Record<AllocationKey, number>;
  weights: Record<AllocationKey, number>;
  total: number;
};

const ALLOCATION_KEYS: AllocationKey[] = ["equity", "bond", "cash"];

const ALLOCATION_LABELS: Record<AllocationKey, string> = {
  equity: "Equity",
  bond: "Bond",
  cash: "Cash",
};

const ALLOCATION_COLORS: Record<AllocationKey, string> = {
  equity: "var(--cyan)",
  bond: "var(--amber)",
  cash: "var(--green)",
};

const PERIOD_PRESETS = [
  { value: "1d", label: "1D" },
  { value: "5d", label: "5D" },
  { value: "1mo", label: "1M" },
  { value: "3mo", label: "3M" },
  { value: "6mo", label: "6M" },
  { value: "1y", label: "1Y" },
  { value: "2y", label: "2Y" },
  { value: "5y", label: "5Y" },
  { value: "10y", label: "10Y" },
  { value: "max", label: "All" },
  { value: "custom", label: "Custom" },
] as const;

const INTERVAL_PRESETS = [
  { value: "1m", label: "1 min" },
  { value: "5m", label: "5 min" },
  { value: "15m", label: "15 min" },
  { value: "30m", label: "30 min" },
  { value: "60m", label: "60 min" },
  { value: "1h", label: "1 hour" },
  { value: "1d", label: "1 day" },
  { value: "1wk", label: "1 week" },
  { value: "1mo", label: "1 month" },
  { value: "custom", label: "Custom" },
] as const;

const TIMEFRAMES = [
  { key: "6mo-1d", label: "6M / 1 day", period: "6mo", interval: "1d" },
] as const;

type Tab = "fragility" | "portfolio" | "markets" | "backtest" | "evidence" | "scenarios" | "model";

function App() {
  const [assets, setAssets] = useState<Record<AssetCategory, Asset[]> | null>(null);
  const [household, setHousehold] = useState<HouseholdInput>(DEFAULT_HOUSEHOLD);
  const [holdings, setHoldings] = useState<HoldingInput[]>(DEFAULT_HOLDINGS);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("fragility");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const allocation = useMemo(
    () => calculatePortfolioAllocation(holdings, household.liquidSavings),
    [holdings, household.liquidSavings],
  );

  useEffect(() => {
    fetchAssets()
      .then((payload) => setAssets(payload.assets))
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => {
      void runAnalysis(false);
    }, 650);
    return () => window.clearTimeout(id);
  }, [household, holdings]);

  async function runAnalysis(showLoading = true) {
    try {
      if (showLoading) setLoading(true);
      const payloadHousehold = { ...household, portfolioWeights: allocation.weights };
      const payload = await analyzePortfolio({ household: payloadHousehold, holdings });
      setAnalysis(payload);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  const categories = useMemo(() => Object.keys(assets ?? {}) as AssetCategory[], [assets]);

  return (
    <div className="terminal-shell">
      <TopBar loading={loading} analysis={analysis} onAnalyze={() => void runAnalysis(true)} />
      <div className="terminal-grid">
        <aside className="left-rail">
          <ControlDeck household={household} onChange={setHousehold} />
          <QuickAllocation analysis={analysis} allocation={allocation} />
          <MacroPanel analysis={analysis} />
        </aside>

        <main className="workspace">
          {error ? <div className="alert">{error}</div> : null}
          <Nav active={activeTab} onChange={setActiveTab} />
          {activeTab === "fragility" && <FragilityPanel analysis={analysis} />}
          {activeTab === "portfolio" && (
            <PortfolioBuilder
              assets={assets}
              holdings={holdings}
              onChange={setHoldings}
              analysis={analysis}
            />
          )}
          {activeTab === "markets" && <Markets assets={assets} categories={categories} />}
          {activeTab === "backtest" && <Backtesting household={household} holdings={holdings} />}
          {activeTab === "evidence" && <Evidence analysis={analysis} />}
          {activeTab === "scenarios" && <Scenarios analysis={analysis} />}
          {activeTab === "model" && <ModelPanel analysis={analysis} />}
        </main>

        <aside className="right-rail">
          <RecommendationStack analysis={analysis} />
          <ActionStack analysis={analysis} />
        </aside>
      </div>
    </div>
  );
}

function TopBar({
  loading,
  analysis,
  onAnalyze,
}: {
  loading: boolean;
  analysis: AnalysisResponse | null;
  onAnalyze: () => void;
}) {
  return (
    <header className="topbar">
      <div>
        <div className="brand">HFFI INVESTMENT TERMINAL</div>
      </div>
      <div className="topbar-tape">
        <TapeItem label="HFFI" value={analysis ? number(analysis.hffi.score, 1) : "--"} tone={scoreTone(analysis?.hffi.score)} />
        <TapeItem label="BAND" value={analysis?.hffi.band ?? "--"} />
        <TapeItem label="REGIME" value={analysis?.riskOff ? "RISK OFF" : "RISK ON"} tone={analysis?.riskOff ? "warn" : "good"} />
        <TapeItem label="SEGMENT" value={analysis?.segment ?? "--"} />
      </div>
      <button className="primary-button" onClick={onAnalyze} disabled={loading}>
        {loading ? "Analyzing" : "Run Analysis"}
      </button>
    </header>
  );
}

function TapeItem({ label, value, tone = "neutral" }: { label: string; value: string; tone?: string }) {
  return (
    <div className={`tape-item ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ControlDeck({
  household,
  onChange,
}: {
  household: HouseholdInput;
  onChange: (value: HouseholdInput) => void;
}) {
  const update = (key: keyof HouseholdInput, value: number) => onChange({ ...household, [key]: value });
  const updateText = (key: keyof HouseholdInput, value: string) => onChange({ ...household, [key]: value });

  return (
    <section className="panel">
      <PanelTitle title="Household Inputs" caption="Live analysis updates after edits" />
      <NumberField label="Monthly income" value={household.monthlyIncome} step={100} onChange={(v) => update("monthlyIncome", v)} />
      <NumberField label="Total expenses" value={household.monthlyTotalExpenses} step={100} onChange={(v) => update("monthlyTotalExpenses", v)} />
      <NumberField label="Essential expenses" value={household.monthlyEssentialExpenses} step={100} onChange={(v) => update("monthlyEssentialExpenses", v)} />
      <NumberField label="Liquid savings" value={household.liquidSavings} step={500} onChange={(v) => update("liquidSavings", v)} />
      <NumberField label="Total debt" value={household.totalDebt} step={500} onChange={(v) => update("totalDebt", v)} />
      <NumberField label="Debt payment" value={household.monthlyDebtPayment} step={50} onChange={(v) => update("monthlyDebtPayment", v)} />
      <label className="field">
        <span>Employment type</span>
        <select value={household.employmentType} onChange={(e) => updateText("employmentType", e.target.value)}>
          <option value="full_time">Full time / salaried</option>
          <option value="part_time">Part time</option>
          <option value="contract">Contract</option>
          <option value="self_employed">Self employed</option>
          <option value="unemployed">Unemployed</option>
          <option value="retired">Retired</option>
        </select>
      </label>
      <NumberField label="Dependents" value={household.dependents} step={1} onChange={(v) => update("dependents", Math.max(0, Math.round(v)))} />
      <SliderField label="Portfolio volatility" value={household.portfolioVolatility} max={0.5} step={0.01} onChange={(v) => update("portfolioVolatility", v)} />
      <SliderField label="Expected drawdown" value={household.expectedDrawdown} max={0.8} step={0.01} onChange={(v) => update("expectedDrawdown", v)} />
      <SliderField label="Rate sensitivity" value={household.rateSensitivity} max={1} step={0.05} onChange={(v) => update("rateSensitivity", v)} />
      <div className="input-divider">Allocation auto-calculates from holdings and Liquid Savings</div>
    </section>
  );
}

function NumberField({
  label,
  value,
  step,
  onChange,
}: {
  label: string;
  value: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min="0" step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function SliderField({
  label,
  value,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider-field">
      <span>
        {label}
        <strong>{percent(value, 0)}</strong>
      </span>
      <input type="range" min="0" max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function QuickAllocation({
  analysis,
  allocation,
}: {
  analysis: AnalysisResponse | null;
  allocation: AllocationBreakdown;
}) {
  const target = analysis?.targetAllocation ?? {};
  return (
    <section className="panel allocation-monitor">
      <PanelTitle title="Allocation Monitor" caption="Auto: Units x Buy Price plus Liquid Savings" />
      <AllocationDonut allocation={allocation} target={target} />
    </section>
  );
}

function AllocationDonut({
  allocation,
  target,
  compact = false,
}: {
  allocation: AllocationBreakdown;
  target?: Record<string, number>;
  compact?: boolean;
}) {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const rawTotal = ALLOCATION_KEYS.reduce((sum, key) => sum + Math.max(finiteNumber(allocation.weights[key]), 0), 0);
  let cursor = 0;
  const slices = ALLOCATION_KEYS
    .map((key) => {
      const value = rawTotal > 0 ? Math.max(finiteNumber(allocation.weights[key]), 0) / rawTotal : 0;
      const start = cursor;
      cursor += value;
      const midpoint = start + value / 2;
      const angle = (-90 + midpoint * 360) * Math.PI / 180;
      const labelRadius = value < 0.08 ? 86 : 70;
      return {
        key,
        value,
        length: value * circumference,
        offset: start * circumference,
        labelX: clamp(110 + labelRadius * Math.cos(angle), 36, 184),
        labelY: clamp(110 + labelRadius * Math.sin(angle), 24, 196),
        label: `${key === "equity" ? "EQ" : key === "bond" ? "BD" : "CS"} ${percent(value, 0)}`,
      };
    })
    .filter((slice) => slice.value > 0.001);
  const showTarget = ALLOCATION_KEYS.some((key) => Number.isFinite(finiteNumber(target?.[key], NaN)));

  return (
    <div className={`allocation-donut-layout ${compact ? "compact" : ""}`}>
      <div className="allocation-donut-stage">
        <svg className="allocation-donut" viewBox="0 0 220 220" role="img" aria-label="Portfolio allocation donut chart">
          <circle className="donut-ring-bg" cx="110" cy="110" r={radius} />
          {slices.map((slice) => (
            <circle
              className={`donut-slice ${slice.key}`}
              key={slice.key}
              cx="110"
              cy="110"
              r={radius}
              style={{ stroke: ALLOCATION_COLORS[slice.key] }}
              strokeDasharray={`${slice.length} ${circumference - slice.length}`}
              strokeDashoffset={-slice.offset}
              transform="rotate(-90 110 110)"
            />
          ))}
          <circle className="donut-center" cx="110" cy="110" r="45" />
          <text className="donut-center-label" x="110" y="104" textAnchor="middle">Total</text>
          <text className="donut-center-value" x="110" y="123" textAnchor="middle">{allocation.total > 0 ? money(allocation.total, 0) : "n/a"}</text>
          {slices.map((slice) => (
            <g key={`${slice.key}-label`}>
              <rect className="donut-slice-label-bg" x={slice.labelX - 29} y={slice.labelY - 11} width="58" height="22" rx="4" />
              <text
                className="donut-slice-label"
                x={slice.labelX}
                y={slice.labelY}
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {slice.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className={`allocation-table ${showTarget ? "with-target" : ""}`}>
        <div className="allocation-table-head">
          <span>Asset</span><span>Amount</span><span>Percentage</span>{showTarget ? <span>Target</span> : null}
        </div>
        {ALLOCATION_KEYS.map((key) => {
          const targetValue = finiteNumber(target?.[key], NaN);
          return (
            <div className="allocation-table-row" key={key}>
              <span><i className={`allocation-swatch ${key}`} />{ALLOCATION_LABELS[key]}</span>
              <strong>{money(allocation.amounts[key], 0)}</strong>
              <em>{percent(allocation.weights[key] ?? 0)}</em>
              {showTarget ? <small>{Number.isFinite(targetValue) ? percent(targetValue) : "n/a"}</small> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}


function MacroPanel({ analysis }: { analysis: AnalysisResponse | null }) {
  return (
    <section className="panel">
      <PanelTitle title="Live Macro Inputs" caption="Used by HFFI, priority actions, feature evidence, and trade recommendations" />
      {analysis ? <MacroDataGrid analysis={analysis} /> : <div className="macro-source">Run analysis to load live/fallback macro percentages.</div>}
    </section>
  );
}


function MacroDataGrid({ analysis }: { analysis: AnalysisResponse }) {
  const macro = analysis.macro ?? {};
  const macroRows = [
    ["Inflation", percent(Number(macro.inflation_rate ?? 0), 2)],
    ["Fed Funds", percent(Number(macro.fed_funds_rate ?? 0), 2)],
    ["Unemployment", percent(Number(macro.unemployment_rate ?? 0), 2)],
    ["Mortgage", percent(Number(macro.mortgage_rate ?? 0), 2)],
    ["10Y Treasury", percent(Number(macro.treasury_10y ?? 0), 2)],
    ["2Y Treasury", percent(Number(macro.treasury_2y ?? 0), 2)],
    ["Yield Spread", signedPercent(Number(macro.yield_curve_spread ?? 0), 2)],
    ["VIX", number(Number(macro.vix ?? 0), 1)],
  ];
  return (
    <>
      <div className="macro-grid">
        {macroRows.map(([label, value]) => (
          <div className="macro-cell" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="macro-source">
        Source: {String(macro.timestamp ?? "live/fallback macro snapshot")} | Macro feeds HFFI M-score, regime, priority actions, feature evidence, and trade suitability.
      </div>
    </>
  );
}


function Nav({ active, onChange }: { active: Tab; onChange: (tab: Tab) => void }) {
  const tabs: Array<[Tab, string]> = [
    ["fragility", "Fragility"],
    ["portfolio", "Portfolio"],
    ["markets", "Markets"],
    ["backtest", "Backtesting"],
    ["evidence", "Evidence"],
    ["scenarios", "Scenarios"],
    ["model", "Model"],
  ];
  return (
    <nav className="tabs">
      {tabs.map(([key, label]) => (
        <button key={key} className={active === key ? "active" : ""} onClick={() => onChange(key)}>
          {label}
        </button>
      ))}
    </nav>
  );
}

function FragilityPanel({ analysis }: { analysis: AnalysisResponse | null }) {
  if (!analysis) return <EmptyState title="Waiting for analysis" detail="Edit inputs or run analysis to populate the terminal." />;
  const components = Object.entries(analysis.hffi.components);
  return (
    <div className="fragility-layout">
      <section className="panel hero-panel">
        <PanelTitle title="Fragility" caption="HFFI, distress probability, macro regime, and component drivers" />
        <div className="hero-grid">
          <Gauge score={analysis.hffi.score} band={analysis.hffi.band} />
          <div className="metric-stack">
            <Metric label="Distress probability" value={percent(analysis.hffi.distressProbability)} />
            <Metric label="Household segment" value={analysis.segment} />
            <Metric label="Best market" value={String(analysis.selectedMarket?.market ?? "n/a")} />
            <Metric label="Recommendation score" value={number(Number(analysis.selectedMarket?.recommendation_score ?? 0), 3)} />
          </div>
        </div>
      </section>
      <section className="panel">
        <PanelTitle title="Component Heatmap" caption="Normalized 0 to 1 fragility drivers" />
        <div className="component-grid">
          {components.map(([key, value]) => (
            <div className="component-tile" key={key}>
              <span>{key}</span>
              <strong>{number(value, 3)}</strong>
              <div className="bar-track">
                <div className={`bar-fill ${value > 0.65 ? "bad" : value > 0.4 ? "warn" : "good"}`} style={{ width: `${value * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="panel wide macro-focus">
        <PanelTitle title="Live Macro Data Percentages" caption="These live/fallback macro inputs directly affect HFFI and recommendations" />
        <MacroDataGrid analysis={analysis} />
      </section>
      <section className="panel wide">
        <PanelTitle title="Priority Actions" caption="Rule-based HFFI interventions" />
        <div className="action-grid">
          {analysis.recommendations.map((rec) => (
            <article className="evidence-card" key={`${rec.priority}-${rec.action}`}>
              <div className="card-kicker">Priority {rec.priority} · {rec.component}</div>
              <h3>{rec.action}</h3>
              <p>{rec.detail}</p>
              <span>{rec.expectedImpact}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function Gauge({ score, band }: { score: number; band: string }) {
  const radius = 78;
  const circumference = 2 * Math.PI * radius;
  const value = clamp(score, 0, 100) / 100;
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 210 210" className="gauge">
        <circle cx="105" cy="105" r={radius} className="gauge-bg" />
        <circle
          cx="105"
          cy="105"
          r={radius}
          className={`gauge-value ${scoreTone(score)}`}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value)}
        />
        <text x="105" y="98" textAnchor="middle" className="gauge-number">{number(score, 1)}</text>
        <text x="105" y="124" textAnchor="middle" className="gauge-label">{band}</text>
      </svg>
    </div>
  );
}

function PortfolioBuilder({
  assets,
  holdings,
  onChange,
  analysis,
}: {
  assets: Record<AssetCategory, Asset[]> | null;
  holdings: HoldingInput[];
  onChange: (holdings: HoldingInput[]) => void;
  analysis: AnalysisResponse | null;
}) {
  const update = (id: string, patch: Partial<HoldingInput>) => {
    onChange(holdings.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  };
  const addRow = (category: "equity" | "bond") => {
    onChange([...holdings, { id: crypto.randomUUID(), category, ticker: "", name: "", units: 0, buyPrice: 0 }]);
  };
  const selectedTickers = new Set(holdings.map((h) => h.ticker).filter(Boolean));
  return (
    <div className="panel">
      <PanelTitle title="Interactive Portfolio Builder" caption="Search tickers, block duplicates, and update allocation instantly" />
      <div className="table-toolbar">
        <button onClick={() => addRow("equity")}>Add Equity</button>
        <button onClick={() => addRow("bond")}>Add Bond</button>
      </div>
      <div className="holdings-table">
        <div className="table-head">
          <span>#</span><span>Category</span><span>Ticker</span><span>Name</span><span>Units</span><span>Buy Price</span><span>Remove</span>
        </div>
        {holdings.map((holding, idx) => (
          <div className="table-row" key={holding.id}>
            <span>{idx + 1}</span>
            <span className={`category-badge ${holding.category}`}>{holding.category === "equity" ? "Equity" : "Bond"}</span>
            <AssetSearch
              assets={assets?.[holding.category] ?? []}
              value={holding.ticker}
              blocked={selectedTickers}
              onPick={(asset) => update(holding.id, { ticker: asset.ticker, name: asset.name })}
            />
            <input value={holding.name} readOnly />
            <input type="number" min="0" value={holding.units} onChange={(e) => update(holding.id, { units: Number(e.target.value) })} />
            <input type="number" min="0" value={holding.buyPrice} onChange={(e) => update(holding.id, { buyPrice: Number(e.target.value) })} />
            <button className="ghost-button" onClick={() => onChange(holdings.filter((h) => h.id !== holding.id))}>Remove</button>
          </div>
        ))}
      </div>
      <PanelTitle title="Holding Guidance" caption="Status, confidence inputs, and timing" />
      <div className="cards-grid">
        {(analysis?.holdingActions ?? []).map((action) => (
          <article className={`decision-card ${action.status.toLowerCase()}`} key={action.ticker}>
            <div>
              <span>{action.ticker}</span>
              <strong>{action.status}</strong>
            </div>
            <h3>{action.name}</h3>
            <p>{action.comment}</p>
            <dl>
              <dt>Invested</dt><dd>{money(action.investedAmount)}</dd>
              <dt>Portfolio</dt><dd>{number(action.allocationWeightPct, 1)}%</dd>
              <dt>Suitability</dt><dd>{action.suitability === null ? "n/a" : signedNumber(action.suitability, 3)}</dd>
            </dl>
            <footer>{action.when}</footer>
          </article>
        ))}
      </div>
    </div>
  );
}

function AssetSearch({
  assets,
  value,
  blocked,
  onPick,
}: {
  assets: Asset[];
  value: string;
  blocked: Set<string>;
  onPick: (asset: Asset) => void;
}) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  useEffect(() => setQuery(value), [value]);
  const matches = assets
    .filter((asset) => !blocked.has(asset.ticker) || asset.ticker === value)
    .filter((asset) => `${asset.ticker} ${asset.name}`.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 8);
  return (
    <div className="asset-search">
      <input
        value={query}
        placeholder="Search ticker"
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 140)}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
      />
      {open && query && matches.length > 0 ? (
        <div className="search-menu">
          {matches.map((asset) => (
            <button
              key={asset.ticker}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onPick(asset);
                setQuery(asset.ticker);
                setOpen(false);
              }}
            >
              <strong>{asset.ticker}</strong><span>{asset.name}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Markets({
  assets,
  categories,
}: {
  assets: Record<AssetCategory, Asset[]> | null;
  categories: AssetCategory[];
}) {
  const [category, setCategory] = useState<AssetCategory>("equity");
  const [market, setMarket] = useState<MarketSnapshotRow[]>([]);
  const [selected, setSelected] = useState("AAPL");
  const [tickerQuery, setTickerQuery] = useState("AAPL - Apple");
  const [tickerOpen, setTickerOpen] = useState(false);
  const [chart, setChart] = useState<ChartRow[]>([]);
  const [source, setSource] = useState("");
  const [period, setPeriod] = useState("6mo");
  const [interval, setIntervalValue] = useState("1d");
  const [timeframeKey, setTimeframeKey] = useState<(typeof TIMEFRAMES)[number]["key"]>("6mo-1d");
  const [customPeriod, setCustomPeriod] = useState("6mo");
  const [customInterval, setCustomInterval] = useState("1d");
  const [liveMoving, setLiveMoving] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [lastRefresh, setLastRefresh] = useState("");
  const periodValue = period === "custom" ? customPeriod.trim() || "6mo" : period;
  const intervalValue = interval === "custom" ? customInterval.trim() || "1d" : interval;
  const availableAssets = assets?.[category] ?? [];
  const tickerMatches = availableAssets
    .filter((asset) => {
      const needle = tickerQuery.trim().toLowerCase();
      return !needle
        || asset.ticker.toLowerCase().includes(needle)
        || asset.name.toLowerCase().includes(needle);
    })
    .slice(0, 12);

  useEffect(() => {
    fetchMarket(category).then((payload) => setMarket(payload.rows)).catch(() => setMarket([]));
    const first = assets?.[category]?.[0];
    if (first) {
      setSelected(first.fetchSymbol || first.ticker);
      setTickerQuery(`${first.ticker} - ${first.name}`);
    }
  }, [category, assets]);

  useEffect(() => {
    if (!fullscreen) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullscreen(false);
    };
    document.body.classList.add("chart-fullscreen-open");
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.classList.remove("chart-fullscreen-open");
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [fullscreen]);

  function chooseAsset(asset: Asset) {
    setSelected(asset.fetchSymbol || asset.ticker);
    setTickerQuery(`${asset.ticker} - ${asset.name}`);
    setTickerOpen(false);
  }

  async function generateChart() {
    const payload = await fetchChart(selected, periodValue, intervalValue);
    setChart(payload.rows);
    setSource(payload.dataSource);
    setLastRefresh(new Date().toLocaleTimeString());
  }

  useEffect(() => {
    if (!liveMoving || !selected) return undefined;
    const timer = window.setInterval(() => {
      void generateChart();
    }, 30000);
    return () => window.clearInterval(timer);
  }, [liveMoving, selected, periodValue, intervalValue]);

  return (
    <div className="market-layout">
      <section className="panel">
        <PanelTitle title="Market Board" caption="Live or fallback provider snapshot" />
        <div className="toolbar-grid">
          <select value={category} onChange={(e) => setCategory(e.target.value as AssetCategory)}>
            {categories.map((cat) => <option key={cat} value={cat}>{cat.toUpperCase()}</option>)}
          </select>
          <div className="market-search">
            <input
              value={tickerQuery}
              placeholder="Search ticker or company"
              onFocus={() => setTickerOpen(true)}
              onBlur={() => window.setTimeout(() => setTickerOpen(false), 140)}
              onChange={(e) => {
                const value = e.target.value;
                setTickerQuery(value);
                setTickerOpen(true);
                const exact = availableAssets.find((asset) => asset.ticker.toLowerCase() === value.trim().toLowerCase());
                if (exact) setSelected(exact.fetchSymbol || exact.ticker);
              }}
            />
            {tickerOpen && tickerMatches.length > 0 ? (
              <div className="search-menu market-search-menu">
                {tickerMatches.map((asset) => (
                  <button
                    key={asset.ticker}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => chooseAsset(asset)}
                  >
                    <strong>{asset.ticker}</strong><span>{asset.name}</span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <select value={interval} onChange={(e) => setIntervalValue(e.target.value)}>
            {INTERVAL_PRESETS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <button className="primary-button" onClick={() => void generateChart()}>Generate</button>
        </div>
        <div className="timeframe-bar">
          {PERIOD_PRESETS.map((option) => (
            <button
              key={option.value}
              className={period === option.value ? "active" : ""}
              onClick={() => setPeriod(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        {(period === "custom" || interval === "custom") ? (
          <div className="custom-timeframe">
            {period === "custom" ? (
              <label>
                <span>Custom period</span>
                <input value={customPeriod} placeholder="Examples: 10d, 18mo, 3y" onChange={(e) => setCustomPeriod(e.target.value)} />
              </label>
            ) : null}
            {interval === "custom" ? (
              <label>
                <span>Custom interval</span>
                <input value={customInterval} placeholder="Examples: 2m, 2h, 3d" onChange={(e) => setCustomInterval(e.target.value)} />
              </label>
            ) : null}
          </div>
        ) : null}
        <label className="live-toggle">
          <input type="checkbox" checked={liveMoving} onChange={(e) => setLiveMoving(e.target.checked)} />
          <span>Moving chart auto-refresh every 30 seconds</span>
        </label>
        <div className="market-board">
          {market.slice(0, 18).map((row) => (
            <button
              key={row.fetchSymbol}
              className={`market-tile ${(row.changePct ?? 0) >= 0 ? "up" : "down"}`}
              onClick={() => {
                setSelected(row.fetchSymbol);
                setTickerQuery(`${row.ticker} - ${row.name}`);
              }}
            >
              <span>{row.ticker}</span>
              <strong>{money(row.price, 2)}</strong>
              <em>{signedPercent(row.changePct, 2)}</em>
            </button>
          ))}
        </div>
      </section>
      <section className={`panel chart-panel ${fullscreen ? "fullscreen-chart" : ""}`}>
        <PanelTitle
          title="Live Market Candlestick Chart"
          caption={`${source || "Select a ticker and generate chart"}${lastRefresh ? ` - last refresh ${lastRefresh}` : ""}`}
        />
        <div className="chart-actions">
          <span>{selected} | {periodValue} | {intervalValue} | {chart.length} candles loaded</span>
          <button className="ghost-button" onClick={() => setFullscreen((value) => !value)}>
            {fullscreen ? "Exit Full Screen" : "Full Screen"}
          </button>
        </div>
        <CandlestickChart rows={chart} moving={liveMoving} />
      </section>
    </div>
  );
}

function MarketsOld({
  assets,
  categories,
}: {
  assets: Record<AssetCategory, Asset[]> | null;
  categories: AssetCategory[];
}) {
  const [category, setCategory] = useState<AssetCategory>("equity");
  const [market, setMarket] = useState<MarketSnapshotRow[]>([]);
  const [selected, setSelected] = useState("AAPL");
  const [tickerQuery, setTickerQuery] = useState("AAPL - Apple");
  const [tickerOpen, setTickerOpen] = useState(false);
  const [chart, setChart] = useState<ChartRow[]>([]);
  const [source, setSource] = useState("");
  const [period, setPeriod] = useState("6mo");
  const [interval, setIntervalValue] = useState("1d");
  const [timeframeKey, setTimeframeKey] = useState<(typeof TIMEFRAMES)[number]["key"]>("6mo-1d");
  const [customPeriod, setCustomPeriod] = useState("6mo");
  const [customInterval, setCustomInterval] = useState("1d");
  const [liveMoving, setLiveMoving] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [lastRefresh, setLastRefresh] = useState("");
  const periodValue = period === "custom" ? customPeriod.trim() || "6mo" : period;
  const intervalValue = interval === "custom" ? customInterval.trim() || "1d" : interval;
  const availableAssets = assets?.[category] ?? [];
  const tickerMatches = availableAssets
    .filter((asset) => {
      const needle = tickerQuery.trim().toLowerCase();
      return !needle
        || asset.ticker.toLowerCase().includes(needle)
        || asset.name.toLowerCase().includes(needle);
    })
    .slice(0, 12);

  useEffect(() => {
    fetchMarket(category).then((payload) => setMarket(payload.rows)).catch(() => setMarket([]));
    const first = assets?.[category]?.[0];
    if (first) {
      setSelected(first.fetchSymbol || first.ticker);
      setTickerQuery(`${first.ticker} - ${first.name}`);
    }
  }, [category, assets]);

  useEffect(() => {
    if (!fullscreen) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullscreen(false);
    };
    document.body.classList.add("chart-fullscreen-open");
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.classList.remove("chart-fullscreen-open");
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [fullscreen]);

  function chooseAsset(asset: Asset) {
    setSelected(asset.fetchSymbol || asset.ticker);
    setTickerQuery(`${asset.ticker} - ${asset.name}`);
    setTickerOpen(false);
  }

  async function generateChart() {
    const payload = await fetchChart(selected, periodValue, intervalValue);
    setChart(payload.rows);
    setSource(payload.dataSource);
    setLastRefresh(new Date().toLocaleTimeString());
  }

  useEffect(() => {
    if (!liveMoving || !selected) return undefined;
    const timer = window.setInterval(() => {
      void generateChart();
    }, 30000);
    return () => window.clearInterval(timer);
  }, [liveMoving, selected, periodValue, intervalValue]);

  return (
    <div className="market-layout">
      <section className="panel">
        <PanelTitle title="Market Board" caption="Live or fallback provider snapshot" />
        <div className="toolbar-grid">
          <select value={category} onChange={(e) => setCategory(e.target.value as AssetCategory)}>
            {categories.map((cat) => <option key={cat} value={cat}>{cat.toUpperCase()}</option>)}
          </select>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {(assets?.[category] ?? []).map((asset) => (
              <option key={asset.ticker} value={asset.fetchSymbol || asset.ticker}>{asset.ticker} · {asset.name}</option>
            ))}
          </select>
          <select value={timeframeKey} onChange={(e) => setTimeframeKey(e.target.value as (typeof TIMEFRAMES)[number]["key"])}>
            {TIMEFRAMES.map((option) => (
              <option key={option.key} value={option.key}>{option.label}</option>
            ))}
          </select>
          <button className="primary-button" onClick={() => void generateChart()}>Generate</button>
        </div>
        <label className="live-toggle">
          <input type="checkbox" checked={liveMoving} onChange={(e) => setLiveMoving(e.target.checked)} />
          <span>Moving chart auto-refresh every 30 seconds</span>
        </label>
        <div className="market-board">
          {market.slice(0, 18).map((row) => (
            <button key={row.fetchSymbol} className={`market-tile ${(row.changePct ?? 0) >= 0 ? "up" : "down"}`} onClick={() => setSelected(row.fetchSymbol)}>
              <span>{row.ticker}</span>
              <strong>{money(row.price, 2)}</strong>
              <em>{signedPercent(row.changePct, 2)}</em>
            </button>
          ))}
        </div>
      </section>
      <section className="panel chart-panel">
        <PanelTitle
          title="Live Market Candlestick Chart"
          caption={`${source || "Select a ticker and generate chart"}${lastRefresh ? ` · last refresh ${lastRefresh}` : ""}`}
        />
        <CandlestickChart rows={chart} moving={liveMoving} />
      </section>
    </div>
  );
}

function Backtesting({
  household,
  holdings,
}: {
  household: HouseholdInput;
  holdings: HoldingInput[];
}) {
  const [startDate, setStartDate] = useState("2021-01-01");
  const [endDate, setEndDate] = useState(new Date().toISOString().slice(0, 10));
  const [initialCapital, setInitialCapital] = useState(100000);
  const [frequency, setFrequency] = useState<"weekly" | "monthly" | "quarterly">("monthly");
  const [transactionCostPct, setTransactionCostPct] = useState(0.1);
  const [benchmarkTicker, setBenchmarkTicker] = useState("SPY");
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitBacktest() {
    try {
      setLoading(true);
      setError("");
      const payload = await runBacktest({
        household,
        holdings,
        startDate,
        endDate,
        initialCapital,
        frequency,
        transactionCostPct: transactionCostPct / 100,
        benchmarkTicker,
      });
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="backtest-layout">
      <section className="panel">
        <PanelTitle title="Backtesting Lab" caption="Replay HFFI recommendations against buy-and-hold and benchmark portfolios" />
        <div className="backtest-controls">
          <label className="field">
            <span>Start date</span>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </label>
          <label className="field">
            <span>End date</span>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          </label>
          <label className="field">
            <span>Initial capital</span>
            <input type="number" min="1000" step="1000" value={initialCapital} onChange={(e) => setInitialCapital(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>Rebalance frequency</span>
            <select value={frequency} onChange={(e) => setFrequency(e.target.value as "weekly" | "monthly" | "quarterly")}>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
            </select>
          </label>
          <label className="field">
            <span>Transaction cost %</span>
            <input type="number" min="0" max="5" step="0.05" value={transactionCostPct} onChange={(e) => setTransactionCostPct(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>Benchmark ticker</span>
            <input value={benchmarkTicker} onChange={(e) => setBenchmarkTicker(e.target.value.toUpperCase())} />
          </label>
          <button className="primary-button" onClick={() => void submitBacktest()} disabled={loading}>
            {loading ? "Running Backtest" : "Run Backtest"}
          </button>
        </div>
        {error ? <div className="alert">{error}</div> : null}
        <div className="backtest-note">
          Uses current Fragility Inputs and Portfolio holdings. Signals are calculated on rebalance dates and executed on the next trading day.
        </div>
      </section>

      {result ? (
        <>
          <section className="panel wide">
            <PanelTitle title="Performance Summary" caption={`${result.dataSource} | ${result.macroSource}`} />
            <BacktestMetrics metrics={result.metrics} />
          </section>
          <section className="panel wide">
            <PanelTitle title="Equity Curve" caption="Recommendation strategy versus passive alternatives" />
            <BacktestLineChart rows={result.equityCurve} mode="value" />
          </section>
          <section className="panel">
            <PanelTitle title="Drawdown" caption="Peak-to-trough loss during the test window" />
            <BacktestLineChart rows={result.drawdown} mode="drawdown" />
          </section>
          <section className="panel">
            <PanelTitle title="Allocation And HFFI Replay" caption="Target allocation pressure through time" />
            <BacktestAllocationChart rows={result.allocations} />
          </section>
      <section className="panel wide">
        <PanelTitle title="Trade Log" caption="Explainable rebalance actions generated by HFFI targets" />
        <BacktestTradeLog result={result} />
      </section>
      <section className="panel wide">
        <PanelTitle title="Signal Verification" caption="Live Buy/Sell/Hold engine compared with the latest backtest replay action" />
        <BacktestSignalAudit result={result} />
      </section>
        </>
      ) : (
        <EmptyState title="No backtest yet" detail="Choose a date range and run a historical replay." />
      )}
    </div>
  );
}

function BacktestMetrics({ metrics }: { metrics: BacktestResponse["metrics"] }) {
  return (
    <div className="backtest-metrics">
      {metrics.map((metric) => (
        <article key={metric.strategy}>
          <h3>{metric.strategy}</h3>
          <dl>
            <dt>Final value</dt><dd>{money(metric.finalValue, 0)}</dd>
            <dt>Total return</dt><dd>{signedPercent(metric.totalReturn)}</dd>
            <dt>CAGR</dt><dd>{signedPercent(metric.cagr)}</dd>
            <dt>Volatility</dt><dd>{percent(metric.volatility)}</dd>
            <dt>Max drawdown</dt><dd>{signedPercent(metric.maxDrawdown)}</dd>
            <dt>Sharpe</dt><dd>{number(metric.sharpe, 2)}</dd>
            <dt>Trades</dt><dd>{number(metric.trades, 0)}</dd>
            <dt>Turnover</dt><dd>{money(metric.turnover, 0)}</dd>
          </dl>
        </article>
      ))}
    </div>
  );
}

function BacktestLineChart({ rows, mode }: { rows: BacktestCurveRow[]; mode: "value" | "drawdown" }) {
  const width = 980;
  const height = 320;
  const topPad = 24;
  const bottomPad = 38;
  const chartHeight = height - topPad - bottomPad;
  const keys: Array<["recommendation" | "buyHold" | "benchmark", string]> = [
    ["recommendation", "HFFI Recommendation"],
    ["buyHold", "Buy and Hold"],
    ["benchmark", "Benchmark"],
  ];
  const values = rows.flatMap((row) => keys.map(([key]) => finiteNumber(row[key])));
  const minValue = mode === "drawdown" ? Math.min(...values, -0.01) : Math.min(...values);
  const maxValue = mode === "drawdown" ? 0 : Math.max(...values);
  const spread = maxValue - minValue || 1;
  const xFor = (index: number) => rows.length <= 1 ? 0 : index / (rows.length - 1) * width;
  const yFor = (value: number) => topPad + (maxValue - value) / spread * chartHeight;
  const pointsFor = (key: "recommendation" | "buyHold" | "benchmark") => (
    rows.map((row, index) => `${xFor(index)},${yFor(finiteNumber(row[key]))}`).join(" ")
  );
  const last = rows.at(-1);
  if (!rows.length) return <div className="empty-chart">No backtest series available.</div>;
  return (
    <div className="backtest-chart">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <g className="price-grid">
          {[0, 1, 2, 3].map((tick) => {
            const y = topPad + tick * chartHeight / 3;
            const value = maxValue - tick * spread / 3;
            return (
              <g key={tick}>
                <line x1="0" x2={width} y1={y} y2={y} />
                <text x={width - 8} y={y - 6}>{mode === "value" ? money(value, 0) : percent(value)}</text>
              </g>
            );
          })}
        </g>
        {keys.map(([key]) => <polyline key={key} className={`series-line ${key}`} points={pointsFor(key)} />)}
      </svg>
      <div className="chart-legend">
        {keys.map(([key, label]) => (
          <span key={key} className={key}>{label}: {last ? (mode === "value" ? money(last[key], 0) : percent(last[key])) : "n/a"}</span>
        ))}
      </div>
    </div>
  );
}

function BacktestAllocationChart({ rows }: { rows: BacktestResponse["allocations"] }) {
  const sampled = sampleRows(rows, 18);
  if (!sampled.length) return <div className="empty-chart">No allocation replay available.</div>;
  return (
    <div className="allocation-replay">
      {sampled.map((row) => (
        <article key={row.date}>
          <div><strong>{row.date}</strong><span>HFFI {number(row.hffi, 1)}</span></div>
          <div className="stacked-bar">
            <span className="equity" style={{ width: `${clamp(row.equity * 100, 0, 100)}%` }} />
            <span className="bond" style={{ width: `${clamp(row.bond * 100, 0, 100)}%` }} />
            <span className="cash" style={{ width: `${clamp(row.cash * 100, 0, 100)}%` }} />
          </div>
          <footer>{row.band} | E {percent(row.equity, 0)} B {percent(row.bond, 0)} C {percent(row.cash, 0)}</footer>
        </article>
      ))}
    </div>
  );
}

function BacktestTradeLog({ result }: { result: BacktestResponse }) {
  const rows = result.trades.slice(0, 80);
  if (!rows.length) return <div className="empty-chart">No rebalance trades were needed for this setup.</div>;
  return (
    <div className="trade-log">
      <div className="trade-head">
        <span>Signal</span><span>Execute</span><span>Action</span><span>Category</span><span>Amount</span><span>From</span><span>Target</span><span>Comment</span>
      </div>
      {rows.map((trade, index) => (
        <div className="trade-row" key={`${trade.signalDate}-${trade.category}-${index}`}>
          <span>{trade.signalDate}</span>
          <span>{trade.executionDate}</span>
          <strong>{trade.action}</strong>
          <span>{trade.category.toUpperCase()}</span>
          <span>{money(trade.amount, 0)}</span>
          <span>{percent(trade.fromWeight)}</span>
          <span>{percent(trade.targetWeight)}</span>
          <em>{trade.comment}</em>
        </div>
      ))}
    </div>
  );
}

function BacktestSignalAudit({ result }: { result: BacktestResponse }) {
  const rows = result.signalAudit ?? [];
  if (!rows.length) return <div className="empty-chart">No holding signals available for verification.</div>;
  return (
    <div className="signal-audit-grid">
      {rows.map((row) => (
        <article key={`${row.ticker}-${row.category}`}>
          <div>
            <strong>{row.ticker}</strong>
            <span className={row.matched ? "good" : "warn"}>{row.matched ? "MATCH" : "REVIEW"}</span>
          </div>
          <dl>
            <dt>Live engine</dt><dd>{row.liveAction}</dd>
            <dt>Backtest replay</dt><dd>{row.latestBacktestAction}</dd>
            <dt>HFFI</dt><dd>{number(row.hffi, 1)}</dd>
            <dt>Suitability</dt><dd>{row.suitability === null ? "n/a" : number(row.suitability, 3)}</dd>
          </dl>
          <p>{row.comment}</p>
        </article>
      ))}
    </div>
  );
}

function Evidence({ analysis }: { analysis: AnalysisResponse | null }) {
  if (!analysis) return <EmptyState title="No evidence yet" detail="Run analysis to populate the Evidence Lab." />;
  return (
    <div className="evidence-layout">
      <section className="panel">
        <PanelTitle title="Counterfactual Simulator" caption="Ranked by estimated HFFI improvement" />
        <div className="rank-list">
          {analysis.counterfactuals.map((row) => (
            <article key={row.Action}>
              <span>Rank {row.Rank}</span>
              <h3>{row.Action}</h3>
              <strong>{signedNumber(row["HFFI improvement"], 1)} HFFI</strong>
              <p>{row["Why it matters"]}</p>
              <footer>{row.Method}</footer>
            </article>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelTitle title="Decision Evidence" caption="Confidence, timing, and rationale" />
        <div className="data-table compact">
          <div className="evidence-head"><span>Decision</span><span>Reco</span><span>Conf</span><span>When</span></div>
          {analysis.decisionEvidence.map((row) => (
            <div className="evidence-row" key={row.Decision}>
              <span>{row.Decision}</span><strong>{row.Recommendation}</strong><span>{percent(row.Confidence)}</span><em>{row.When}</em>
            </div>
          ))}
        </div>
      </section>
      <RecommendationAnalytics analysis={analysis} />
      <InvestmentPlanning analysis={analysis} />
    </div>
  );
}

function Scenarios({ analysis }: { analysis: AnalysisResponse | null }) {
  if (!analysis) return <EmptyState title="No scenarios yet" detail="Run analysis to load stress scenarios." />;
  return (
    <div className="scenario-layout">
      <section className="panel">
        <PanelTitle title="Stress Scenario Replay" caption="HFFI response under named economic shocks" />
        <div className="scenario-grid">
          {analysis.stress.map((row) => (
            <article className="scenario-card" key={String(row.scenario)}>
              <div><span>{String(row.scenario).replaceAll("_", " ")}</span><strong>{number(Number(row.HFFI), 1)}</strong></div>
              <p>{String(row.band)}</p>
              <div className="mini-bars">
                {["L", "D", "E", "P", "M"].map((key) => (
                  <span key={key} style={{ height: `${Number(row[key] ?? 0) * 100}%` }} title={key} />
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>
      <MonteCarloPanel analysis={analysis} />
    </div>
  );
}

function RecommendationAnalytics({ analysis }: { analysis: AnalysisResponse }) {
  const actualBreakdown = allocationBreakdownFromAnalysis(analysis);
  const portfolioRows = analysis.portfolioScores.slice(0, 4).map((row) => ({
    label: String(row.portfolio ?? "Portfolio"),
    suitability: finiteNumber(row.suitability_score),
    expectedReturn: finiteNumber(row.exp_return),
    volatility: finiteNumber(row.volatility),
    drawdown: finiteNumber(row.max_drawdown),
  }));
  const signalRows = analysis.marketSignals.slice(0, 8).map((signal) => ({
    label: signal.ticker,
    name: signal.name,
    score: finiteNumber(signal.ds_score),
    probability: finiteNumber(signal.ml_probability),
    recommendation: signal.recommendation,
  }));
  const bestPortfolio = portfolioRows[0];

  return (
    <section className="panel wide recommendation-graphs">
      <PanelTitle
        title="Recommendation Analytics"
        caption="Allocation target, HFFI portfolio fit, and ML-ranked trade candidates"
      />
      <div className="analytics-grid">
        <div className="analytics-card">
          <h3>Allocation Monitor</h3>
          <AllocationDonut allocation={actualBreakdown} target={analysis.targetAllocation} compact />
        </div>
        <div className="analytics-card">
          <h3>Portfolio Choice</h3>
          {bestPortfolio ? <p className="chart-callout">Recommended: <strong>{bestPortfolio.label}</strong></p> : null}
          <div className="score-bars">
            {portfolioRows.map((row) => (
              <div className="score-row" key={row.label}>
                <span>{row.label}</span>
                <div className="bar-rail"><span className="suitability" style={{ width: `${clamp(row.suitability * 100, 0, 100)}%` }} /></div>
                <em>{number(row.suitability, 3)}</em>
              </div>
            ))}
          </div>
          <div className="portfolio-risk-grid">
            {portfolioRows.slice(0, 3).map((row) => (
              <article key={`${row.label}-risk`}>
                <strong>{row.label}</strong>
                <span>Return {percent(row.expectedReturn)}</span>
                <span>Vol {percent(row.volatility)}</span>
                <span>Max DD {percent(row.drawdown)}</span>
              </article>
            ))}
          </div>
        </div>
        <div className="analytics-card">
          <h3>Trade Signal Ranking</h3>
          <div className="score-bars">
            {signalRows.map((row) => (
              <div className="score-row signal-score-row" key={row.label}>
                <span>{row.label}</span>
                <div className="bar-rail"><span className="signal" style={{ width: `${clamp(row.score * 100, 0, 100)}%` }} /></div>
                <em>{row.recommendation} {number(row.score, 3)}</em>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function InvestmentPlanning({ analysis }: { analysis: AnalysisResponse }) {
  const rows = analysis.investmentPlan ?? [];
  return (
    <section className="panel wide investment-plan">
      <PanelTitle
        title="Investment Planning"
        caption="AI model ranking across Equity, Bond, Sector, Commodity, Forex, and Index candidates"
      />
      <div className="plan-strategy">
        Strategy: RF/GB suitability model + live/fallback market momentum, safety, drawdown, HFFI guardrails, allocation gap, and buying capacity.
      </div>
      {rows.length ? (
        <div className="investment-table">
          <div className="investment-head">
            <span>Ticker</span><span>Category</span><span>Reco</span><span>Amount</span><span>Current</span><span>Entry</span><span>Approx target</span><span>Downside</span><span>Hold</span><span>Comment</span>
          </div>
          {rows.map((row) => (
            <div className="investment-row" key={`${row.ticker}-${row.category}`}>
              <strong>{row.ticker}<small>{row.name}</small></strong>
              <span>{row.category.toUpperCase()}</span>
              <span>{row.recommendation}</span>
              <span>{money(row.suggestedAmount, 0)}</span>
              <span>{money(row.currentPrice, 2)}</span>
              <span>{money(row.expectedEntryPrice, 2)}</span>
              <span>{money(row.targetPrice, 2)}<small>{signedPercent(row.expectedMovePct)}</small></span>
              <span>{money(row.downsideTriggerPrice, 2)}</span>
              <span>{row.holdingPeriod}</span>
              <em>{row.comment}</em>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-chart">Run analysis to generate investment planning candidates.</div>
      )}
    </section>
  );
}

function MonteCarloPanel({ analysis }: { analysis: AnalysisResponse }) {
  const mc = analysis.monteCarlo;
  const bins = mc?.histogram ?? [];
  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const width = 940;
  const height = 280;
  const topPad = 24;
  const bottomPad = 36;
  const chartHeight = height - topPad - bottomPad;
  const barWidth = width / Math.max(bins.length, 1);
  const xForScore = (score: number) => clamp(score / 100 * width, 0, width);

  return (
    <section className="panel wide monte-carlo-panel">
      <PanelTitle title="Monte Carlo Stress Distribution" caption="2000 random macro, income, rate, and drawdown shock paths" />
      <div className="monte-grid">
        <div className="monte-chart">
          <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            <g className="price-grid">
              {[0, 25, 50, 75, 100].map((tick) => {
                const x = xForScore(tick);
                return (
                  <g key={tick}>
                    <line x1={x} x2={x} y1={topPad} y2={height - bottomPad} />
                    <text x={x + 4} y={height - 10}>{tick}</text>
                  </g>
                );
              })}
            </g>
            {bins.map((bin, index) => {
              const barHeight = bin.count / maxCount * chartHeight;
              return (
                <rect
                  key={`${bin.binStart}-${bin.binEnd}`}
                  className="histogram-bar"
                  x={index * barWidth + 1}
                  y={height - bottomPad - barHeight}
                  width={Math.max(barWidth - 2, 2)}
                  height={barHeight}
                />
              );
            })}
            {[
              { label: "P05", value: mc.p05 },
              { label: "P50", value: mc.p50 },
              { label: "P95", value: mc.p95 },
            ].map((mark) => (
              <g className="percentile-mark" key={mark.label}>
                <line x1={xForScore(mark.value)} x2={xForScore(mark.value)} y1={topPad} y2={height - bottomPad} />
                <text x={xForScore(mark.value) + 5} y={topPad + 14}>{mark.label}</text>
              </g>
            ))}
          </svg>
        </div>
        <div className="monte-metrics">
          <Metric label="Mean HFFI" value={number(mc.mean, 1)} />
          <Metric label="Std dev" value={number(mc.std, 1)} />
          <Metric label="5th - 95th" value={`${number(mc.p05, 0)} - ${number(mc.p95, 0)}`} />
          <Metric label="P(severe)" value={percent(mc.probSevere)} />
        </div>
      </div>
    </section>
  );
}

function ModelPanel({ analysis }: { analysis: AnalysisResponse | null }) {
  if (!analysis) return <EmptyState title="No model output yet" detail="Run analysis to load model diagnostics." />;
  return (
    <div className="model-layout">
      <section className="panel">
        <PanelTitle title="Feature Evidence" caption="Current household values against model importance" />
        <div className="feature-list">
          {analysis.featureEvidence.map((row) => (
            <div className="feature-row" key={String(row.Feature)}>
              <span>{String(row.Feature)}</span>
              <strong>{number(Number(row.Importance ?? 0), 4)}</strong>
              <em>{String(row.Interpretation)}</em>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelTitle title="Model Card" caption="Validation, limitations, and security posture" />
        <div className="model-card-list">
          {analysis.modelCard.map((row) => (
            <article key={row.Section}>
              <h3>{row.Section}</h3>
              <p>{row.Evidence}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function RecommendationStack({ analysis }: { analysis: AnalysisResponse | null }) {
  return (
    <section className="panel">
      <PanelTitle title="Market Signals" caption="ML + HFFI ranked candidates" />
      <div className="signal-list">
        {(analysis?.marketSignals ?? []).slice(0, 7).map((signal) => (
          <article key={signal.ticker}>
            <div><strong>{signal.ticker}</strong><span>{signal.recommendation}</span></div>
            <p>{signal.name}</p>
            <footer>
              <span>DS {number(signal.ds_score, 3)}</span>
              <span>ML {percent(signal.ml_probability)}</span>
            </footer>
          </article>
        ))}
      </div>
    </section>
  );
}

function ActionStack({ analysis }: { analysis: AnalysisResponse | null }) {
  return (
    <section className="panel">
      <PanelTitle title="Execution Notes" caption="Educational decision support only" />
      <div className="note-stack">
        {(analysis?.holdingActions ?? []).slice(0, 4).map((action) => (
          <article key={action.ticker}>
            <span>{action.status}</span>
            <h3>{action.ticker}</h3>
            <p>{action.when}</p>
          </article>
        ))}
        <article>
          <span>GUARDRAIL</span>
          <h3>No trade execution</h3>
          <p>The terminal only produces decision-support signals. It does not place orders.</p>
        </article>
      </div>
    </section>
  );
}

function CandlestickChart({ rows, moving }: { rows: ChartRow[]; moving?: boolean }) {
  const [hovered, setHovered] = useState<ChartRow | null>(null);
  const [windowEnd, setWindowEnd] = useState(100);
  const [windowSize, setWindowSize] = useState(65);
  if (!rows.length) return <div className="empty-chart">Generate a chart to view price history.</div>;
  const fullCandles = rows
    .filter((row) => (
      Number.isFinite(row.open)
      && Number.isFinite(row.high)
      && Number.isFinite(row.low)
      && Number.isFinite(row.close)
    ));
  const visibleCount = Math.min(
    fullCandles.length,
    Math.max(20, Math.round(fullCandles.length * windowSize / 100)),
  );
  const endIndex = Math.min(
    fullCandles.length,
    Math.max(visibleCount, Math.round(fullCandles.length * windowEnd / 100)),
  );
  const candles = fullCandles.slice(Math.max(0, endIndex - visibleCount), endIndex);
  if (!candles.length) return <div className="empty-chart">No readable OHLC data for this symbol/timeframe.</div>;
  const display = hovered ?? candles.at(-1)!;
  const min = Math.min(...candles.map((row) => Number(row.low)));
  const max = Math.max(...candles.map((row) => Number(row.high)));
  const spread = max - min || 1;
  const width = 1180;
  const height = 520;
  const topPad = 26;
  const bottomPad = 42;
  const chartHeight = height - topPad - bottomPad;
  const step = width / Math.max(candles.length, 1);
  const candleWidth = clamp(step * 0.62, 4, 14);
  const yFor = (value: number) => topPad + (max - value) / spread * chartHeight;
  const formatDate = (value: string) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };
  return (
    <div className="chart-wrap">
      {moving ? <div className="live-badge">LIVE</div> : null}
      <div className="ohlc-strip">
        <span>{formatDate(display.date)}</span>
        <span>O {money(display.open, 2)}</span>
        <span>H {money(display.high, 2)}</span>
        <span>L {money(display.low, 2)}</span>
        <span>C {money(display.close, 2)}</span>
        <span>V {number(Number(display.volume ?? 0), 0)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" onMouseLeave={() => setHovered(null)}>
        <g className="price-grid">
          {[0, 1, 2, 3, 4].map((tick) => {
            const y = topPad + tick * chartHeight / 4;
            const price = max - tick * spread / 4;
            return (
              <g key={tick}>
                <line x1="0" x2={width} y1={y} y2={y} />
                <text x={width - 8} y={y - 6}>{money(price, 2)}</text>
              </g>
            );
          })}
        </g>
        {candles.map((row, index) => {
          const open = Number(row.open);
          const close = Number(row.close);
          const high = Number(row.high);
          const low = Number(row.low);
          const up = close >= open;
          const x = index * step + step / 2;
          const yHigh = yFor(high);
          const yLow = yFor(low);
          const yOpen = yFor(open);
          const yClose = yFor(close);
          const bodyY = Math.min(yOpen, yClose);
          const bodyHeight = Math.max(Math.abs(yClose - yOpen), 3);
          return (
            <g
              key={`${row.date}-${index}`}
              className={`candle ${up ? "up" : "down"} ${moving ? "moving" : ""}`}
              onMouseEnter={() => setHovered(row)}
            >
              <line className="candle-wick" x1={x} x2={x} y1={yHigh} y2={yLow} />
              <rect className="candle-body" x={x - candleWidth / 2} y={bodyY} width={candleWidth} height={bodyHeight} rx="1.5" />
              <rect className="candle-hit" x={x - step / 2} y={topPad} width={step} height={chartHeight} />
            </g>
          );
        })}
      </svg>
      <div className="range-controls">
        <label>
          <span>Visible candles</span>
          <input type="range" min="20" max="100" step="5" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))} />
        </label>
        <label>
          <span>Slide chart</span>
          <input type="range" min="0" max="100" step="1" value={windowEnd} onChange={(e) => setWindowEnd(Number(e.target.value))} />
        </label>
      </div>
      <div className="chart-meta">
        <span>Range low {money(min, 2)}</span>
        <span>Range high {money(max, 2)}</span>
        <span>Candles {candles.length} of {fullCandles.length}</span>
      </div>
    </div>
  );
}

function PanelTitle({ title, caption }: { title: string; caption: string }) {
  return (
    <div className="panel-title">
      <h2>{title}</h2>
      <p>{caption}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-box">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <section className="panel empty-state">
      <h2>{title}</h2>
      <p>{detail}</p>
    </section>
  );
}

function calculatePortfolioAllocation(holdings: HoldingInput[], liquidSavings: number): AllocationBreakdown {
  const amounts: Record<AllocationKey, number> = {
    equity: 0,
    bond: 0,
    cash: Math.max(finiteNumber(liquidSavings), 0),
  };
  holdings.forEach((holding) => {
    const category = holding.category;
    const units = Math.max(finiteNumber(holding.units), 0);
    const buyPrice = Math.max(finiteNumber(holding.buyPrice), 0);
    amounts[category] += units * buyPrice;
  });
  return buildAllocationBreakdown(amounts);
}

function allocationBreakdownFromAnalysis(analysis: AnalysisResponse): AllocationBreakdown {
  const amounts: Record<AllocationKey, number> = { equity: 0, bond: 0, cash: 0 };
  analysis.allocationSummary.forEach((row) => {
    const category = String(row.category ?? "").toLowerCase() as AllocationKey;
    if (!ALLOCATION_KEYS.includes(category)) return;
    amounts[category] = Math.max(finiteNumber(row.invested_amount), 0);
  });
  const breakdown = buildAllocationBreakdown(amounts);
  if (breakdown.total > 0) return breakdown;
  return {
    amounts,
    weights: {
      equity: finiteNumber(analysis.actualAllocation.equity),
      bond: finiteNumber(analysis.actualAllocation.bond),
      cash: finiteNumber(analysis.actualAllocation.cash),
    },
    total: 0,
  };
}

function buildAllocationBreakdown(amounts: Record<AllocationKey, number>): AllocationBreakdown {
  const total = ALLOCATION_KEYS.reduce((sum, key) => sum + amounts[key], 0);
  const weights: Record<AllocationKey, number> = {
    equity: total > 0 ? amounts.equity / total : 0,
    bond: total > 0 ? amounts.bond / total : 0,
    cash: total > 0 ? amounts.cash / total : 0,
  };
  return { amounts, weights, total };
}

function finiteNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function sampleRows<T>(rows: T[], limit: number): T[] {
  if (rows.length <= limit) return rows;
  const step = Math.ceil(rows.length / limit);
  return rows.filter((_, index) => index % step === 0).slice(0, limit);
}

function scoreTone(score?: number): string {
  if (score === undefined) return "neutral";
  if (score < 30) return "good";
  if (score < 60) return "warn";
  return "bad";
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
