import type {
  AnalysisResponse,
  AnalyzeRequest,
  Asset,
  AssetCategory,
  BacktestRequest,
  BacktestResponse,
  ChartRow,
  MarketSnapshotRow,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchAssets(): Promise<{
  categories: AssetCategory[];
  assets: Record<AssetCategory, Asset[]>;
}> {
  return requestJson("/api/assets");
}

export async function analyzePortfolio(payload: AnalyzeRequest): Promise<AnalysisResponse> {
  return requestJson("/api/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runBacktest(payload: BacktestRequest): Promise<BacktestResponse> {
  return requestJson("/api/backtest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchMarket(category: AssetCategory): Promise<{
  category: AssetCategory;
  rows: MarketSnapshotRow[];
}> {
  return requestJson(`/api/market/${category}`);
}

export async function fetchChart(
  ticker: string,
  period = "6mo",
  interval = "1d",
): Promise<{ ticker: string; dataSource: string; rows: ChartRow[] }> {
  const query = new URLSearchParams({ period, interval }).toString();
  return requestJson(`/api/chart/${encodeURIComponent(ticker)}?${query}`);
}
