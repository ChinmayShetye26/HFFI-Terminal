import type {
  AnalysisResponse,
  AnalyzeRequest,
  Asset,
  AssetCategory,
  BacktestRequest,
  BacktestResponse,
  ChartRow,
  MarketSnapshotRow,
  ReportRequest,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "hffi_access_token";

export type AuthUser = {
  username: string;
  role: string;
};

export type SecurityConfig = {
  authEnabled: boolean;
  tokenTtlMinutes: number;
  allowedOrigins: string[];
};

export type LoginResponse = {
  accessToken: string;
  tokenType: string;
  expiresIn: number;
  user: AuthUser;
};

export function getStoredToken(): string {
  return window.localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setStoredToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearStoredToken(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...Object.fromEntries(new Headers(options?.headers ?? {}).entries()),
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, {
    headers,
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    if (response.status === 401) clearStoredToken();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchSecurityConfig(): Promise<SecurityConfig> {
  return requestJson("/api/security/config");
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const payload = await requestJson<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setStoredToken(payload.accessToken);
  return payload;
}

export async function fetchCurrentUser(): Promise<{ user: AuthUser }> {
  return requestJson("/api/auth/me");
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

export async function exportExcelReport(payload: ReportRequest): Promise<Blob> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}/api/report/excel`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    if (response.status === 401) clearStoredToken();
    throw new Error(text || `Report export failed with ${response.status}`);
  }
  return response.blob();
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
