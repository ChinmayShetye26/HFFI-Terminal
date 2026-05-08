export type AssetCategory = "equity" | "bond" | "sector" | "commodity" | "forex" | "index";

export type Asset = {
  ticker: string;
  name: string;
  category: AssetCategory;
  subcategory: string;
  fetchSymbol: string;
};

export type HoldingInput = {
  id: string;
  category: "equity" | "bond";
  ticker: string;
  name: string;
  units: number;
  buyPrice: number;
};

export type HouseholdInput = {
  monthlyIncome: number;
  monthlyEssentialExpenses: number;
  monthlyTotalExpenses: number;
  liquidSavings: number;
  totalDebt: number;
  monthlyDebtPayment: number;
  portfolioVolatility: number;
  expectedDrawdown: number;
  rateSensitivity: number;
  employmentType: string;
  dependents: number;
  portfolioWeights: {
    equity: number;
    bond: number;
    cash: number;
  };
};

export type AnalyzeRequest = {
  household: HouseholdInput;
  holdings: HoldingInput[];
};

export type BacktestRequest = {
  household: HouseholdInput;
  holdings: HoldingInput[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  frequency: "weekly" | "monthly" | "quarterly";
  transactionCostPct: number;
  benchmarkTicker: string;
};

export type HffiResult = {
  score: number;
  band: string;
  distressProbability: number;
  components: Record<string, number>;
  contributions: Record<string, number>;
};

export type Recommendation = {
  priority: number;
  component: string;
  action: string;
  detail: string;
  expectedImpact: string;
};

export type HoldingAction = {
  category: string;
  ticker: string;
  name: string;
  status: "BUY" | "HOLD" | "SELL";
  strategy: string;
  when: string;
  comment: string;
  investedAmount: number;
  currentValue: number;
  unrealizedPl: number;
  unrealizedPlPct: number;
  allocationWeightPct: number;
  targetCategoryPct: number;
  suitability: number | null;
};

export type MarketSignal = {
  ticker: string;
  name: string;
  category: string;
  recommendation: string;
  suggested_monthly_amount?: number;
  ml_probability?: number;
  suitability_score?: number;
  allocation_gap?: number;
  ds_score?: number;
  segment?: string;
  comment?: string;
};

export type InvestmentPlanItem = {
  ticker: string;
  name: string;
  category: string;
  recommendation: string;
  modelStrategy: string;
  suggestedAmount: number;
  currentPrice: number | null;
  expectedEntryPrice: number | null;
  targetPrice: number | null;
  downsideTriggerPrice: number | null;
  expectedMovePct: number;
  holdingPeriod: string;
  confidence: number;
  comment: string;
};

export type EvidenceRow = {
  Decision: string;
  Recommendation: string;
  Confidence: number;
  "Confidence band": string;
  "Primary evidence": string;
  When: string;
  Comment: string;
};

export type CounterfactualRow = {
  Action: string;
  Method: string;
  "New HFFI": number;
  "HFFI improvement": number;
  "New band": string;
  "Distress probability change": number;
  "Why it matters": string;
  Rank: number;
};

export type MonteCarloBin = {
  binStart: number;
  binEnd: number;
  count: number;
};

export type MonteCarloSummary = {
  mean: number;
  std: number;
  p05: number;
  p50: number;
  p95: number;
  probSevere: number;
  histogram: MonteCarloBin[];
};

export type AnalysisResponse = {
  macro: Record<string, number | string | null>;
  hffi: HffiResult;
  segment: string;
  recommendations: Recommendation[];
  targetAllocation: Record<string, number>;
  actualAllocation: Record<string, number>;
  allocationSummary: Array<Record<string, number | string | null>>;
  holdings: Array<Record<string, number | string | null>>;
  holdingActions: HoldingAction[];
  selectedMarket: Record<string, number | string | null>;
  marketSignals: MarketSignal[];
  investmentPlan: InvestmentPlanItem[];
  decisionEvidence: EvidenceRow[];
  counterfactuals: CounterfactualRow[];
  featureEvidence: Array<Record<string, number | string | null>>;
  modelCard: Array<Record<string, string>>;
  modelPerformance: Array<Record<string, number | string>>;
  stress: Array<Record<string, number | string>>;
  monteCarlo: MonteCarloSummary;
  portfolioScores: Array<Record<string, number | string>>;
  riskOff: boolean;
};

export type MarketSnapshotRow = {
  ticker: string;
  fetchSymbol: string;
  name: string;
  category: string;
  subcategory: string;
  price: number | null;
  change: number | null;
  changePct: number | null;
  dataSource: string;
};

export type ChartRow = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
};

export type BacktestMetric = {
  strategy: string;
  finalValue: number;
  totalReturn: number;
  cagr: number;
  volatility: number;
  maxDrawdown: number;
  sharpe: number;
  trades: number;
  turnover: number;
};

export type BacktestCurveRow = {
  date: string;
  recommendation: number;
  buyHold: number;
  benchmark: number;
};

export type BacktestAllocationRow = {
  date: string;
  equity: number;
  bond: number;
  cash: number;
  hffi: number;
  band: string;
};

export type BacktestTrade = {
  signalDate: string;
  executionDate: string;
  category: string;
  action: string;
  amount: number;
  fromWeight: number;
  targetWeight: number;
  hffi: number;
  band: string;
  comment: string;
};

export type BacktestSignalAudit = {
  ticker: string;
  category: string;
  liveAction: string;
  latestBacktestAction: string;
  matched: boolean;
  hffi: number;
  suitability: number | null;
  comment: string;
};

export type BacktestResponse = {
  settings: Record<string, number | string>;
  dataSource: string;
  macroSource: string;
  metrics: BacktestMetric[];
  equityCurve: BacktestCurveRow[];
  drawdown: BacktestCurveRow[];
  allocations: BacktestAllocationRow[];
  trades: BacktestTrade[];
  signalAudit: BacktestSignalAudit[];
};
