$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

Write-Host "Running HFFI security checks..."

Write-Host "`n[1/4] Python compile check"
.\.venv\Scripts\python.exe -m py_compile api\main.py api\security.py hffi_core\portfolio_advisor.py hffi_core\scoring.py

Write-Host "`n[2/4] Frontend dependency audit"
Push-Location frontend
try {
  npm.cmd audit --audit-level=high
}
finally {
  Pop-Location
}

Write-Host "`n[3/4] Python dependency audit"
$pipAudit = Get-Command pip-audit -ErrorAction SilentlyContinue
if ($pipAudit) {
  pip-audit
}
else {
  Write-Host "pip-audit is not installed. Install it with: pip install pip-audit"
}

Write-Host "`n[4/4] Secret placeholder check"
$patterns = @(
  "FRED_API_KEY=.+[A-Za-z0-9]{10}",
  "NEWSAPI_KEY=.+[A-Za-z0-9]{10}",
  "POLYGON_API_KEY=.+[A-Za-z0-9]{10}",
  "ALPACA_API_KEY=.+[A-Za-z0-9]{10}",
  "ALPACA_SECRET_KEY=.+[A-Za-z0-9]{10}"
)
$hits = Select-String -Path ".env.example", ".env.example.txt" -Pattern $patterns -ErrorAction SilentlyContinue
if ($hits) {
  $hits | Format-Table Path, LineNumber, Line -AutoSize
  throw "Potential real secret detected in sample env files."
}
Write-Host "No sample-file secrets detected."

Write-Host "`nSecurity checks complete."
