# HFFI Terminal Security Controls

The terminal handles sensitive household finance inputs. The current build adds
the following safeguards:

- Authentication with HMAC-signed bearer tokens.
- Role-based access for analytical endpoints.
- Rate limits for login, chart, market, analysis, and backtesting endpoints.
- Strict CORS configured through `HFFI_ALLOWED_ORIGINS`.
- Security headers on API responses, including CSP, frame protection, no-sniff,
  referrer policy, permissions policy, and no-store cache control.
- Server-side validation for tickers, holdings count, financial input ranges,
  date ranges, and chart period/interval values.
- Safe audit logging in `logs/security_audit.log` without raw income, savings,
  debt, or expense values.
- API keys and credentials read from `.env`; sample files contain placeholders
  only.
- No trade execution. The app remains educational decision-support software.

## Local Credentials

Create `.env` from `.env.example` and change the defaults:

```text
HFFI_AUTH_ENABLED=true
HFFI_SECRET_KEY=replace-with-a-long-random-secret
HFFI_ADMIN_USERNAME=admin
HFFI_ADMIN_PASSWORD=change-me-now
HFFI_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Optional role accounts:

```text
HFFI_ANALYST_USERNAME=analyst
HFFI_ANALYST_PASSWORD=replace-me
HFFI_VIEWER_USERNAME=viewer
HFFI_VIEWER_PASSWORD=replace-me
```

`admin` and `analyst` can run analysis/backtesting. `viewer` can access read
endpoints such as assets, charts, market snapshots, and `/api/auth/me`.

## Production Checklist

- Replace `HFFI_SECRET_KEY` with a long random value.
- Replace all demo passwords and prefer password hashes.
- Deploy behind HTTPS/TLS.
- Set `HFFI_ALLOWED_ORIGINS` to the production frontend domain only.
- Keep `.env` out of Git.
- Rotate API keys if they were ever committed or shared.
- Run dependency scans regularly:
  - `npm audit`
  - `pip-audit` when available
- Or run the bundled check script:
  - `.\scripts\security_check.ps1`
- Review `logs/security_audit.log` for repeated login failures or rate-limit
  events.
