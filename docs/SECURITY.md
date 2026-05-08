# Security Measures For HFFI Terminal

Implemented safeguards:

1. **No chatbot tab in the terminal**: the Streamlit UI no longer sends user data to an LLM service.
2. **Secret hygiene**: `.env.example` and `.env.example.txt` contain placeholders only. Real keys belong only in local `.env`, which is gitignored.
3. **No direct trading execution**: recommendations are educational decision-support outputs. The app never places orders or connects to brokerage trading endpoints.
4. **Ticker validation**: chart and market-data requests accept only bounded ticker symbols with safe characters.
5. **Provider scoping**: paid provider keys are optional and selected explicitly with `MARKET_PROVIDER`.
6. **FRED no-key fallback**: macro data can use FRED public CSV, reducing the need to distribute API keys for demos.
7. **Local persistence only**: household runs remain in local SQLite by default.
8. **Input minimization**: detailed holdings are used for current calculations and are not added to outbound model calls.
9. **Cache isolation**: market-data cache files are keyed by provider and ticker set to avoid mixing results across categories or portfolios.

Recommended production hardening:

1. Add Streamlit authentication and deploy behind HTTPS.
2. Encrypt SQLite or move sensitive records to an encrypted managed database.
3. Rotate any API key that was ever committed, copied into a report, or shared in screenshots.
4. Add role-based access controls for advisors, analysts, and household users.
5. Add audit logging for report generation and data exports.
6. Add automated secret scanning in CI.
7. Add explicit privacy notices before collecting real household holdings.
