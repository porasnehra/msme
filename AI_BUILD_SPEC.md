# AI/ML Build Spec — MSME Financial Health Score (with Explainable AI)

> Paste this whole document as the instruction/prompt when handing the AI/ML
> part of this project to an agentic coding tool (e.g. Antigravity, Cursor,
> Claude Code, Copilot Workspace). It fully specifies the data, modeling,
> serving, and explainability requirements so the tool can build or extend
> the backend without further back-and-forth.

## 1. Problem

Build the AI/ML backend for a system that scores the creditworthiness of
Micro, Small, and Medium Enterprises (MSMEs) that lack traditional audited
financial paperwork. Instead of balance sheets, the system uses alternative
digital-footprint data: UPI transaction behavior, GST filing history, and
digital cash-flow signals. Output is a 0–100 "Financial Health Score" a bank
can use for loan decisioning.

## 2. Dataset

Generate a synthetic dataset of MSME profiles (12,000+ rows) with these
feature groups:

- **Business profile**: sector (categorical, 8 classes), business vintage in months
- **UPI signals**: monthly inflow, monthly outflow, transaction count, average
  transaction value, 6-month inflow growth rate, cash-flow volatility
  (coefficient of variation), average bank balance
- **GST signals**: monthly turnover, filing regularity (fraction on-time),
  count of late filings in the last 12 months
- **Payment discipline**: digital invoice volume, on-time payment ratio,
  bounced payment count, existing loan EMI-to-inflow ratio

Construct the target score as a **weighted blend of normalized 0–100
sub-scores** (revenue strength, GST compliance, payment discipline, cash-flow
stability, debt burden, vintage, growth) rather than one linear formula —
this keeps the distribution realistic (a bell curve, not saturated at 0/100)
and gives non-linear/tree-based models real signal to outperform linear
regression. Add a few explicit non-linear interaction terms (e.g., poor GST
compliance AND high volatility compounds the penalty) and irreducible
Gaussian noise so no model can fit it perfectly.

## 3. Modeling

Train and compare **at least these five model families** on an 80/20 train/test
split, using R², MAE, and RMSE:

1. Ridge Regression (linear baseline)
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost Regressor
5. A small feed-forward Deep Neural Network (MLPRegressor or Keras equivalent)

Select the model with the **highest test-set R²** and persist it, along with
the fitted `StandardScaler`, the sector-to-integer encoding map, and the
ordered feature name list — all four are required to reproduce predictions
at serving time.

## 4. Explainable AI (required)

The score must never be a black box to a credit officer. Add SHAP-based
explainability:

- If the winning model is tree-based (Random Forest / Gradient Boosting /
  XGBoost), use `shap.TreeExplainer(model)` — no extra training step needed,
  it introspects the trained trees directly and is fast enough (<5ms) to run
  per-request in the API.
- For every prediction, compute the SHAP values for that single input and
  return the **top 5 features by absolute impact**, each with:
  - human-readable feature label (not the raw column name)
  - the value the applicant actually provided
  - the signed impact on the score (SHAP value)
  - a `direction` field: `"increased"` or `"decreased"`
- Also return the explainer's `expected_value` (the model's average
  prediction across the training population) as `base_score`, so the officer
  can see "this business started from the population baseline of X and was
  pushed up/down to Y because of these specific factors."
- Explainability must be a **non-blocking bonus**: if SHAP computation fails
  for any reason, the endpoint should still return the core score and simply
  omit `top_factors`, wrapped in its own try/except so it can never break the
  primary prediction.
- If the winning model turns out to be the neural network (which
  `TreeExplainer` doesn't support), fall back gracefully — either skip SHAP
  for that model or use `shap.KernelExplainer` with a small background
  sample, accepting the extra latency.

## 5. API contract

Serve the model behind a FastAPI app with:

- `GET /health` — returns whether the model loaded successfully, its type,
  and the valid sector list (for debugging deploys, e.g. on Render)
- `POST /predict` — accepts the raw business metrics (not pre-encoded/scaled)
  and returns:
  ```json
  {
    "financial_health_score": 76.2,
    "risk_category": "Low Risk | Medium Risk | High Risk",
    "loan_recommendation": "human-readable explanation",
    "model_used": "GradientBoostingRegressor",
    "base_score": 63.25,
    "top_factors": [
      {"feature": "GST Filing Regularity", "value_provided": 0.9, "impact": 3.59, "direction": "increased"}
    ]
  }
  ```
- Validate all inputs with Pydantic (correct types, sensible bounds, sector
  must be one of the trained categories — reject with HTTP 400 otherwise).
- Enable CORS for all origins so any frontend (React, Streamlit, etc.) can
  call it without extra proxy configuration.
- Load all model artifacts once at startup, not per-request. Never crash the
  whole process if artifacts fail to load — surface the error through
  `/health` instead so deployment logs are debuggable.

## 6. Non-functional requirements

- Keep dependencies pinned in `requirements.txt` to the exact versions used
  during training — pickled scikit-learn/XGBoost models are sensitive to
  version drift, and this alone is the most common cause of deploy failures.
- The service must run on a free-tier PaaS (Render/Railway) with:
  `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Total artifact size (model + scaler + encoders) should stay well under
  10MB so it can be committed directly to the repo rather than needing
  external storage.

---

*This spec was generated for the MSME Financial Health Score hackathon
project (Track 03) and reflects a backend already implemented in
`app.py` / `train_model.py`, with SHAP explainability wired into the
`/predict` response. Use it as-is to extend the same project, or as a
template to brief a coding agent on a similar alt-data scoring system.*
