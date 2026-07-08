# MSME Financial Health Score API

AI system for Track 03 — turns a small business's alternative digital
footprint (UPI transactions, GST filings, digital cash flow) into a
0–100 Financial Health Score + risk category + loan recommendation,
for businesses that don't have audited balance sheets.

## What's in this folder

| File | Purpose |
|---|---|
| `train_model.py` | Generates the synthetic dataset and trains/compares 5 models |
| `msme_financial_data.csv` | The generated dataset (12,000 synthetic MSMEs) |
| `app.py` | FastAPI backend — this is what Render actually runs |
| `model.pkl` | The winning trained model (Gradient Boosting, R²≈0.70) |
| `scaler.pkl`, `sector_encoder.pkl`, `feature_names.pkl` | Supporting artifacts `app.py` needs to preprocess input |
| `metrics.json` | Accuracy comparison across all 5 models tried |
| `requirements.txt` | Exact pinned dependencies for Render |
| `render.yaml` | Optional blueprint so Render auto-fills the config |

**Model chosen:** Gradient Boosting Regressor beat Ridge Regression,
Random Forest, XGBoost, and a Deep Neural Network (MLP) on held-out
test data:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Ridge Regression | 0.662 | 3.83 | 4.80 |
| Random Forest | 0.640 | 3.97 | 4.96 |
| **Gradient Boosting (winner)** | **0.698** | **3.63** | **4.54** |
| XGBoost | 0.694 | 3.66 | 4.57 |
| Deep Neural Network (MLP) | 0.687 | 3.68 | 4.63 |

The dataset/model files are already generated and included — you do
**not** need to re-run `train_model.py` before deploying. It's included
so you can regenerate or retrain later (e.g. once you swap in real data).

## Deploy on Render (phone-only, no laptop needed)

**Step 1 — Get these files into a GitHub repo**
1. Go to github.com on your phone browser, create a new repository (e.g. `msme-financial-health-api`).
2. Use "Add file → Upload files" and upload every file in this folder
   (`app.py`, `model.pkl`, `scaler.pkl`, `sector_encoder.pkl`,
   `feature_names.pkl`, `requirements.txt`, `render.yaml`, and optionally
   `train_model.py` + `msme_financial_data.csv` + `metrics.json`).
3. Commit directly to the `main` branch.

**Step 2 — Create the Render service**
1. Go to render.com → sign in → **New → Web Service**.
2. Connect your GitHub account and select the repo you just created.
3. Render should auto-detect `render.yaml` and pre-fill everything. If not, set manually:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
4. Click **Create Web Service**. First deploy takes 2–5 minutes.

**Step 3 — Test it**
Once deployed, Render gives you a URL like `https://msme-financial-health-api.onrender.com`.
- Visit `/docs` for interactive Swagger UI (test predictions right in the browser).
- Visit `/health` to confirm the model loaded correctly.

## Example request

```
POST /predict
Content-Type: application/json

{
  "sector": "grocery_retail",
  "business_vintage_months": 36,
  "monthly_upi_inflow": 250000,
  "monthly_upi_outflow": 180000,
  "upi_transaction_count": 620,
  "avg_transaction_value": 400,
  "upi_inflow_growth_rate": 8.5,
  "cash_flow_volatility": 0.18,
  "avg_bank_balance": 45000,
  "gst_monthly_turnover": 230000,
  "gst_filing_regularity": 0.9,
  "gst_late_filings_count": 1,
  "digital_invoice_count_monthly": 150,
  "on_time_payment_ratio": 0.85,
  "bounced_payment_count": 0,
  "existing_loan_emi_ratio": 0.15
}
```

Response:
```json
{
  "financial_health_score": 76.2,
  "risk_category": "Low Risk",
  "loan_recommendation": "Strong financial health score (76.2/100)...",
  "model_used": "GradientBoostingRegressor"
}
```

Valid `sector` values: `grocery_retail`, `restaurant_food`,
`textile_apparel`, `electronics_repair`, `manufacturing_small`,
`services_professional`, `pharmacy_medical`, `wholesale_trade`.

## Notes for your hackathon pitch

- The score is built from a weighted blend of 7 sub-signals (revenue
  strength, GST compliance, payment discipline, cash-flow stability,
  debt burden, business vintage, growth trend) plus non-linear
  interaction effects — which is exactly why a tree-based model
  (Gradient Boosting) captures it better than plain linear regression.
- For the real product, `train_model.py`'s `generate_dataset()` function
  is the place you'd swap synthetic data for a real pipeline pulling
  from an Account Aggregator (UPI), GSTN API, and invoicing platforms.
