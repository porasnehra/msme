"""
MSME Financial Health Score - Data Generation + Model Training
================================================================
Generates a realistic synthetic dataset of MSME alternative-data signals
(UPI transactions, GST filings, cash flow behaviour) and trains several
regression models to predict a 0-100 "Financial Health Score".

Run this once locally (or in this sandbox) to produce:
    - msme_financial_data.csv   (the generated dataset)
    - model.pkl                 (best trained model)
    - scaler.pkl                (StandardScaler fit on training features)
    - sector_encoder.pkl        (dict mapping sector -> encoded value)
    - feature_names.pkl         (ordered list of feature columns the model expects)
    - metrics.json              (comparison of all models tried)

app.py (the FastAPI server) loads model.pkl / scaler.pkl / sector_encoder.pkl /
feature_names.pkl at startup, so these 4 files + app.py + requirements.txt
are everything you need to deploy on Render.
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# --------------------------------------------------------------------------
# We simulate the kind of alternative digital-footprint data a bank could
# pull (with consent) from UPI apps / GSTN / account aggregator rails for
# an MSME that has NO audited balance sheet.

N_SAMPLES = 12000

SECTORS = ["grocery_retail", "restaurant_food", "textile_apparel",
           "electronics_repair", "manufacturing_small", "services_professional",
           "pharmacy_medical", "wholesale_trade"]

# sector risk multipliers - some sectors are inherently more seasonal/volatile
SECTOR_RISK = {
    "grocery_retail": 0.05, "restaurant_food": -0.05, "textile_apparel": -0.10,
    "electronics_repair": -0.02, "manufacturing_small": 0.0,
    "services_professional": 0.08, "pharmacy_medical": 0.10,
    "wholesale_trade": 0.0,
}

def generate_dataset(n=N_SAMPLES, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n):
        sector = rng.choice(SECTORS)

        # Business vintage (how long they've been operating digitally)
        business_vintage_months = max(1, int(rng.gamma(shape=3.0, scale=14)))

        # Core UPI signals -----------------------------------------------
        # Base monthly UPI inflow (revenue proxy), log-normal to mimic real money data
        monthly_upi_inflow = float(np.clip(rng.lognormal(mean=11.2, sigma=0.9), 8000, 4_000_000))

        # Outflow is usually 55%-95% of inflow depending on margin/sector
        outflow_ratio = np.clip(rng.normal(0.75, 0.12), 0.4, 1.15)
        monthly_upi_outflow = monthly_upi_inflow * outflow_ratio

        # Number & size of transactions
        avg_transaction_value = float(np.clip(rng.lognormal(mean=6.0, sigma=0.7), 50, 20000))
        upi_transaction_count = max(5, int(monthly_upi_inflow / max(avg_transaction_value, 50)))

        # Growth trend over last 6 months (%), healthier biz trend upward
        upi_inflow_growth_rate = float(np.clip(rng.normal(4, 12), -60, 80))

        # Cash flow volatility = coefficient of variation of monthly inflow (lower = more stable)
        cash_flow_volatility = float(np.clip(rng.gamma(shape=2.0, scale=0.09), 0.02, 1.2))

        avg_bank_balance = float(np.clip(
            monthly_upi_inflow * rng.uniform(0.05, 0.35), 500, 1_500_000))

        # GST signals ------------------------------------------------------
        # GST turnover should roughly correlate with UPI inflow but with noise
        # (some cash-heavy leakage / rounding is normal & realistic)
        gst_monthly_turnover = float(np.clip(
            monthly_upi_inflow * rng.uniform(0.6, 1.15), 5000, 4_500_000))

        gst_filing_regularity = float(np.clip(rng.beta(a=6, b=2), 0, 1))  # fraction on-time filings
        gst_late_filings_count = int(np.clip(rng.poisson(lam=(1 - gst_filing_regularity) * 6), 0, 12))

        # Digital invoicing & payment discipline ---------------------------
        digital_invoice_count_monthly = max(0, int(rng.poisson(lam=upi_transaction_count * 0.3)))
        on_time_payment_ratio = float(np.clip(rng.beta(a=7, b=2), 0, 1))
        bounced_payment_count = int(np.clip(rng.poisson(lam=(1 - on_time_payment_ratio) * 5), 0, 15))

        # Existing debt burden
        existing_loan_emi_ratio = float(np.clip(rng.beta(a=2, b=6), 0, 1))  # EMI / monthly inflow

        # ------------------------------------------------------------------
        # TRUE (latent) financial health score - built as a weighted blend of
        # normalized 0-100 sub-scores (similar to how real alt-data credit
        # scoring engines work), so the final distribution stays realistic
        # (spread across the full 0-100 range) instead of saturating at 100.
        # ------------------------------------------------------------------
        revenue_sub = np.clip((np.log1p(monthly_upi_inflow) - 9) / (15.2 - 9) * 100, 0, 100)
        gst_sub = np.clip(gst_filing_regularity * 100 - gst_late_filings_count * 6, 0, 100)
        payment_sub = np.clip(on_time_payment_ratio * 100 - bounced_payment_count * 5, 0, 100)
        stability_sub = np.clip(100 - cash_flow_volatility * 110, 0, 100)
        debt_sub = np.clip(100 - existing_loan_emi_ratio * 120, 0, 100)
        vintage_sub = np.clip(business_vintage_months / 60 * 100, 0, 100)
        growth_sub = np.clip(50 + upi_inflow_growth_rate * 0.9, 0, 100)

        score = (
            revenue_sub * 0.18 +
            gst_sub * 0.20 +
            payment_sub * 0.17 +
            stability_sub * 0.15 +
            debt_sub * 0.15 +
            vintage_sub * 0.08 +
            growth_sub * 0.07
        )

        # Mild non-linear interactions (rewards/penalizes combinations of
        # weak/strong signals together) - gives tree/NN models real signal
        # to pick up beyond a simple linear blend.
        if gst_sub < 40 and stability_sub < 40:
            score -= 6.0          # compounding risk: irregular GST + volatile cash flow
        if payment_sub > 80 and revenue_sub > 70:
            score += 5.0          # reliable + strong revenue = extra confidence bonus
        if debt_sub < 30 and stability_sub < 30:
            score -= 5.0          # over-leveraged AND unstable is a red flag combo

        score += SECTOR_RISK[sector] * 25
        score += rng.normal(0, 4.5)  # irreducible noise

        score = float(np.clip(score, 0, 100))

        rows.append(dict(
            sector=sector,
            business_vintage_months=business_vintage_months,
            monthly_upi_inflow=round(monthly_upi_inflow, 2),
            monthly_upi_outflow=round(monthly_upi_outflow, 2),
            upi_transaction_count=upi_transaction_count,
            avg_transaction_value=round(avg_transaction_value, 2),
            upi_inflow_growth_rate=round(upi_inflow_growth_rate, 2),
            cash_flow_volatility=round(cash_flow_volatility, 4),
            avg_bank_balance=round(avg_bank_balance, 2),
            gst_monthly_turnover=round(gst_monthly_turnover, 2),
            gst_filing_regularity=round(gst_filing_regularity, 4),
            gst_late_filings_count=gst_late_filings_count,
            digital_invoice_count_monthly=digital_invoice_count_monthly,
            on_time_payment_ratio=round(on_time_payment_ratio, 4),
            bounced_payment_count=bounced_payment_count,
            existing_loan_emi_ratio=round(existing_loan_emi_ratio, 4),
            financial_health_score=round(score, 2),
        ))

    return pd.DataFrame(rows)


def risk_category(score):
    if score >= 70:
        return "Low Risk"
    elif score >= 45:
        return "Medium Risk"
    else:
        return "High Risk"


if __name__ == "__main__":
    print("Generating synthetic MSME dataset...")
    df = generate_dataset()
    df["risk_category"] = df["financial_health_score"].apply(risk_category)
    df.to_csv("msme_financial_data.csv", index=False)
    print(f"Saved dataset: msme_financial_data.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(df["risk_category"].value_counts())

    # ----------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ----------------------------------------------------------------------
    sector_encoder = {s: i for i, s in enumerate(SECTORS)}
    df["sector_encoded"] = df["sector"].map(sector_encoder)

    feature_names = [
        "sector_encoded",
        "business_vintage_months",
        "monthly_upi_inflow",
        "monthly_upi_outflow",
        "upi_transaction_count",
        "avg_transaction_value",
        "upi_inflow_growth_rate",
        "cash_flow_volatility",
        "avg_bank_balance",
        "gst_monthly_turnover",
        "gst_filing_regularity",
        "gst_late_filings_count",
        "digital_invoice_count_monthly",
        "on_time_payment_ratio",
        "bounced_payment_count",
        "existing_loan_emi_ratio",
    ]

    X = df[feature_names].values
    y = df["financial_health_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ----------------------------------------------------------------------
    # 3. TRAIN & COMPARE MULTIPLE MODELS
    # ----------------------------------------------------------------------
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Deep Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", alpha=1e-4, batch_size=64,
            learning_rate_init=0.001, max_iter=600,
            early_stopping=True, n_iter_no_change=20,
            random_state=RANDOM_STATE),
    }

    results = {}
    trained_models = {}

    print("\nTraining & evaluating models...\n" + "-" * 60)
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        results[name] = {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}
        trained_models[name] = model

        print(f"{name:28s} | R2: {r2:.4f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    # ----------------------------------------------------------------------
    # 4. SELECT BEST MODEL (highest R2 on held-out test set)
    # ----------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = trained_models[best_name]

    print("-" * 60)
    print(f"BEST MODEL: {best_name}  (R2={results[best_name]['r2']}, "
          f"RMSE={results[best_name]['rmse']})")

    # ----------------------------------------------------------------------
    # 5. SAVE ARTIFACTS FOR DEPLOYMENT
    # ----------------------------------------------------------------------
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(sector_encoder, "sector_encoder.pkl")
    joblib.dump(feature_names, "feature_names.pkl")

    with open("metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "n_samples": int(df.shape[0]),
            "features": feature_names,
        }, f, indent=2)

    print("\nSaved: model.pkl, scaler.pkl, sector_encoder.pkl, "
          "feature_names.pkl, metrics.json")
    print("Ready to deploy with app.py")
