from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Detection de fraude bancaire", layout="centered")
st.title("Detection de fraude bancaire")
st.caption("Le modele predit si une transaction est frauduleuse (classe 1) ou legitime (classe 0).")

api_url = st.text_input("URL de l'API", value=API_URL)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        hour_of_day = st.number_input("hour_of_day", 0, 23, 2)
        is_weekend = st.selectbox("is_weekend", [0, 1])
        is_night_transaction = st.selectbox("is_night_transaction", [0, 1], index=1)
        customer_age = st.number_input("customer_age", 18, 100, 29)
        credit_score = st.number_input("credit_score", 300, 850, 520)
        account_age_years = st.number_input("account_age_years", 0.0, 50.0, 1.2)
        account_balance = st.number_input("account_balance", 0.0, value=80.0)
        transaction_amount = st.number_input("transaction_amount", 0.0, value=4200.0)
        num_prev_transactions = st.number_input("num_prev_transactions", 0, value=12)
        transaction_freq_monthly = st.number_input("transaction_freq_monthly", 0, value=3)
    with col2:
        distance_from_home_km = st.number_input("distance_from_home_km", 0.0, value=320.0)
        time_since_last_txn_hrs = st.number_input("time_since_last_txn_hrs", 0.0, value=0.5)
        is_international = st.selectbox("is_international", [0, 1], index=1)
        failed_attempts = st.number_input("failed_attempts", 0, value=3)
        pin_changed_recently = st.selectbox("pin_changed_recently", [0, 1], index=1)
        country = st.selectbox("country", ["USA", "UK", "France", "Canada"], index=2)
        merchant_category = st.selectbox(
            "merchant_category", ["Grocery", "Electronics", "Utilities", "Clothing"], index=1
        )
        payment_method = st.selectbox(
            "payment_method", ["Credit Card", "Debit Card", "Crypto", "Cheque"]
        )
        device_type = st.selectbox("device_type", ["Mobile", "Desktop", "POS Terminal", "ATM"])

    submitted = st.form_submit_button("Predire")

if submitted:
    payload = {
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "is_night_transaction": is_night_transaction,
        "customer_age": customer_age,
        "credit_score": credit_score,
        "account_age_years": account_age_years,
        "account_balance": account_balance,
        "transaction_amount": transaction_amount,
        "num_prev_transactions": num_prev_transactions,
        "transaction_freq_monthly": transaction_freq_monthly,
        "distance_from_home_km": distance_from_home_km,
        "time_since_last_txn_hrs": time_since_last_txn_hrs,
        "is_international": is_international,
        "failed_attempts": failed_attempts,
        "pin_changed_recently": pin_changed_recently,
        "country": country,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "device_type": device_type,
    }
    try:
        with st.spinner("Prediction en cours..."):
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as exc:
        st.error(f"Appel a l'API impossible : {exc}")
    else:
        pred = int(result["prediction"])
        proba = float(result["probability"])
        label = "FRAUDE" if pred == 1 else "LEGITIME"
        col1, col2 = st.columns(2)
        col1.metric("Resultat", f"{label} (classe {pred})")
        col2.metric("Probabilite de fraude", f"{proba:.1%}")
        if pred == 1:
            st.error("Transaction frauduleuse detectee (classe 1)")
        else:
            st.success("Transaction legitime (classe 0)")
        st.caption("Classe 1 = transaction frauduleuse · Classe 0 = transaction legitime")
