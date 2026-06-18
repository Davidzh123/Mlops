from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "fraude-bancaire")
AIRFLOW_UI_URL = os.environ.get("AIRFLOW_UI_URL", "http://localhost:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "airflow")

st.set_page_config(page_title="Detection de fraude bancaire", layout="centered")
st.title("Detection de fraude bancaire")
st.caption("Projet MLOps - classification binaire (classe 1 = fraude, classe 0 = legitime)")

tab_pred, tab_problem, tab_perf, tab_airflow, tab_archi = st.tabs(
    [
        "Prediction",
        "Problematique & dataset",
        "Performance du modele",
        "Orchestration (Airflow)",
        "Architecture",
    ]
)


# ============================================================================
# Onglet Prediction
# ============================================================================
with tab_pred:
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
            c1, c2 = st.columns(2)
            c1.metric("Resultat", f"{label} (classe {pred})")
            c2.metric("Probabilite de fraude", f"{proba:.1%}")
            if pred == 1:
                st.error("Transaction frauduleuse detectee (classe 1)")
            else:
                st.success("Transaction legitime (classe 0)")


# ============================================================================
# Onglet Problematique & dataset
# ============================================================================
with tab_problem:
    st.subheader("Problematique metier")
    st.markdown(
        """
- **Objectif** : detecter les **transactions bancaires frauduleuses** pour les bloquer ou les verifier.
- **Classe 1** = transaction frauduleuse · **Classe 0** = transaction legitime.
- **Enjeu** : limiter les pertes liees a la fraude **sans** bloquer a tort les clients legitimes.
- Jeu de donnees **desequilibre** (~5,5 % de fraude) -> on suit le **ROC AUC** plutot que l'accuracy.
        """
    )
    st.subheader("Jeu de donnees")
    st.markdown(
        """
[Bank Transaction Fraud Detection Dataset](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset)
(Kaggle, synthetique, 1 000 000 transactions, licence CC0). Cible : `is_fraud`.

**Features utilisees** : montant, score de credit, solde, age, distance, tentatives echouees,
pays, type de marchand, moyen de paiement, type d'appareil...

**Colonnes exclues** : identifiants, dates brutes, `city` (trop de valeurs) et `fraud_type`
(fuite de cible : n'existe que si la transaction est deja une fraude).
        """
    )


# ============================================================================
# Onglet Performance du modele (MLflow)
# ============================================================================
with tab_perf:
    st.subheader("Performance des modeles (runs MLflow)")
    st.link_button("Ouvrir l'interface MLflow", MLFLOW_UI_URL)

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
    except Exception as exc:
        runs = pd.DataFrame()
        st.warning(f"MLflow indisponible ({exc}).")

    if len(runs) and "metrics.roc_auc" in runs.columns:
        df = runs.dropna(subset=["metrics.roc_auc"]).copy()
        name_col = "tags.mlflow.runName"
        keep = [c for c in [name_col, "metrics.roc_auc", "metrics.f1"] if c in df.columns]
        view = df[keep].rename(
            columns={name_col: "modele", "metrics.roc_auc": "roc_auc", "metrics.f1": "f1"}
        )
        st.dataframe(view, width="stretch")
        if "roc_auc" in view.columns and "modele" in view.columns:
            st.caption("ROC AUC par run")
            st.bar_chart(view.set_index("modele")["roc_auc"])
        st.caption("Historique du ROC AUC (par date de run)")
        hist = df[["start_time", "metrics.roc_auc"]].set_index("start_time").sort_index()
        st.line_chart(hist)
    else:
        st.info(
            "Aucun run trouve. Lancez un entrainement suivi MLflow "
            "(`make train-models` ou `make train-optuna`)."
        )


# ============================================================================
# Onglet Orchestration (Airflow)
# ============================================================================
with tab_airflow:
    st.subheader("Orchestration du re-entrainement (Airflow)")
    st.link_button("Ouvrir l'interface Airflow", AIRFLOW_UI_URL)

    st.markdown(
        """
Deux DAGs planifient le pipeline :

- **`model_retraining`** : `prepare_data → train → check_quality` (lundis 3h). La metrique
  `roc_auc` passe de `train` a `check_quality` via **XCom** ; garde-fou si `roc_auc < 0.65`.
- **`daily_predictions`** : envoie chaque jour (10h) un lot de transactions a l'API `/predict`.
        """
    )

    st.caption("Statut en direct (si l'API Airflow est joignable)")
    try:
        with httpx.Client(
            base_url=AIRFLOW_UI_URL, auth=(AIRFLOW_USER, AIRFLOW_PASSWORD), timeout=5.0
        ) as client:
            dags = client.get("/api/v1/dags").json().get("dags", [])
            rows = []
            for d in dags:
                dag_id = d["dag_id"]
                runs = client.get(
                    f"/api/v1/dags/{dag_id}/dagRuns",
                    params={"order_by": "-start_date", "limit": 1},
                ).json().get("dag_runs", [])
                state = runs[0]["state"] if runs else "aucun run"
                rows.append(
                    {"dag": dag_id, "actif": not d["is_paused"], "dernier_run": state}
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Aucun DAG trouve dans Airflow.")
    except Exception:
        st.info(
            "Airflow non joignable depuis le frontend. Ouvrez l'interface Airflow "
            "(bouton ci-dessus) pour suivre les DAGs."
        )


# ============================================================================
# Onglet Architecture
# ============================================================================
with tab_archi:
    st.subheader("Architecture MLflow")
    st.markdown(
        """
**MLflow** trace et versionne les experiences :

- **Client** : le code Python (`train_models.py`, `train_optuna.py`) logue params, metriques et modeles.
- **Backend store** : params & metriques (ici **SQLite** `mlflow.db`).
- **Artifact store** : modeles & fichiers (matrice de confusion, SHAP...).
- **Model Registry** : versionne le meilleur modele (`fraude-bancaire-classifier`),
  cycle **None -> Staging -> Production**.
        """
    )
    st.subheader("Deploiement (Docker / VPS)")
    st.markdown(
        """
La stack est orchestree par **docker-compose** (services `mlflow`, `train`, `api`, `frontend`),
deployable telle quelle sur un **VPS** :

- `train` entraine et ecrit le modele dans un **volume partage**.
- `api` (FastAPI) lit ce modele et expose `/predict`.
- `frontend` (cette interface) appelle l'API via son **nom de service** (`http://api:8000`).
- **CI** (GitHub Actions) verifie la qualite ; **CD** construit l'image de l'API et la pousse
  sur le registre **GHCR**, prete a etre tiree par le VPS.
        """
    )
