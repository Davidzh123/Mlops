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
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/Davidzh123/Mlops")

st.set_page_config(page_title="FraudGuard", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
      :root { --brand:#0b3d91; --brand2:#1f6feb; --ok:#1a7f37; --ko:#b42318; }
      .block-container { padding-top: 1.5rem; }
      .fg-header {
        background: linear-gradient(100deg, #0b3d91 0%, #1f6feb 100%);
        color: #fff; padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1.2rem;
      }
      .fg-header h1 { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: .2px; }
      .fg-header p { margin: .35rem 0 0; font-size: .95rem; opacity: .9; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        background: #f1f5fb; border-radius: 8px 8px 0 0; padding: 8px 14px; font-weight: 600;
      }
      .stTabs [aria-selected="true"] { background: #0b3d91; color: #fff; }
      div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e6ebf3; border-radius: 10px; padding: 12px 14px;
      }
      .fg-badge-ok { color: var(--ok); font-weight: 700; }
      .fg-badge-ko { color: var(--ko); font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_health() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


with st.sidebar:
    st.markdown("### FraudGuard")
    st.caption("Détection de fraude bancaire — démonstrateur MLOps")
    st.divider()
    online = api_health()
    st.markdown(
        f"<span class='{'fg-badge-ok' if online else 'fg-badge-ko'}'>"
        f"{'Service en ligne' if online else 'Service indisponible'}</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.link_button("Code source (GitHub)", GITHUB_URL, use_container_width=True)
    st.link_button("MLflow", MLFLOW_UI_URL, use_container_width=True)
    st.link_button("Airflow", AIRFLOW_UI_URL, use_container_width=True)
    st.divider()
    st.caption("Auteur : David Zhou — 5IABD2")
    st.caption("Projet MLOps · ESGI / IABD")

st.markdown(
    """
    <div class="fg-header">
      <h1>FraudGuard — Détection de fraude bancaire</h1>
      <p>Évaluez le risque de fraude d'une transaction en temps réel, et explorez la performance
      et l'industrialisation du modèle.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_biz, tab_perf, tab_orch, tab_archi = st.tabs(
    [
        "Analyse de transaction",
        "Enjeu métier",
        "Performance du modèle",
        "Pipeline & orchestration",
        "Architecture technique",
    ]
)


# ============================================================================
# Analyse de transaction
# ============================================================================
with tab_pred:
    st.subheader("Analyser une transaction")
    st.caption("Renseignez les caractéristiques, le modèle estime le risque de fraude.")
    api_url = st.text_input("URL de l'API", value=API_URL, help="Point d'accès du service d'inférence")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Transaction**")
            transaction_amount = st.number_input("Montant (€)", 0.0, value=4200.0)
            hour_of_day = st.number_input("Heure (0-23)", 0, 23, 2)
            is_weekend = st.selectbox("Week-end", [0, 1])
            is_night_transaction = st.selectbox("Transaction de nuit", [0, 1], index=1)
            is_international = st.selectbox("International", [0, 1], index=1)
            distance_from_home_km = st.number_input("Distance domicile (km)", 0.0, value=320.0)
        with c2:
            st.markdown("**Client / compte**")
            customer_age = st.number_input("Âge du client", 18, 100, 29)
            credit_score = st.number_input("Score de crédit", 300, 850, 520)
            account_age_years = st.number_input("Ancienneté compte (ans)", 0.0, 50.0, 1.2)
            account_balance = st.number_input("Solde du compte (€)", 0.0, value=80.0)
            num_prev_transactions = st.number_input("Transactions passées", 0, value=12)
            transaction_freq_monthly = st.number_input("Fréquence mensuelle", 0, value=3)
        with c3:
            st.markdown("**Contexte / sécurité**")
            time_since_last_txn_hrs = st.number_input("Délai dernière txn (h)", 0.0, value=0.5)
            failed_attempts = st.number_input("Tentatives échouées", 0, value=3)
            pin_changed_recently = st.selectbox("PIN changé récemment", [0, 1], index=1)
            country = st.selectbox("Pays", ["USA", "UK", "France", "Canada"], index=2)
            merchant_category = st.selectbox(
                "Type de marchand", ["Grocery", "Electronics", "Utilities", "Clothing"], index=1
            )
            payment_method = st.selectbox(
                "Moyen de paiement", ["Credit Card", "Debit Card", "Crypto", "Cheque"]
            )
            device_type = st.selectbox("Appareil", ["Mobile", "Desktop", "POS Terminal", "ATM"])

        submitted = st.form_submit_button(
            "Évaluer le risque", type="primary", use_container_width=True
        )

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
            with st.spinner("Analyse en cours..."):
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Service d'inférence injoignable : {exc}")
        else:
            pred = int(result["prediction"])
            proba = float(result["probability"])
            c1, c2 = st.columns(2)
            c1.metric("Risque de fraude", f"{proba:.1%}")
            c2.metric("Décision (seuil 50%)", "FRAUDE" if pred == 1 else "LÉGITIME")
            if pred == 1:
                st.error("Recommandation : bloquer / vérifier — transaction à haut risque (classe 1).")
            else:
                st.success("Recommandation : autoriser — transaction à faible risque (classe 0).")
            st.caption("Classe 1 = fraude · Classe 0 = légitime · décision au seuil de 50 %.")


# ============================================================================
# Enjeu métier
# ============================================================================
with tab_biz:
    st.subheader("Pourquoi détecter la fraude ?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraude dans le jeu de données", "5,5 %")
    c2.metric("Transactions analysées", "1 000 000")
    c3.metric("Métrique de référence", "ROC AUC")
    st.markdown(
        """
La fraude bancaire coûte cher : pertes financières directes, litiges, atteinte à la confiance
des clients. Mais bloquer **à tort** une transaction légitime crée de la friction et fait fuir
les clients. L'enjeu est donc un **équilibre** :

- détecter un maximum de fraudes (limiter les pertes),
- sans multiplier les fausses alertes (préserver l'expérience client).

> Classe **1** = transaction frauduleuse (à bloquer) · Classe **0** = transaction légitime.
> Jeu de données **déséquilibré** (~5,5 % de fraude) : pilotage sur le **ROC AUC** et le **F1**,
> pas sur l'accuracy (trompeuse).
        """
    )
    st.subheader("Le jeu de données")
    st.markdown(
        """
[Bank Transaction Fraud Detection Dataset](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset)
(Kaggle, synthétique, 1 M transactions, licence CC0). Variables : montant, score de crédit,
solde, pays, type de marchand, tentatives échouées, distance, appareil...

Exclues (anti-fuite / bruit) : identifiants, dates brutes, `city` (cardinalité), `fraud_type`.
        """
    )


# ============================================================================
# Performance du modèle (MLflow)
# ============================================================================
with tab_perf:
    st.subheader("Performance & suivi des expériences")
    st.link_button("Ouvrir MLflow", MLFLOW_UI_URL)
    st.caption(
        "Trois familles de modèles (RandomForest, XGBoost, LightGBM) comparées et optimisées "
        "(GridSearch / Optuna). Chaque run est tracé dans MLflow."
    )
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
            columns={name_col: "modèle", "metrics.roc_auc": "roc_auc", "metrics.f1": "f1"}
        )
        st.dataframe(view, width="stretch")
        if "roc_auc" in view.columns and "modèle" in view.columns:
            st.caption("ROC AUC par run")
            st.bar_chart(view.set_index("modèle")["roc_auc"])
    else:
        st.info("Aucun run MLflow trouvé. Lancez `make train-models` ou `make train-optuna`.")


# ============================================================================
# Pipeline & orchestration (Airflow)
# ============================================================================
with tab_orch:
    st.subheader("Industrialisation : ré-entraînement automatisé")
    st.link_button("Ouvrir Airflow", AIRFLOW_UI_URL)
    st.markdown(
        """
Le modèle ne reste pas figé : un pipeline **Airflow** le **ré-entraîne automatiquement** et
**contrôle sa qualité** avant toute mise en service.

- **`model_retraining`** : `prepare_data → train → check_quality` (lundis 3h). Garde-fou : rejet
  si `roc_auc < 0.65`.
- **`daily_predictions`** : envoie chaque jour un lot de transactions à l'API (simulation de trafic).
        """
    )
    st.caption("Statut en direct (si Airflow est joignable)")
    try:
        with httpx.Client(
            base_url=AIRFLOW_UI_URL, auth=(AIRFLOW_USER, AIRFLOW_PASSWORD), timeout=5.0
        ) as client:
            dags = client.get("/api/v1/dags").json().get("dags", [])
            rows = []
            for d in dags:
                dag_id = d["dag_id"]
                dr = (
                    client.get(
                        f"/api/v1/dags/{dag_id}/dagRuns",
                        params={"order_by": "-start_date", "limit": 1},
                    )
                    .json()
                    .get("dag_runs", [])
                )
                rows.append(
                    {
                        "dag": dag_id,
                        "actif": not d["is_paused"],
                        "dernier_run": dr[0]["state"] if dr else "aucun",
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Aucun DAG trouvé dans Airflow.")
    except Exception:
        st.info("Airflow non joignable depuis le frontend — ouvrez l'interface via le bouton.")


# ============================================================================
# Architecture technique
# ============================================================================
with tab_archi:
    st.subheader("Architecture de la solution")
    st.markdown(
        """
```
Données --> Pré-traitement --> Entraînement --> MLflow (tracking + registry)
                                    |
                                    v
                           models/model.joblib
                                    |
                                    v
       Frontend Streamlit --> API FastAPI (/predict) <-- modèle servi
                                    ^
                  Airflow (ré-entraînement planifié + garde-fou qualité)
```

Industrialisation MLOps :
- Reproductibilité : `uv` + `uv.lock`, seed fixée, données figées.
- Suivi : MLflow (params, métriques, SHAP) + Model Registry.
- Service : API FastAPI conteneurisée (Docker).
- Orchestration : stack `docker-compose` (MLflow + API + frontend) ; Airflow pour le ré-entraînement.
- CI/CD : GitHub Actions (qualité + entraînement) puis build/push de l'image API sur GHCR.
- Déploiement : stack déployable sur un VPS (cloud).
        """
    )
    st.markdown(f"Code source complet : [{GITHUB_URL}]({GITHUB_URL})")
    st.caption("David Zhou — 5IABD2 · ESGI / IABD")
