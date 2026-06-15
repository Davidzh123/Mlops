# Détection de fraude bancaire — Projet MLOps

Pipeline MLOps de **classification binaire** appliqué à la détection de transactions
bancaires frauduleuses (projet du module d'orchestration Machine Learning, ESGI / IABD).

## Problématique

À partir des caractéristiques d'une transaction bancaire (montant, pays, type de marchand,
score de crédit, tentatives échouées, etc.), prédire si elle est **frauduleuse** (`1`) ou
**légitime** (`0`).

- Les `1` = transactions frauduleuses à détecter et bloquer.
- Les `0` = transactions normales.
- Enjeu métier : limiter la fraude tout en évitant de bloquer à tort les clients.

> Le jeu de données est **déséquilibré** (~5,5 % de fraude). La métrique suivie en priorité
> est donc le `roc_auc` plutôt que l'`accuracy`.

## Jeu de données

[Bank Transaction Fraud Detection Dataset](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset)
(Kaggle, synthétique, 1 000 000 transactions, licence CC0). Cible : `is_fraud`.

Colonnes **exclues** des features (pour éviter la fuite de données et le bruit) :
`transaction_id`, `customer_id` (identifiants), `transaction_date`, `transaction_time`
(bruts), `city` (cardinalité trop élevée) et `fraud_type` (n'existe que si la transaction
est déjà une fraude → fuite de cible).

## Stack technique

- Python 3.13, environnement géré par **uv** (`pyproject.toml` + `uv.lock`)
- scikit-learn, XGBoost, LightGBM pour les modèles
- MLflow (tracking + Model Registry), Optuna (optimisation), SHAP (explicabilité)
- FastAPI + uvicorn (API d'inférence), Streamlit (frontend de test)
- Docker / docker-compose, Airflow (orchestration), GitHub Actions (CI/CD)

## Installation

### 1. Environnement Python

L'environnement est géré par [uv](https://docs.astral.sh/uv/) à partir du `pyproject.toml`
(Python 3.13). Pour créer le venv et installer les dépendances :

```bash
uv sync --extra dev      # crée .venv et installe le projet + les outils de dev
```

### 2. Récupérer le jeu de données

Le CSV (~154 Mo) n'est **pas versionné**. Il se télécharge automatiquement via l'API Kaggle.

1. Générer un token Kaggle : kaggle.com → photo de profil → **Settings** → **API** →
   *Create New Token*.
2. L'installer (token unique, nouveau format) :
   ```bash
   mkdir -p ~/.kaggle
   echo "VOTRE_TOKEN" > ~/.kaggle/access_token
   chmod 600 ~/.kaggle/access_token
   ```
3. Lancer le script de téléchargement :
   ```bash
   bash scripts/get_dataset.sh
   ```
   Le dataset arrive dans `data/raw/bank_fraud.csv`.

> Le token Kaggle est **personnel** et ne doit jamais être committé : il reste dans
> `~/.kaggle/`, hors du dépôt. Chaque personne utilise son propre token avec le même script.
