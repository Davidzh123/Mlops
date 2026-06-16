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

## Modélisation et suivi MLflow

Trois familles de modèles sont comparées : **RandomForest**, **XGBoost**, **LightGBM**. Le
déséquilibre des classes est géré (`class_weight="balanced"`, `scale_pos_weight`) et la
métrique optimisée est le `roc_auc`.

Deux stratégies d'optimisation des hyperparamètres :

| Script | Méthode | Description |
|--------|---------|-------------|
| `src/mlproject/train.py` | aucune | baseline (régression logistique), point de référence |
| `src/mlproject/train_models.py` | **GridSearchCV** | recherche exhaustive sur une grille fixe |
| `src/mlproject/train_optuna.py` | **Optuna (TPE)** | recherche bayésienne, plus efficace |

Deux modules utilitaires partagés :
- `src/mlproject/tracking.py` : configuration MLflow (expérience, tags) et log du dataset.
- `src/mlproject/evaluation.py` : graphique d'importance SHAP loggé comme artefact.

Chaque modèle est suivi dans **MLflow** (backend SQLite `mlflow.db`) : hyperparamètres,
métriques, matrice de confusion, rapport de classification, importance SHAP. Le meilleur
modèle est enregistré dans le **Model Registry** sous `fraude-bancaire-classifier`.

### Pipeline (via Makefile)

```bash
make install          # 1. environnement + dépendances (uv)
make train            # 2. baseline (régression logistique)
make train-models     # 3. comparaison GridSearchCV + MLflow
make train-optuna     # 4. optimisation Optuna + MLflow
make mlflow           # 5. interface MLflow sur http://127.0.0.1:5000
```

> Dans l'interface MLflow, ouvrir l'expérience `fraude-bancaire` puis l'onglet
> **Training runs** (l'onglet *Overview/Usage* concerne les traces GenAI et reste vide).

### Résultats

Comparaison des modèles (échantillon de test) :

![Comparaison des modèles](docs/metrics_comparison.png)

Matrices de confusion :

![Matrices de confusion](docs/confusion_matrices.png)

> Le `roc_auc` est la métrique de référence (données déséquilibrées). Optuna atteint
> `roc_auc ≈ 0.72`, au-dessus de la baseline (0.711). Les figures sont régénérables avec
> `uv run python scripts/make_report.py`.
