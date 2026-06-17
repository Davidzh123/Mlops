# ==============================================================================
# Projet de detection de fraude bancaire - Makefile
# ==============================================================================
# Pipeline pilotee par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna mlflow api predict frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: ## Telecharge le vrai jeu de donnees Kaggle dans data/raw/
	bash scripts/get_dataset.sh

train: ## Entraine la baseline -> models/model.joblib (C=.. MAX_ITER=..)
	$(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RandomForest / XGBoost / LightGBM (GridSearchCV) + MLflow (CV=.. SAMPLE=..)
	$(PYTHON) -m mlproject.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna + MLflow (N_TRIALS=.. CV=..)
	$(PYTHON) -m mlproject.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

mlflow: ## Demarre l'interface MLflow (backend SQLite) sur http://127.0.0.1:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $(MLFLOW_PORT)

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn mlproject.api:app --reload --host $(API_HOST) --port $(API_PORT)

predict: ## Appelle l'API /predict avec une transaction d'exemple (scripts/predict.py)
	$(PYTHON) scripts/predict.py

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	docker build -f docker/Dockerfile.train -t mlproject-train .

docker-run: ## Lance l'entrainement one-shot en conteneur (profil train)
	docker compose --profile train run --rm train

docker-up: ## Demarre la stack (mlflow, api, frontend)
	docker compose up -d --build mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	docker compose down


# ==============================================================================
# Qualite
# ==============================================================================

lint: ## Verifie le style (ruff)
	$(RUN) ruff check src scripts tests

format: ## Formate le code (ruff)
	$(RUN) ruff format src scripts tests

type: ## Verifie les types (mypy)
	$(RUN) mypy src/mlproject

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
