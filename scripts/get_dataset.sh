#!/usr/bin/env bash
# ==============================================================================
# Telecharge le dataset Kaggle "Bank Transaction Fraud Detection" dans data/raw/
# ==============================================================================
# Usage :
#   bash scripts/get_dataset.sh
#
# Pre-requis : identifiants Kaggle (token API). Deux formats acceptes :
#
#   Nouveau format (token unique) :
#     mkdir -p ~/.kaggle
#     echo <TON_TOKEN> > ~/.kaggle/access_token
#     chmod 600 ~/.kaggle/access_token
#     (ou : export KAGGLE_API_TOKEN=<TON_TOKEN>)
#
#   Ancien format (kaggle.json username + key) :
#     mkdir -p ~/.kaggle
#     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
#     chmod 600 ~/.kaggle/kaggle.json
#
# Token a generer sur kaggle.com : photo de profil > Settings > API.
# Le CLI kaggle est lance via `uvx` (uv), aucune installation globale requise.
# ==============================================================================
set -euo pipefail

DATASET="nafiulislam490/bank-transaction-fraud-detection-dataset"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw"

# --- Verification des identifiants Kaggle -------------------------------------
if [ ! -f "${HOME}/.kaggle/access_token" ] \
   && [ ! -f "${HOME}/.kaggle/kaggle.json" ] \
   && [ -z "${KAGGLE_API_TOKEN:-}" ] \
   && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "[ERREUR] Identifiants Kaggle introuvables." >&2
  echo "  Genere un token : Kaggle > Settings > API, puis installe-le :" >&2
  echo "    mkdir -p ~/.kaggle" >&2
  echo "    echo <TON_TOKEN> > ~/.kaggle/access_token" >&2
  echo "    chmod 600 ~/.kaggle/access_token" >&2
  exit 1
fi

# --- Telechargement -----------------------------------------------------------
mkdir -p "$RAW_DIR"
echo ">> Telechargement de '$DATASET' dans $RAW_DIR ..."
uvx kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip

echo ""
echo "[OK] Dataset disponible dans data/raw/ :"
ls -lh "$RAW_DIR"
