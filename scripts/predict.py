from __future__ import annotations

import argparse
import json
import logging

import httpx

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from mlproject.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_payloads(n: int = 5) -> list[dict]:
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = load_data().sample(n=n, random_state=0)[cols]
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    payloads = build_payloads(args.n)
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health")
        logger.info("GET /health -> %s %s", health.status_code, health.json())

        for i, payload in enumerate(payloads):
            response = client.post("/predict", json=payload)
            logger.info("POST /predict (#%d) -> %s %s", i, response.status_code, response.json())

        info = client.get("/model-info")
        logger.info("GET /model-info -> %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()
