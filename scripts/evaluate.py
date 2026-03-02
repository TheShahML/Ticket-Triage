from __future__ import annotations

"""Mini evaluation harness for local triage classification backends."""

import argparse
import json
import random
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embed import EmbeddingEngine, cosine_similarity

DATA_FILE = ROOT / "data" / "labeled_tickets.jsonl"
SEED = 7
Backend = Literal["tfidf", "sentence-transformers"]


def load_rows() -> List[dict]:
    """Load labeled ticket fixtures used for evaluation."""
    rows = []
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def split_data(rows: List[dict], ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
    """Split records into train/test using a fixed random seed."""
    rng = random.Random(SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]


def keyword_baseline(text: str) -> str:
    """Tiny rule baseline used as an interpretable comparison point."""
    lower = text.lower()
    if any(k in lower for k in ["charged", "invoice", "refund", "payment", "tax"]):
        return "Billing"
    if any(k in lower for k in ["login", "password", "account", "mfa", "sign-in"]):
        return "Account"
    if any(k in lower for k in ["error", "crash", "bug", "fails", "blank", "500"]):
        return "Bug"
    if any(k in lower for k in ["feature", "would like", "add", "request", "support sso"]):
        return "Feature"
    return "Other"


def knn_predict(
    train: List[dict],
    embedder: EmbeddingEngine,
    train_matrix: np.ndarray,
    test_text: str,
    top_k: int = 5,
) -> str:
    """Predict a category from nearest neighbors in embedding space."""
    q = embedder.embed_query(test_text)
    scores = cosine_similarity(q, train_matrix)
    idx = np.argsort(scores)[::-1][:top_k]

    votes: Dict[str, float] = defaultdict(float)
    for i in idx:
        votes[train[int(i)]["category"]] += max(float(scores[int(i)]), 0.0)
    return max(votes, key=votes.get)


def evaluate_backend(train: List[dict], test: List[dict], backend: Backend) -> Tuple[float, Dict[str, float]]:
    """Evaluate kNN accuracy for one embedding backend.

    Returns:
        Tuple of overall accuracy and per-category accuracy map.
    """
    embedder = EmbeddingEngine([row["text"] for row in train], force_local=True, local_backend=backend)
    train_matrix = embedder.embed_documents([row["text"] for row in train])

    knn_correct = 0

    per_category_total = Counter()
    per_category_correct = Counter()

    for row in test:
        gold = row["category"]
        pred_knn = knn_predict(train, embedder, train_matrix, row["text"], top_k=5)

        if pred_knn == gold:
            knn_correct += 1
            per_category_correct[gold] += 1

        per_category_total[gold] += 1

    total = len(test)
    overall = knn_correct / total
    per_category = {
        cat: per_category_correct[cat] / per_category_total[cat] for cat in sorted(per_category_total)
    }
    return overall, per_category


def _render_loading_bar(step: int, total: int, label: str) -> None:
    """Render a small terminal progress bar for backend preparation steps."""
    width = 24
    filled = int((step / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {step}/{total} {label}")


def _sentence_transformer_available() -> Tuple[bool, str]:
    """Perform a quick availability check before initializing sentence-transformers.

    This prevents long retry loops when network/model access is unavailable.
    """
    try:
        socket.getaddrinfo("huggingface.co", 443)
    except OSError:
        return False, "Cannot resolve huggingface.co (network/DNS unavailable)."

    try:
        with urlopen("https://huggingface.co", timeout=3):
            return True, ""
    except (URLError, TimeoutError):
        return False, "Cannot reach huggingface.co (model download unavailable)."


def evaluate_keyword_baseline(test: List[dict]) -> float:
    """Evaluate the keyword-only baseline accuracy on test set."""
    baseline_correct = 0
    for row in test:
        baseline_correct += keyword_baseline(row["text"]) == row["category"]
    return baseline_correct / len(test)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for backend selection."""
    parser = argparse.ArgumentParser(description="Evaluate ticket triage kNN classifiers.")
    parser.add_argument(
        "--backend",
        choices=["tfidf", "sentence-transformers", "both"],
        default="both",
        help="Which local embedding backend(s) to evaluate.",
    )
    return parser.parse_args()


def evaluate(selected_backend: str) -> None:
    """Run evaluation and print readable comparison metrics."""
    rows = load_rows()
    train, test = split_data(rows)
    if selected_backend == "both":
        backends: List[Backend] = ["tfidf", "sentence-transformers"]
    else:
        backends = [cast(Backend, selected_backend)]

    baseline_accuracy = evaluate_keyword_baseline(test)
    total = len(test)

    print("Ticket Triage Mini Evaluation")
    print(f"Train size: {len(train)} | Test size: {total} | Seed: {SEED}")
    print(f"Keyword accuracy:  {baseline_accuracy:.2%}")

    for backend in backends:
        print(f"\nkNN backend: {backend}")
        if backend == "sentence-transformers":
            _render_loading_bar(1, 3, "Checking model availability")
            ok, reason = _sentence_transformer_available()
            if not ok:
                _render_loading_bar(3, 3, "Skipped")
                print(f"Skipped sentence-transformers backend: {reason}")
                print("Tip: run once with internet access to cache the local model.")
                continue
            _render_loading_bar(2, 3, "Initializing sentence-transformers model")
            time.sleep(0.05)

        overall, per_category = evaluate_backend(train, test, backend)
        if backend == "sentence-transformers":
            _render_loading_bar(3, 3, "Model ready")
        print(f"kNN accuracy:      {overall:.2%}")
        print("Per-category kNN accuracy:")
        for cat, acc in per_category.items():
            print(f"  - {cat:<8} {acc:.2%}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.backend)
