"""
feature_engineering.py
----------------------
Computes the 14 engineered features used by the 6 non-BERT models in the
Enterprise Sentinel adversarial prompt firewall.

The feature order defined in FEATURE_NAMES is the single source of truth and is
used consistently for training, scaling, and inference.
"""

import re
import numpy as np
import pandas as pd

# Canonical feature order. Do not reorder — the scaler and models depend on it.
FEATURE_NAMES = [
    "prompt_length",
    "word_count",
    "log_prompt_length",
    "log_word_count",
    "token_density",
    "avg_word_length",
    "prompt_length_bucket",
    "has_special_chars",
    "uppercase_ratio",
    "sentence_count",
    "exclamation_count",
    "question_count",
    "unique_word_ratio",
    "punctuation_density",
]

# One-line descriptions used in the Interactive Report tab.
FEATURE_DESCRIPTIONS = {
    "prompt_length": "Total number of characters in the prompt.",
    "word_count": "Number of whitespace-separated tokens.",
    "log_prompt_length": "log(1 + prompt_length) — compresses the huge length range.",
    "log_word_count": "log(1 + word_count) — compresses the word-count range.",
    "token_density": "Words per character — how 'dense' the text is.",
    "avg_word_length": "Average characters per word.",
    "prompt_length_bucket": "Discrete length band (0–3) from EDA breakpoints.",
    "has_special_chars": "1 if any non-alphanumeric/space character is present.",
    "uppercase_ratio": "Fraction of characters that are uppercase.",
    "sentence_count": "Count of sentence terminators (. ! ?).",
    "exclamation_count": "Number of exclamation marks.",
    "question_count": "Number of question marks.",
    "unique_word_ratio": "Unique lowercased words / total words (lexical diversity).",
    "punctuation_density": "Punctuation characters (.,;:!?) / total characters.",
}


def compute_features(text: str) -> dict:
    """Compute the 14 engineered features for a single prompt string."""
    text = "" if text is None else str(text)

    prompt_length = len(text)
    words = text.split()
    word_count = len(words)

    log_prompt_length = float(np.log1p(prompt_length))
    log_word_count = float(np.log1p(word_count))
    token_density = word_count / (prompt_length + 1)
    avg_word_length = prompt_length / (word_count + 1)
    # np.digitize with the EDA breakpoints, minus 1 to make buckets 0-indexed.
    prompt_length_bucket = int(np.digitize(prompt_length, [0, 100, 500, 2000]) - 1)
    has_special_chars = int(bool(re.search(r"[^a-zA-Z0-9\s]", text)))
    uppercase_ratio = (
        sum(1 for c in text if c.isupper()) / prompt_length if prompt_length else 0.0
    )
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    exclamation_count = text.count("!")
    question_count = text.count("?")
    lowered = [w.lower() for w in words]
    unique_word_ratio = (len(set(lowered)) / word_count) if word_count else 0.0
    punctuation_density = (
        sum(1 for c in text if c in ".,;:!?") / prompt_length if prompt_length else 0.0
    )

    return {
        "prompt_length": prompt_length,
        "word_count": word_count,
        "log_prompt_length": log_prompt_length,
        "log_word_count": log_word_count,
        "token_density": token_density,
        "avg_word_length": avg_word_length,
        "prompt_length_bucket": prompt_length_bucket,
        "has_special_chars": has_special_chars,
        "uppercase_ratio": uppercase_ratio,
        "sentence_count": sentence_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "unique_word_ratio": unique_word_ratio,
        "punctuation_density": punctuation_density,
    }


def featurize_dataframe(texts) -> pd.DataFrame:
    """Vectorized helper: compute features for a Series/list of texts."""
    rows = [compute_features(t) for t in texts]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def features_to_vector(text: str) -> np.ndarray:
    """Return a single (1, 14) float array in canonical feature order."""
    feats = compute_features(text)
    return np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float64)
