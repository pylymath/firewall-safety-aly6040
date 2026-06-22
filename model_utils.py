"""
model_utils.py
--------------
Loading and inference helpers for the Enterprise Sentinel Streamlit app.

- Loads the 6 non-BERT models + scaler from models/.
- Loads the fine-tuned DistilBERT (from the HF Hub repo ID, with a local
  fallback to the bundled distilbert_model/ folder).
- Picks the best non-BERT model by adversarial Recall from the comparison CSV.
- Provides a plain rule-based explanation panel (not a model).

All heavy loads are designed to be wrapped by @st.cache_resource in app.py.
"""

import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import torch

from feature_engineering import compute_features, features_to_vector, FEATURE_NAMES
from mlp_model import PromptMLP, INPUT_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
LOCAL_BERT_DIR = os.path.join(HERE, "distilbert_model")

# --- Hugging Face fine-tuned DistilBERT ------------------------------------ #
DISTILBERT_REPO_ID = "yp27/enterprise-sentinel-distilbert"
# The model repo is public, so no token is required. If you later make it
# private, set HF_TOKEN in the environment (or Streamlit secrets) and it will be
# picked up here. Never hardcode a token in committed code.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Jailbreak Q1 length threshold from Phase 1 EDA.
LONG_PROMPT_THRESHOLD = 900

# Instruction-override patterns for the rule layer. Regexes so we catch
# variants ("ignore your safety guidelines", "ignore all previous rules", etc.)
# rather than only exact phrases.
OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+|your\s+|the\s+|any\s+)?(previous\s+|above\s+|prior\s+)?"
    r"(instructions?|guidelines?|rules?|prompts?|safety|restrictions?|filters?)",
    r"disregard\s+(all\s+|the\s+|your\s+|any\s+)?(above|previous|prior|instructions?|guidelines?|rules?)",
    r"forget\s+(everything|all|your|the|previous|prior)\b",
    r"you\s+are\s+now\b",
    r"pretend\s+(you\s+are|to\s+be|you'?re)\b",
    r"act\s+as\s+(if|though|an?\b)",
    r"(enable|enter|activate)\s+(developer|dan|jailbreak)\s+mode",
    r"\bdan\s+mode\s+(enabled|on)\b",
    r"no\s+(restrictions?|filters?|rules?|limits?|content\s+policy|guidelines?)",
    r"without\s+(any\s+)?(filter|restriction|censorship|limitation)",
    r"do\s+anything\s+now",
    r"bypass\s+(your\s+|the\s+|any\s+)?(safety|filter|restriction|guideline)",
]
_OVERRIDE_RE = [re.compile(p, re.IGNORECASE) for p in OVERRIDE_PATTERNS]

# Friendly filename map for the 6 non-BERT models.
NONBERT_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "PyTorch MLP": "mlp_state_dict.pt",
}
SCALED_MODELS = {"Logistic Regression", "PyTorch MLP"}


def log(msg):
    print(f"[model_utils] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Comparison CSV helpers
# --------------------------------------------------------------------------- #
def load_comparison() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, "seven_model_comparison.csv")
    return pd.read_csv(path)


def best_nonbert_by_recall(comparison: pd.DataFrame) -> str:
    """Return the name of the highest-adversarial-Recall non-BERT model."""
    non_bert = comparison[~comparison["Model"].str.contains("DistilBERT", case=False)]
    non_bert = non_bert[non_bert["Model"].isin(NONBERT_FILES.keys())]
    best = non_bert.sort_values("Recall", ascending=False).iloc[0]
    return str(best["Model"])


# --------------------------------------------------------------------------- #
# Non-BERT model loading
# --------------------------------------------------------------------------- #
def load_nonbert_models():
    """Load scaler + all available non-BERT models. Returns (scaler, dict)."""
    t0 = time.time()
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    models = {}
    for name, fname in NONBERT_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            log(f"  (missing {fname}; skipping {name})")
            continue
        if name == "PyTorch MLP":
            mlp = PromptMLP(INPUT_DIM)
            mlp.load_state_dict(torch.load(path, map_location="cpu"))
            mlp.eval()
            models[name] = mlp
        else:
            models[name] = joblib.load(path)
    log(f"Loaded {len(models)} non-BERT models + scaler in {time.time()-t0:.2f}s")
    return scaler, models


def predict_nonbert(name, model, scaler, text):
    """Return (label:int, proba_adversarial:float) for one non-BERT model."""
    X = features_to_vector(text)
    if name in SCALED_MODELS:
        X = scaler.transform(X)
    if name == "PyTorch MLP":
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1)[0, 1].item()
    else:
        proba = float(model.predict_proba(X)[0, 1])
    return int(proba >= 0.5), proba


# --------------------------------------------------------------------------- #
# DistilBERT loading
# --------------------------------------------------------------------------- #
def load_distilbert():
    """Load tokenizer + model. Try HF Hub first, fall back to local folder."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    t0 = time.time()
    source = None
    tok = mdl = None

    # 1) Hugging Face Hub
    try:
        log(f"Loading DistilBERT from HF Hub: {DISTILBERT_REPO_ID}")
        tok = AutoTokenizer.from_pretrained(DISTILBERT_REPO_ID, token=HF_TOKEN)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            DISTILBERT_REPO_ID, token=HF_TOKEN
        )
        source = f"HF Hub ({DISTILBERT_REPO_ID})"
    except Exception as e:
        log(f"  HF Hub load failed ({e}); trying local distilbert_model/ ...")

    # 2) Local bundled folder fallback
    if mdl is None and os.path.isdir(LOCAL_BERT_DIR):
        try:
            tok = AutoTokenizer.from_pretrained(LOCAL_BERT_DIR)
            mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_BERT_DIR)
            source = f"local folder ({LOCAL_BERT_DIR})"
        except Exception as e:
            log(f"  Local load failed too: {e}")

    if mdl is None:
        raise RuntimeError("Could not load DistilBERT from HF Hub or local folder.")

    mdl.eval()
    log(f"DistilBERT loaded from {source} in {time.time()-t0:.2f}s")
    log(f"  id2label: {getattr(mdl.config, 'id2label', None)} "
        f"-> adversarial index = {adversarial_label_index(mdl)}")
    return tok, mdl, source


def adversarial_label_index(model) -> int:
    """Determine which output index is the adversarial class.

    Prefers the model's own id2label mapping (a properly retrained model should
    label its classes, e.g. {0: 'normal', 1: 'adversarial'}). Falls back to
    index 1, the conventional positive class for this binary fine-tune.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        lab = str(label).lower()
        if any(k in lab for k in ("adversar", "jailbreak", "malicious", "attack",
                                  "unsafe", "positive")):
            return int(idx)
    return 1 if int(getattr(model.config, "num_labels", 2)) > 1 else 0


def predict_distilbert(tokenizer, model, text):
    """Return (label:int, proba_adversarial:float)."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    adv_idx = adversarial_label_index(model)
    adv_idx = min(adv_idx, probs.shape[0] - 1)
    proba = probs[adv_idx].item()
    return int(proba >= 0.5), proba


# --------------------------------------------------------------------------- #
# Rule-based explanation panel (plain logic, not a model)
# --------------------------------------------------------------------------- #
def rule_based_signals(text):
    """Return a list of (flagged:bool, label, detail) signals for the panel."""
    feats = compute_features(text)
    lowered = text.lower()

    signals = []

    long_prompt = feats["prompt_length"] > LONG_PROMPT_THRESHOLD
    signals.append((
        long_prompt,
        "Unusually long prompt",
        f"{feats['prompt_length']} chars "
        f"({'>' if long_prompt else '≤'} {LONG_PROMPT_THRESHOLD} jailbreak-Q1 threshold)",
    ))

    high_punct = feats["punctuation_density"] > 0.05
    signals.append((
        high_punct,
        "High punctuation density",
        f"{feats['punctuation_density']:.3f} punctuation ratio",
    ))

    found = detect_override(text)
    signals.append((
        len(found) > 0,
        "Instruction-override phrasing",
        ("matched: " + ", ".join(f'"{p}"' for p in found)) if found
        else "no override phrases found",
    ))

    high_upper = feats["uppercase_ratio"] > 0.30
    signals.append((
        high_upper,
        "Excessive uppercase",
        f"{feats['uppercase_ratio']:.2%} of characters uppercase",
    ))

    return signals


def detect_override(text):
    """Return the list of instruction-override snippets found in the text."""
    hits = []
    for rx in _OVERRIDE_RE:
        m = rx.search(text or "")
        if m:
            hits.append(m.group(0).strip())
    return hits


def firewall_decision(bert_label, override_hits, sim_tier=None):
    """Combine three layers (defense in depth) into one decision.

    Blocks if ANY of these fire:
      - DistilBERT flags the prompt as adversarial, OR
      - the rule layer detects instruction-override phrasing, OR
      - the similarity layer finds a close match to a known jailbreak (BLOCK tier).
    A similarity "SUSPICIOUS" tier does not block on its own but is noted.
    Returns (blocked: bool, reason: str).
    """
    reasons = []
    block = False
    if bert_label == 1:
        block = True
        reasons.append("DistilBERT flagged it as adversarial")
    if override_hits:
        block = True
        reasons.append("the rule layer caught override phrasing")
    if sim_tier == "BLOCK":
        block = True
        reasons.append("it closely matches a known jailbreak (similarity layer)")

    if block:
        return True, "; ".join(reasons)
    if sim_tier == "SUSPICIOUS":
        return False, "allowed, but it resembles known jailbreaks - worth monitoring"
    return False, "none of the three layers flagged it"
