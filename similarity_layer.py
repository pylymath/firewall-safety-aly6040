"""
similarity_layer.py
-------------------
A semantic retrieval layer that runs in parallel with the classifier.

Idea: embed the incoming prompt and compare it (cosine similarity) against a
bank of known jailbreak prompts. If it is very close to a known attack, that is
a strong signal even when the DistilBERT classifier misses it (e.g. short or
novel attacks). We also keep a bank of benign prompts and use the margin
(closest-jailbreak similarity minus closest-benign similarity) to avoid flagging
benign prompts that merely talk *about* jailbreaks.

This implements behaviours (a) and (b) from the design discussion:
  - tiered thresholds: BLOCK / SUSPICIOUS / CLEAR
  - the score feeds the firewall decision as an explicit signal (we do NOT
    mutate the text fed to BERT)

Encoder: sentence-transformers/all-MiniLM-L6-v2 (small, fast, CPU-friendly).
The embedding bank is built once and cached to models/similarity_bank.npz.
"""

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
BANK_PATH = os.path.join(MODELS_DIR, "similarity_bank.npz")
RAW_JAILBREAKS = os.path.join(HERE, "raw_adversarial_jailbreaks.csv")
RAW_CONVERSATIONS = os.path.join(HERE, "raw_clean_conversations.csv")

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# How many reference prompts to keep in each bank.
N_JAILBREAKS = 2071     # all of them
N_BENIGN = 3000         # a sample is enough for the margin

# Tiered decision thresholds on the closest-jailbreak cosine similarity.
# Calibrated on a held-out 80/20 jailbreak split (see build_similarity_bank.py):
#   thr 0.65 -> 80.7% attack recall at 1.1% benign false-positive rate
#   thr 0.50 -> 94.2% attack recall at 10.8% benign false-positive rate
# So we block at 0.65 (with a margin guard) and merely flag as suspicious at 0.50.
BLOCK_THRESHOLD = 0.65       # >= this -> block outright
SUSPICIOUS_THRESHOLD = 0.50  # >= this -> "probable", let the model decide
MARGIN_MIN = 0.05            # jailbreak sim must beat benign sim by this much


def log(msg):
    print(f"[similarity] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Encoder (loaded lazily, cached by the app via st.cache_resource)
# --------------------------------------------------------------------------- #
_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        _ENCODER = SentenceTransformer(ENCODER_NAME)
        log(f"Loaded encoder {ENCODER_NAME} in {time.time()-t0:.2f}s")
    return _ENCODER


def _embed(texts, encoder=None):
    enc = encoder or get_encoder()
    return enc.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=64,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Bank build / load
# --------------------------------------------------------------------------- #
def _first_user_texts(n):
    """Pull n benign first-user prompts from the conversations CSV."""
    from train_models import _extract_first_user
    texts = []
    reader = pd.read_csv(RAW_CONVERSATIONS, usecols=["conversation"], chunksize=50_000)
    for chunk in reader:
        for c in chunk["conversation"]:
            t = _extract_first_user(c)
            if t:
                texts.append(t)
            if len(texts) >= n:
                return texts
    return texts


def build_bank(save=True):
    """Build and (optionally) cache the jailbreak + benign embedding banks."""
    encoder = get_encoder()

    jb = pd.read_csv(RAW_JAILBREAKS, usecols=["prompt"]).dropna()
    jb_texts = jb["prompt"].astype(str).str.strip()
    jb_texts = jb_texts[jb_texts.str.len() > 0].head(N_JAILBREAKS).tolist()
    log(f"Embedding {len(jb_texts)} jailbreak prompts ...")
    jb_emb = _embed(jb_texts, encoder)

    log(f"Embedding up to {N_BENIGN} benign prompts ...")
    bn_texts = _first_user_texts(N_BENIGN)
    bn_emb = _embed(bn_texts, encoder)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.savez_compressed(
            BANK_PATH,
            jb_emb=jb_emb,
            bn_emb=bn_emb,
            jb_texts=np.array(jb_texts, dtype=object),
        )
        log(f"Saved similarity bank -> {BANK_PATH}")
    return jb_emb, bn_emb, jb_texts


def load_bank():
    """Load the cached bank, building it on first use if absent."""
    if not os.path.exists(BANK_PATH):
        log("No cached bank found; building it now (one-time) ...")
        return build_bank(save=True)
    data = np.load(BANK_PATH, allow_pickle=True)
    return data["jb_emb"], data["bn_emb"], list(data["jb_texts"])


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #
class SimilarityScorer:
    """Holds the encoder + banks; scores a prompt against known attacks."""

    def __init__(self):
        self.encoder = get_encoder()
        self.jb_emb, self.bn_emb, self.jb_texts = load_bank()

    def score(self, text):
        """Return a dict with the closest-jailbreak similarity, the nearest
        attack text, the benign similarity, the margin, and a tier label."""
        q = _embed([text], self.encoder)[0]  # normalized
        jb_sims = self.jb_emb @ q             # cosine (both normalized)
        bn_sims = self.bn_emb @ q
        jb_idx = int(np.argmax(jb_sims))
        max_jb = float(jb_sims[jb_idx])
        max_bn = float(np.max(bn_sims))
        margin = max_jb - max_bn

        if max_jb >= BLOCK_THRESHOLD and margin >= MARGIN_MIN:
            tier = "BLOCK"
        elif max_jb >= SUSPICIOUS_THRESHOLD and margin >= 0:
            tier = "SUSPICIOUS"
        else:
            tier = "CLEAR"

        return {
            "tier": tier,
            "max_jailbreak_sim": max_jb,
            "max_benign_sim": max_bn,
            "margin": margin,
            "nearest_jailbreak": self.jb_texts[jb_idx],
        }
