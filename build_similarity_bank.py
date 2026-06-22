"""
build_similarity_bank.py
------------------------
Builds the cached embedding bank used by similarity_layer.py, and calibrates the
tiered thresholds on a held-out split so they are grounded in the data rather
than guessed.

Run once during setup:  python build_similarity_bank.py
"""

import numpy as np
import pandas as pd

import similarity_layer as sl
from similarity_layer import get_encoder, _embed, _first_user_texts, RAW_JAILBREAKS


def calibrate():
    enc = get_encoder()
    jb = pd.read_csv(RAW_JAILBREAKS, usecols=["prompt"]).dropna()
    jb_texts = jb["prompt"].astype(str).str.strip()
    jb_texts = jb_texts[jb_texts.str.len() > 0].tolist()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(jb_texts))
    cut = int(0.8 * len(idx))
    train_jb = [jb_texts[i] for i in idx[:cut]]
    test_jb = [jb_texts[i] for i in idx[cut:]]

    bn_texts = _first_user_texts(2000)

    print(f"bank(train jb)={len(train_jb)}  test jb={len(test_jb)}  benign={len(bn_texts)}")
    bank = _embed(train_jb, enc)
    test_jb_emb = _embed(test_jb, enc)
    bn_emb = _embed(bn_texts, enc)

    # closest-jailbreak similarity for held-out jailbreaks vs benign
    jb_max = (test_jb_emb @ bank.T).max(axis=1)
    bn_max = (bn_emb @ bank.T).max(axis=1)

    def pct(a, p):
        return float(np.percentile(a, p))

    print("\nHeld-out JAILBREAK closest-sim:  "
          f"p5={pct(jb_max,5):.3f} p25={pct(jb_max,25):.3f} "
          f"median={pct(jb_max,50):.3f} p75={pct(jb_max,75):.3f}")
    print("BENIGN closest-sim:              "
          f"median={pct(bn_max,50):.3f} p90={pct(bn_max,90):.3f} "
          f"p99={pct(bn_max,99):.3f} max={bn_max.max():.3f}")

    # Sweep thresholds: jailbreak recall vs benign false-positive rate
    print("\n thr   jb_recall  benign_FPR")
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]:
        rec = float((jb_max >= thr).mean())
        fpr = float((bn_max >= thr).mean())
        print(f" {thr:.2f}    {rec:6.1%}     {fpr:6.1%}")


if __name__ == "__main__":
    print("=== Calibration (held-out split) ===")
    calibrate()
    print("\n=== Building full cached bank ===")
    sl.build_bank(save=True)
    print("DONE.")
