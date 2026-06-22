# 🛡️ Enterprise Sentinel — Adversarial Prompt Firewall

A Streamlit app that classifies LLM prompts as **jailbreak attempts**
(`is_adversarial = 1`) or **normal user traffic** (`is_adversarial = 0`). It ties
together a graduate data-mining project that compared **seven models**: six
classical models on 14 engineered features, plus a fine-tuned **DistilBERT** that
reads the raw prompt text semantically.

The app has two sections:

1. **Live Prompt Classifier** — paste a prompt and see DistilBERT and the best
   statistical model judge it side by side (BLOCKED / ALLOWED + confidence),
   plus a rule-based explanation panel of red-flag signals.
2. **Interactive Report** — the project narrative: the problem, the data
   (1,874 jailbreak vs. 49,979 normal prompts; an 18.8× median-length gap), the
   14 features, all 7 models, the results table, and combined ROC / PR curves.

---

## Project layout

```
.
├── app.py                      # Streamlit app (two sections)
├── feature_engineering.py      # The 14 engineered features
├── mlp_model.py                # PyTorch MLP architecture (shared train/inference)
├── model_utils.py              # Model loading + inference + rule-based panel
├── train_models.py             # Part A + B: rebuild dataset, train 6 models, merge BERT
├── requirements.txt
├── README.md
├── distilbert_model/           # Local copy of the fine-tuned DistilBERT (fallback)
├── model_comparison_full.csv   # DistilBERT's metrics row (from Colab, Part 1)
├── fig_distilbert_loss_curve.png
└── models/                     # Created by train_models.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── random_forest.joblib
    ├── xgboost.joblib
    ├── lightgbm.joblib
    ├── mlp_state_dict.pt
    ├── scaler.joblib
    ├── six_model_comparison.csv
    ├── seven_model_comparison.csv
    ├── fig_roc_comparison_7models.png
    └── fig_pr_comparison_7models.png
```

---

## Run locally

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Train / rebuild the 6 non-BERT models and merge the DistilBERT row.
#    Run ONCE during setup. Downloads the two HF datasets and writes models/.
python train_models.py

# 3. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

> **Note on `train_models.py`:** it reconstructs the dataset from Hugging Face
> (Dataset A: `TrustAIRLab/in-the-wild-jailbreak-prompts`; Dataset B: a
> 49,979-row sample of `lmsys/lmsys-chat-1m`, seed 42), computes the 14 features,
> does a stratified 80/20 split, trains all six models with the project's
> hyperparameters, and saves everything to `models/`. If Hugging Face is
> unreachable it falls back to a small synthetic dataset so the pipeline still
> runs end-to-end.

### DistilBERT loading

At runtime the app loads the fine-tuned DistilBERT from the **Hugging Face Hub**:

```python
DISTILBERT_REPO_ID = "yp27/enterprise-sentinel-distilbert"
```

If the Hub is unreachable, it falls back to the bundled `distilbert_model/`
folder. The startup log prints exactly where each model was loaded from and how
long it took — handy for diagnosing issues live during a demo.

The app does **not** hardcode the adversarial output index — it reads the
model's own `id2label` mapping (`adversarial_label_index()` in `model_utils.py`)
and only falls back to index 1 if the labels are unnamed. So once you push a
corrected checkpoint, the app picks the right class automatically.

> ✅ **Resolved (data-leakage bug, now fixed).** The first published checkpoint
> appeared to score a perfect 1.00 in Colab but flagged ~99.5% of real prompts —
> including *"What is the capital of France?"* — as adversarial in live inference.
>
> **Root cause:** the Colab `extract_prompt()` helper used `ast.literal_eval()`
> on the lmsys `conversation` column, which is stored as a **NumPy-array repr**
> (dicts separated by `'\n '`, not `', '`). `ast.literal_eval` raised on every
> row and a bare `except: return val` silently returned the *entire raw
> conversation blob* as the "normal" prompt. The model therefore learned
> "starts with `[{'content'` → normal" vs. "clean text → adversarial" — trivially
> separable (hence the perfect-but-meaningless 1.00), but useless on real clean
> prompts. This is a textbook data-leakage / silent-parsing-failure trap.
>
> **Fix:** the conversation parser was corrected to extract the first *user*
> turn as clean text (regex on `'content': … 'role': 'user'`), a sanity assert
> was added to reject blob-shaped normals, and the model was retrained. The
> current checkpoint reports honest metrics — **Precision 0.846 / Recall 0.871 /
> AUC-ROC 0.993** on the adversarial class — and the pre-push smoke test
> separates benign (~0.000) from jailbreak (~0.98) correctly. The app reads
> `id2label = {0: "normal", 1: "adversarial"}` automatically, so no app code
> changes were needed.

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub. **Do not commit `distilbert_model/`** (~268 MB) —
   the app loads DistilBERT from the HF Hub at runtime, so the repo stays small
   and fits comfortably within Streamlit Cloud's git size limits.
2. On <https://share.streamlit.io>, point a new app at `app.py`.
3. Pre-build the `models/` folder by running `python train_models.py` locally
   and committing the (small) `.joblib` / `.pt` / `.csv` / `.png` artifacts, **or**
   let the app train on first launch via the cached loaders.
4. The HF token in `model_utils.py` is only needed if the repo is private; for a
   public repo you can remove it. Prefer storing it as a Streamlit **secret**
   (`st.secrets`) rather than hard-coding it for any real deployment.

---

## The 14 engineered features

`prompt_length`, `word_count`, `log_prompt_length`, `log_word_count`,
`token_density`, `avg_word_length`, `prompt_length_bucket`, `has_special_chars`,
`uppercase_ratio`, `sentence_count`, `exclamation_count`, `question_count`,
`unique_word_ratio`, `punctuation_density`.

See `feature_engineering.py` for exact definitions.

---

## Why adversarial Recall is the headline metric

A **missed jailbreak** (false negative) that slips through the firewall is
operationally far worse than a **false positive** (a blocked benign prompt). The
report tab pulls the actual numbers from `seven_model_comparison.csv` and states
plainly whether DistilBERT beat the tree models on adversarial Recall.
