# CLAUDE.md — Project Context & AI-Assistance Notes

This file gives context to Claude Code (the AI coding assistant) when working in
this repository, and documents how AI assistance was used on the project. It is
written so a reader can see exactly where the team led and where the assistant
supported.

## Project

**Enterprise Sentinel / AI Jailbreak Prompt Firewall** — a system that classifies
prompts sent to an LLM as adversarial (jailbreak attempts, label 1) or normal
traffic (label 0), and blocks the adversarial ones at the ingestion layer.

- **Course:** ALY 6040 — Data Mining
- **Team (Group 2):** Yash Patel, Emmanuel Ushindi, Kevin Cross
- **Deliverables:** the written report, the seven-model analysis, the fine-tuned
  DistilBERT model, and a Streamlit application implementing the layered firewall.

## Who did what

This section records the division of labor between the team and the AI assistant.

The **team** built the core data-science work:

- Framing the problem and the business question, and the EDA across Phases 1–3
  that found the length signal and the class imbalance.
- Building and training the seven models — the six classical models on the 14
  engineered features (Logistic Regression, Decision Tree, Random Forest,
  XGBoost, LightGBM, PyTorch MLP), including their configurations and metric
  choices.
- Fine-tuning DistilBERT in Colab: tokenization, the weighted-loss setup, the
  hyperparameters, and diagnosing and fixing the data-leakage bug.
- Designing and implementing the distance-based (cosine-similarity) retrieval
  layer: the embedding banks, the margin logic, and the calibrated thresholds.
- Interpreting all results and writing the report.

**Claude Code (the AI assistant)** handled the engineering and delivery around
that core, under the team's direction:

- Building the Streamlit web application that ties the layers together for the
  live demo, and deploying it to Streamlit Community Cloud.
- Wiring up the Hugging Face `pipeline` inference path against the published Hub
  model.
- Managing the GitHub repository and commits (history, `.gitignore`, dependency
  pinning for the cloud build).

The data-science decisions, the modeling, and the analysis are the team's; the
assistant focused on the application, the serving/inference plumbing, and the
repository.

## Tech stack & conventions

- **Python 3.12+**; the 14 engineered features live in `feature_engineering.py`.
- **Classical models** (`scikit-learn`, `xgboost`, `lightgbm`, a small PyTorch
  MLP) are trained in `train_models.py`; trained artifacts go in `models/`.
- **DistilBERT** is fine-tuned in Colab and published to the Hugging Face Hub at
  `yp27/enterprise-sentinel-distilbert`; the app loads it at runtime.
- **Firewall layers** live in `model_utils.py` (rule layer) and
  `similarity_layer.py` (cosine-similarity retrieval).
- **App**: `app.py` (Streamlit), deployed to Streamlit Community Cloud.
- All randomness uses `random_state = 42`. Do not commit secrets, tokens, the
  raw multi-GB CSVs, or the local model copy (see `.gitignore`).

## Working agreements for the assistant

- Match the existing code style and the report's writing voice.
- Never hardcode credentials; read tokens from the environment if needed.
- Prefer small, reviewable changes; explain trade-offs rather than deciding
  silently.
