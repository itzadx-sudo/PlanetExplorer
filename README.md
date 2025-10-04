PlanetExplorer — Exoplanet Candidate Classifier

Team: Five Guys
Challenge: NASA Space Apps Challenge 2025 — A World Away: Hunting for Exoplanets with AI
Mission Dataset: Kepler KOI (Kepler Objects of Interest) — NASA Exoplanet Archive

Project Overview
PlanetExplorer is an AI-driven exoplanet classification system designed to help astronomers and citizen scientists identify likely exoplanets in large space telescope datasets. The system uses curated Kepler mission tabular data and a custom Multilayer Perceptron (MLP) neural network built in PyTorch to classify Kepler Objects of Interest (KOIs) into CONFIRMED planets, CANDIDATE planets, and FALSE POSITIVES.
Our solution transforms curated NASA data into an interpretable model that can support discovery and outreach while remaining efficient enough to run on laptops or cloud notebooks.

Features

Robust AI Model: Fully connected MLP with over 2.6M parameters and class-weighted loss to handle imbalanced data.

NASA-backed Inputs: 30 astrophysical and transit-fit features curated from the Kepler mission (0% missing data).

Fast and Reproducible: Deterministic training with fixed seeds and early stopping.

Interpretable Outputs: Predictions include probability scores and confidence levels (High/Medium/Low).

Easy to Use: Jupyter-based pipeline with optional lightweight Gradio interface for interactive predictions.

Extensible: Ready for alternative models such as gradient-boosted trees, random forests, or CNNs on raw light curves.

Dataset and Preprocessing
Sources: NASA Exoplanet Archive — Kepler Objects of Interest.
Files Used:

kepler_top_features_0pct_missing_train90.csv — 6,530 rows, curated training and validation set

kepler_top_features_0pct_missing_test10.csv — 726 rows, held-out test set

Classes:

CONFIRMED (approximately 36%)

CANDIDATE (approximately 20%)

FALSE POSITIVE (approximately 44%)

Feature Handling:

30 numeric features such as koi_depth, koi_ror, koi_model_snr

Identifier kepid dropped before training to avoid data leakage

StandardScaler applied for normalization

Label encoding used for target classes

Model Architecture
MLP Classifier (PyTorch):
Input (30) → Linear(2048) → ReLU → Linear(1024) → ReLU → Linear(512) → ReLU → Linear(3) → Softmax

Loss: Weighted Cross-Entropy
Optimizer: AdamW (learning rate 1e-6, weight decay 1e-6)
Batch Size: 96
Epochs: Up to 600 with early stopping patience of 40
Hardware: CPU and GPU compatible

Artifacts saved for reproducibility:

Minimal checkpoint: mlp_kepler_<timestamp>_min.pt

Full checkpoint: mlp_kepler_<timestamp>_full.pt

Preprocessing pipeline: scaler, label encoder, and feature order

Evaluation
Validation split: Stratified 85/15 from training pool
Metrics: Accuracy, Macro-F1, Precision/Recall per class, Confusion Matrix
Confidence Levels:

High: probability ≥ 0.90 and margin ≥ 0.30

Medium: probability ≥ 0.75 and margin ≥ 0.15

Low: otherwise

Weighted loss improved detection of the minority CANDIDATE class while maintaining strong CONFIRMED recall.

Quick Start

Environment Setup
Clone the repository and install requirements:
git clone https://github.com/
<your-repo>/PlanetExplorer.git
cd PlanetExplorer
pip install -r requirements.txt

Train or Load Model
Train using the prepared dataset:
python train.py --train-file kepler_top_features_0pct_missing_train90.csv
(or use the provided pre-trained mlp_kepler_<timestamp>_min.pt)

Run Inference
python predict.py --input your_data.csv --weights artifacts/mlp_kepler_min.pt
Generates predictions.csv with predicted class and probabilities.

Interactive Demo (optional)
python app.py
Launches a simple web interface for file upload and instant predictions.

Future Work

Integration of gradient boosted trees (XGBoost) for interpretability.

Incorporation of 1D-CNN light curve analysis for end-to-end detection.

Model ensembling for improved recall on borderline candidates.

Integration with NASA’s Exoplanet Archive API for automated updates.

License
MIT License — free to use and modify for educational and research purposes.

Acknowledgments

NASA Exoplanet Archive and Kepler Mission Data for publicly accessible curated datasets.

NASA Space Apps Challenge 2025 for inspiring global collaboration in space science and artificial intelligence.
