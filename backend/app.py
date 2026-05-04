import glob
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.predict import load_model, predict

BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

DATASET_PATH = os.environ.get("DATASET_PATH", str(BASE_DIR / "data" / "processed" / "test.csv"))

model_path_env = os.environ.get("MODEL_PATH")
if model_path_env:
    MODEL_PATH = model_path_env
else:
    pt_files = glob.glob(str(BASE_DIR / "artifacts" / "mlp_*_min.pt"))
    MODEL_PATH = max(pt_files, key=os.path.getmtime) if pt_files else None

model_obj = None
preprocessor = None
class_names = None
device_str = "none"

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        model_obj, preprocessor, class_names, device_str = load_model(MODEL_PATH)
        print(f"Model loaded on {device_str}: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"No model found at: {MODEL_PATH}")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_obj is not None,
        "device": device_str,
        "dataset_ready": os.path.exists(DATASET_PATH),
    })


@app.route("/predict-dataset", methods=["GET"])
def predict_dataset():
    if model_obj is None:
        return jsonify({"error": "Model not loaded.", "success": False}), 503

    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": f"Dataset not found at {DATASET_PATH}.", "success": False}), 404

    try:
        df = pd.read_csv(DATASET_PATH)
        results_df = predict(model_obj, preprocessor, df, class_names, device_str)

        results = [
            {
                "row": int(i),
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
                "margin": float(row["margin"]),
                "confidence_level": row["confidence_level"],
            }
            for i, row in results_df.iterrows()
        ]

        return jsonify({
            "success": True,
            "count": len(results),
            "dataset": os.path.basename(DATASET_PATH),
            "predictions": results,
        })
    except Exception as e:
        print(f"Error in /predict-dataset: {traceback.format_exc()}")
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
