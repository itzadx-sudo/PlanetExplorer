from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:8000", "https://yourfrontend.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for .pt file in the same directory
MODEL_FILENAME = "mlp_kepler_20251003_205847_min.pt"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Expected columns from your specification
EXPECTED_COLUMNS = [
    'kepid', 'koi_disposition', 'koi_dicco_msky',
    'koi_dikco_msky', 'koi_prad', 'koi_smet_err2',
    'koi_max_mult_ev', 'koi_model_snr', 'koi_steff_err1',
    'koi_smet_err1', 'koi_prad_err2', 'koi_steff_err2',
    'koi_ror', 'koi_prad_err1', 'koi_duration_err1',
    'koi_duration_err2', 'koi_fittype_LS+MCMC', 'koi_count',
    'koi_fwm_sdec_err', 'koi_fwm_srao_err', 'koi_fwm_sdeco_err',
    'koi_srad_err1', 'koi_ror_err2', 'koi_dor',
    'koi_smass_err1', 'koi_fwm_stat_sig', 'koi_ror_err1',
    'koi_fwm_sra_err', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_depth', 'koi_time0_err1'
]

# Configuration for prediction
TARGET_COL = "koi_disposition"
DROP_ID_COLS = ["kepid"]
SCALER = "standard"
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15

KEEP_COLS = [
    "koi_dicco_msky","koi_dikco_msky","koi_prad","koi_smet_err2","koi_max_mult_ev","koi_model_snr",
    "koi_steff_err1","koi_smet_err1","koi_prad_err2","koi_steff_err2","koi_ror","koi_prad_err1",
    "koi_duration_err1","koi_duration_err2","koi_fittype_LS+MCMC","koi_count","koi_fwm_sdec_err",
    "koi_fwm_srao_err","koi_fwm_sdeco_err","koi_srad_err1","koi_ror_err2","koi_dor","koi_smass_err1",
    "koi_fwm_stat_sig","koi_ror_err1","koi_fwm_sra_err","koi_time0bk_err1","koi_time0bk_err2",
    "koi_depth","koi_time0_err1"
]

# Load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = None

print(f"Looking for model at: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print(f"✓ Model checkpoint loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        checkpoint = None
else:
    print(f"⚠ Model file not found: {MODEL_PATH}")
    print(f"  Please place '{MODEL_FILENAME}' in the same directory as this script")
    print(f"  Backend will run in TEST MODE (no predictions available)")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(file):
    """Read CSV or Excel file and return DataFrame"""
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext == 'csv':
            df = pd.read_csv(file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        return df
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


def validate_columns(df):
    """Validate that DataFrame has required columns"""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    return True, "All required columns present"


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Exoplanet Prediction Backend is running!',
        'status': 'online',
        'model_loaded': checkpoint is not None,
        'model_path': MODEL_PATH,
        'endpoints': {
            '/': 'Home page',
            '/health': 'Health check',
            '/columns': 'Get expected columns (GET)',
            '/test-upload': 'Test file upload without model (POST)',
            '/predict': 'Single file prediction (POST) - requires model',
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': checkpoint is not None,
        'device': str(device),
        'model_path': MODEL_PATH
    })


@app.route('/columns', methods=['GET'])
def get_expected_columns():
    """Return list of expected columns"""
    return jsonify({
        'columns': EXPECTED_COLUMNS,
        'count': len(EXPECTED_COLUMNS)
    })


@app.route('/test-upload', methods=['POST'])
def test_upload():
    """
    Test file upload without running model inference
    Good for testing before model is ready
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file
        df = read_file(file)
        
        # Validate columns
        is_valid, message = validate_columns(df)
        
        # Get file info
        present_cols = set(df.columns) & set(EXPECTED_COLUMNS)
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        
        return jsonify({
            'success': True,
            'message': 'File read successfully!',
            'file_info': {
                'filename': file.filename,
                'rows': len(df),
                'columns': len(df.columns)
            },
            'validation': {
                'all_required_present': is_valid,
                'present_columns': len(present_cols),
                'missing_columns': list(missing_cols) if missing_cols else []
            },
            'sample_data': df.head(3).to_dict('records')
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects: CSV or Excel file in form-data with key 'file'
    Returns: JSON with predictions
    """
    temp_filepath = None
    
    try:
        # Check if model is loaded
        if checkpoint is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file is in the correct location.',
                'model_path': MODEL_PATH,
                'success': False
            }), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_filepath)

        # Import required libraries
        from typing import List, Optional
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import torch.nn as nn

        # Minimal model definition (needed to load weights)
        class MLP(nn.Module):
            def __init__(self, input_dim: int, n_classes: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_classes),
                )
            def forward(self, x):
                if x.dim() > 2: x = x.view(x.size(0), -1)
                return self.net(x)

        # Minimal preprocessor used only if ckpt has none
        class TabularPreprocessor:
            def __init__(self, scaler: str = "standard"):
                self.scaler = scaler
                self.num_cols: List[str] = []
                self.cat_cols: List[str] = []
                self.ct: Optional[ColumnTransformer] = None
                self.label_encoder: Optional[LabelEncoder] = None
                self.feature_names_: List[str] = []
            
            def _make_num_pipeline(self):
                if self.scaler == "standard":
                    return Pipeline([("scaler", StandardScaler())])
                elif self.scaler == "robust":
                    from sklearn.preprocessing import RobustScaler
                    return Pipeline([("scaler", RobustScaler())])
                elif self.scaler == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    return Pipeline([("scaler", MinMaxScaler())])
                else:
                    return "passthrough"
            
            def fit(self, df_train: pd.DataFrame, target_col: str):
                feats = [c for c in df_train.columns if c != target_col]
                obj_like = df_train[feats].select_dtypes(include=["object","category","bool"]).columns.tolist()
                num_like = [c for c in feats if c not in obj_like]
                self.num_cols, self.cat_cols = num_like, obj_like
                self.ct = ColumnTransformer(
                    [("num", self._make_num_pipeline(), self.num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)],
                    remainder="drop", verbose_feature_names_out=False
                )
                self.ct.fit(df_train[feats])
                self.feature_names_ = list(self.ct.get_feature_names_out())
                return target_col
            
            def transform_X(self, df: pd.DataFrame) -> np.ndarray:
                return self.ct.transform(df[self.num_cols + self.cat_cols]).astype(np.float32)

        def restore_preprocessor_from_training(train_path: str, target_col: str, drop_id_cols: List[str],
            scaler: str = "standard", seed: int = 42,
            val_size: float = 0.15, test_size: float = 0.15) -> TabularPreprocessor:
            df = pd.read_csv(train_path, low_memory=False)
            df = df.drop(columns=[c for c in drop_id_cols if c in df.columns], errors="ignore")
            if target_col not in df.columns:
                raise ValueError(f"Target '{target_col}' not in CSV")
            y_all = df[target_col]
            X_all = df.drop(columns=[target_col])
            strat = y_all if y_all.value_counts().min() >= 2 and y_all.nunique() > 1 else None
            X_tr, X_tmp, y_tr, _ = train_test_split(X_all, y_all, test_size=(val_size+test_size),
                                                    stratify=strat, random_state=seed)
            prep = TabularPreprocessor(scaler=scaler)
            prep.fit(pd.concat([X_tr, y_tr], axis=1), target_col=target_col)
            return prep

        # Load checkpoint
        bi = checkpoint["build_info"]
        model_loaded = MLP(bi["input_dim"], bi["n_classes"])
        model_loaded.load_state_dict(checkpoint["state_dict"])
        model_loaded.eval()
        device_pred = "cuda" if torch.cuda.is_available() else "cpu"
        model_loaded.to(device_pred)

        # Use preprocessor from ckpt if present; else rebuild from TRAIN CSV
        prep = checkpoint.get("preprocessor", None)
        if prep is None:
            print("[INFO] Checkpoint has no preprocessor; restoring scaler/OHE from CSV.")
            prep = restore_preprocessor_from_training(temp_filepath, TARGET_COL, DROP_ID_COLS,
                                                    scaler=SCALER, seed=SEED, val_size=VAL_SIZE, test_size=TEST_SIZE)

        # Load inference CSV
        df_src = pd.read_csv(temp_filepath)
        print(f"[INFO] Loaded {len(df_src):,} rows from {temp_filepath}")

        # Build features
        df_new = df_src.drop(columns=(DROP_ID_COLS + [TARGET_COL]), errors="ignore")
        expected_raw = getattr(prep, "num_cols", []) + getattr(prep, "cat_cols", [])
        allowed_cols = sorted(set(KEEP_COLS).union(expected_raw))
        df_new = df_new.filter(allowed_cols, axis=1)
        for c in expected_raw:
            if c not in df_new.columns:
                df_new[c] = np.nan
        df_new = df_new[expected_raw]

        # Numeric cleanup & impute from scaler means if available
        num_cols = getattr(prep, "num_cols", [])
        if num_cols:
            df_new[num_cols] = df_new[num_cols].apply(pd.to_numeric, errors="coerce")
            scaler_pipe = prep.ct.named_transformers_.get("num", None)
            scaler_obj = getattr(getattr(scaler_pipe, "named_steps", {}), "get", lambda *_: None)("scaler")
            if scaler_obj is not None and hasattr(scaler_obj, "mean_"):
                means = pd.Series(scaler_obj.mean_, index=num_cols)
                df_new[num_cols] = df_new[num_cols].fillna(means)
            else:
                df_new[num_cols] = df_new[num_cols].fillna(0.0)

        # Predict
        X_new = prep.transform_X(df_new).astype(np.float32)
        with torch.no_grad():
            logits = model_loaded(torch.from_numpy(X_new).to(device_pred))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        pred_idx = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)
        class_names = bi["class_names"]
        pred_label = [class_names[i] for i in pred_idx]

        top2_idx = np.argsort(probs, axis=1)[:, -2]
        top2_prob = probs[np.arange(len(probs)), top2_idx]
        pred_margin = pred_conf - top2_prob

        def bucket(p, m):
            return "High" if (p>=0.90 and m>=0.30) else ("Medium" if (p>=0.75 and m>=0.15) else "Low")

        confidence_level = [bucket(p, m) for p, m in zip(pred_conf, pred_margin)]

        # Clean up temp file
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        # Return results to frontend
        results = []
        for i in range(len(pred_label)):
            results.append({
                'row': int(i),
                'prediction': pred_label[i],
                'confidence': float(pred_conf[i]),
                'margin': float(pred_margin[i]),
                'confidence_level': confidence_level[i]
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })
    
    except Exception as e:
        # Clean up temp file if it exists
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /predict: {error_trace}")
        
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Exoplanet Prediction Backend")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Status: {'✓ Loaded' if checkpoint else '✗ Not loaded (TEST MODE)'}")
    print(f"Device: {device}")
    print("\nEndpoints:")
    print("  • http://localhost:5000/          - Home")
    print("  • http://localhost:5000/health    - Health check")
    print("  • http://localhost:5000/columns   - Get expected columns")
    print("  • http://localhost:5000/test-upload - Test file upload (no model)")
    if checkpoint:
        print("  • http://localhost:5000/predict   - Run predictions")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
