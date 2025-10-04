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
MODEL_PATH = "C:/Users/fahad/OneDrive/Desktop/NASA/mlp_kepler_20251003_205847_min.pt"  # Path to your .pt model file
model_path     = r"C:/Users/fahad/OneDrive/Desktop/NASA/mlp_kepler_20251003_205847_min.pt"     # your current minimal ckpt
# NEW_ROWS_PATH  = r""         # CSV to predict on
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

# Load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

if os.path.exists(MODEL_PATH):
    try:
        try:
            model = torch.load(MODEL_PATH, map_location=device)
            model.eval()
            print(f"✓ Model loaded successfully on {device}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            model = None

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
else:
    print(f"⚠ Model file not found: {MODEL_PATH}")
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


def preprocess_data(df):
    """
    Preprocess the data for model inference
    Adjust this function based on your model's preprocessing requirements
    """
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Select only the expected columns in the correct order
    data = data[EXPECTED_COLUMNS]
    
    # Handle missing values - adjust strategy as needed
    data = data.fillna(0)  # Or use data.dropna() or other strategies
    
    # Handle categorical columns if any (e.g., koi_disposition, koi_fittype_LS+MCMC)
    # You may need to encode these based on your model's training
    categorical_cols = ['koi_disposition', 'koi_fittype_LS+MCMC']
    for col in categorical_cols:
        if col in data.columns:
            # Simple label encoding - adjust based on your model
            data[col] = pd.Categorical(data[col]).codes
    
    # Convert to numpy array
    data_array = data.values.astype(np.float32)
    
    return data_array


def run_inference(data):
    """
    Run model inference on preprocessed data
    Adjust based on your model's input/output format
    """
    if model is None:
        raise Exception("Model not loaded")
    
    # Convert to tensor
    input_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)
        
        # Convert predictions to numpy
        predictions = predictions.cpu().numpy()
    
    return predictions


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Exoplanet Prediction Backend is running!',
        'status': 'online',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'Home page',
            '/health': 'Health check',
            '/columns': 'Get expected columns (GET)',
            '/test-upload': 'Test file upload without model (POST)',
            '/predict': 'Single file prediction (POST) - requires model',
            '/batch-predict': 'Multiple file predictions (POST) - requires model'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
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
    try: 
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

        NEW_ROWS_PATH = temp_filepath
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

        import numpy as np  # Removed 'os' from here
        import torch
        from typing import List, Optional
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report
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

        # Load ckpt (may be minimal)
        ckpt = torch.load(model_path, map_location="cpu")
        bi = ckpt["build_info"]
        model_loaded = MLP(bi["input_dim"], bi["n_classes"])
        model_loaded.load_state_dict(ckpt["state_dict"])
        model_loaded.eval()
        device_pred = "cuda" if torch.cuda.is_available() else "cpu"
        model_loaded.to(device_pred)

        # Use preprocessor from ckpt if present; else rebuild from TRAIN CSV
        prep = ckpt.get("preprocessor", None)
        if prep is None:
            print("[INFO] Checkpoint has no preprocessor; restoring scaler/OHE from CSV.")
            prep = restore_preprocessor_from_training(NEW_ROWS_PATH, TARGET_COL, DROP_ID_COLS,
                                                    scaler=SCALER, seed=SEED, val_size=VAL_SIZE, test_size=TEST_SIZE)

        # Load inference CSV
        df_src = pd.read_csv(NEW_ROWS_PATH)
        print(f"[INFO] Loaded {len(df_src):,} rows from {NEW_ROWS_PATH}")

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
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        return jsonify({
            'error': str(e),
            'success': False
        }), 500
    # try:
    #     # Check if file is present
    #     if 'file' not in request.files:
    #         return jsonify({'error': 'No file provided'}), 400
        
    #     file = request.files['file']
        
    #     # Check if file is empty
    #     if file.filename == '':
    #         return jsonify({'error': 'Empty filename'}), 400
        
    #     # Check file type
    #     if not allowed_file(file.filename):
    #         return jsonify({
    #             'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
    #         }), 400
        
    #     # Read file
    #     df = read_file(file)
        
    #     # Validate columns
    #     is_valid, message = validate_columns(df)
    #     if not is_valid:
    #         return jsonify({'error': message}), 400
        
    #     # Store original data for response
    #     original_data = df.to_dict('records')
        
    #     # Preprocess data
    #     processed_data = preprocess_data(df)
        
    #     # Run inference
    #     predictions = run_inference(processed_data)
        
    #     # Format predictions based on your model output
    #     # Adjust this based on whether your model outputs classes, probabilities, etc.
    #     predictions_list = predictions.tolist()
        
    #     # Combine with original data
    #     results = []
    #     for i, row in enumerate(original_data):
    #         results.append({
    #             'input': row,
    #             'prediction': predictions_list[i]
    #         })
        
    #     return jsonify({
    #         'success': True,
    #         'count': df.shape[0],
    #         'predictions': ''
    #     })
    
    # except Exception as e:
    #     return jsonify({
    #         'error': str(e),
    #         'success': False
    #     }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple files
    Expects: Multiple CSV or Excel files
    Returns: JSON with predictions for each file
    """
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        all_results = []
        
        for file in files:
            if not allowed_file(file.filename):
                continue
            
            df = read_file(file)
            is_valid, message = validate_columns(df)
            
            if not is_valid:
                all_results.append({
                    'filename': file.filename,
                    'error': message,
                    'success': False
                })
                continue
            
            processed_data = preprocess_data(df)
            predictions = run_inference(processed_data)
            
            all_results.append({
                'filename': file.filename,
                'success': True,
                'count': len(predictions),
                'predictions': predictions.tolist()
            })
        
        return jsonify({
            'success': True,
            'results': all_results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Exoplanet Prediction Backend")
    print("=" * 60)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not loaded (TEST MODE)'}")
    print(f"Device: {device}")
    print("\nEndpoints:")
    print("  • http://localhost:5000/          - Home")
    print("  • http://localhost:5000/health    - Health check")
    print("  • http://localhost:5000/columns   - Get expected columns")
    print("  • http://localhost:5000/test-upload - Test file upload (no model)")
    if model:
        print("  • http://localhost:5000/predict   - Run predictions")
        print("  • http://localhost:5000/batch-predict - Batch predictions")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
