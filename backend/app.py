from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MODEL_PATH = 'model.pt'  # Path to your .pt model file
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Expected columns from your specification
EXPECTED_COLUMNS = [
    'kepid', 'koi_disposition', 'koi_dicco_msky', 'koi_dikco_msky', 'koi_prad',
    'koi_smet_err2', 'koi_max_mult_ev', 'koi_model_snr', 'koi_steff_err1',
    'koi_smet_err1', 'koi_prad_err2', 'koi_steff_err2', 'koi_ror',
    'koi_prad_err1', 'koi_duration_err1', 'koi_duration_err2',
    'koi_fittype_LS+MCMC', 'koi_count', 'koi_fwm_sdec_err', 'koi_fwm_srao_err',
    'koi_fwm_sdeco_err', 'koi_srad_err1', 'koi_ror_err2', 'koi_dor',
    'koi_smass_err1', 'koi_fwm_stat_sig', 'koi_ror_err1', 'koi_fwm_sra_err',
    'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_depth', 'koi_time0_err1'
]

# Load model at startup
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if model is not None else 'N/A'
    })


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
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file
        df = read_file(file)
        
        # Validate columns
        is_valid, message = validate_columns(df)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Store original data for response
        original_data = df.to_dict('records')
        
        # Preprocess data
        processed_data = preprocess_data(df)
        
        # Run inference
        predictions = run_inference(processed_data)
        
        # Format predictions based on your model output
        # Adjust this based on whether your model outputs classes, probabilities, etc.
        predictions_list = predictions.tolist()
        
        # Combine with original data
        results = []
        for i, row in enumerate(original_data):
            results.append({
                'input': row,
                'prediction': predictions_list[i]
            })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


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


@app.route('/columns', methods=['GET'])
def get_expected_columns():
    """Return list of expected columns"""
    return jsonify({
        'columns': EXPECTED_COLUMNS,
        'count': len(EXPECTED_COLUMNS)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
