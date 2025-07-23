import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import sys
import io
from flask_cors import CORS
import cv2
from dataset_utils import check_model_label_orientation

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Global variable to store model label orientation
INVERT_PREDICTIONS = False

# Load the modelpy
try:
    # Use relative path for better portability
    model_path = os.path.join(os.path.dirname(__file__), 'converted_savedmodel', 'model.savedmodel')
    print(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    print("Model loaded successfully")
    
    # Load labels from labels.txt if it exists
    labels_path = os.path.join(os.path.dirname(__file__), 'converted_savedmodel', 'labels.txt')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            label_list = [line.strip() for line in f.readlines()]
            labels = {i: label for i, label in enumerate(label_list)}
            print(f"Loaded labels: {labels}")
    else:
        # Default labels as fallback
        labels = {0: "Brain Tumour", 1: "No Brain Tumour"}
        print("Using default labels")
    
    # Check model label orientation once at startup
    dataset_path = os.path.join(os.path.dirname(__file__), 'Brain-Mri')
    INVERT_PREDICTIONS = check_model_label_orientation(model_path, dataset_path)
    print(f"Model evaluation: Label inversion needed: {INVERT_PREDICTIONS}")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Don't crash the app, but log the error
    model = None
    infer = None
    labels = {0: "Brain Tumour", 1: "No Brain Tumour"}  # Default fallback

# Constants for allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'tiff', 'dicom', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image according to model's expected format
def preprocess_image(img):
    try:
        # Convert to RGB (to ensure 3 color channels)
        img = img.convert('RGB')
        
        # Enhance contrast and normalize using histogram equalization via OpenCV
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Resize the image for the model (224x224 is standard for many models)
        enhanced_img = cv2.resize(enhanced_img, (224, 224))
        
        # Standard normalization to [0, 1]
        img_array = enhanced_img / 255.0
        
        # Add batch dimension and ensure float32 type
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # If enhancement fails, fall back to simple preprocessing
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array

@app.route('/')
def index():
    return render_template('index.html')

# Route for analyzing the uploaded image
@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if model was loaded successfully
    if model is None or infer is None:
        return jsonify({'error': 'Model not loaded properly. Check server logs.'}), 500
        
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']

    # If no file is selected or the file is not valid
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or invalid file type'}), 400

    # Process the uploaded file
    try:
        # Try to open the file as an image
        img = Image.open(file.stream)
        img_array = preprocess_image(img)

        # Make prediction
        prediction = infer(tf.constant(img_array))
        output_key = list(prediction.keys())[0]
        prediction_array = prediction[output_key].numpy()
        
        # Print raw prediction values for debugging
        raw_values = prediction_array[0]
        print(f"Raw prediction values: {raw_values}")
        
        # Get the class index with highest confidence
        class_idx = np.argmax(raw_values)
        confidence = float(raw_values[class_idx] * 100)
        
        # If validation determined labels are inverted, flip the prediction
        if INVERT_PREDICTIONS:
            class_idx = 1 - class_idx  # Flip between 0 and 1
            confidence = float(raw_values[1 - class_idx] * 100)
        
        # Get the class label
        result = labels.get(class_idx, f"Class {class_idx}")
        
        # Format confidence
        confidence_str = f"{confidence:.1f}%"
        
        # Log the prediction
        print(f"Final Prediction: {result}, Confidence: {confidence_str}")
        
        # Return prediction result and confidence
        return jsonify({
            'result': result,
            'confidence': confidence_str,
            'class_idx': int(class_idx)
        })

    except Exception as e:
        # Log the error for server-side debugging
        print(f"Error in image processing or prediction: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

# Route to get dataset statistics
@app.route('/dataset-info', methods=['GET'])
def dataset_info():
    try:
        # Path to the dataset folders
        dataset_path = os.path.join(os.path.dirname(__file__), 'Brain-Mri')
        yes_dir = os.path.join(dataset_path, 'yes')
        no_dir = os.path.join(dataset_path, 'no')
        
        # Count number of images in each category
        yes_count = len([f for f in os.listdir(yes_dir) if os.path.isfile(os.path.join(yes_dir, f))])
        no_count = len([f for f in os.listdir(no_dir) if os.path.isfile(os.path.join(no_dir, f))])
        
        return jsonify({
            'tumor_images': yes_count,
            'no_tumor_images': no_count,
            'total_images': yes_count + no_count,
            'invert_predictions': INVERT_PREDICTIONS
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to test model performance
@app.route('/evaluate-model', methods=['GET'])
def evaluate_model():
    try:
        from dataset_utils import evaluate_model_on_dataset
        
        model_path = os.path.join(os.path.dirname(__file__), 'converted_savedmodel', 'model.savedmodel')
        dataset_path = os.path.join(os.path.dirname(__file__), 'Brain-Mri')
        
        results = evaluate_model_on_dataset(model_path, dataset_path)
        
        # Simplify results for JSON response
        simple_results = {
            'accuracy': float(results['accuracy']),
            'tumor_precision': float(results['classification_report']['Tumor']['precision']),
            'tumor_recall': float(results['classification_report']['Tumor']['recall']),
            'no_tumor_precision': float(results['classification_report']['No Tumor']['precision']),
            'no_tumor_recall': float(results['classification_report']['No Tumor']['recall']),
            'invert_labels': results['invert_labels']
        }
        
        return jsonify(simple_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
