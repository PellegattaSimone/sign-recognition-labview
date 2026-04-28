import os
import torch
import numpy as np
import io
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# Configuration
MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

app = Flask(__name__)

# Global variables for model
processor = None
model = None

LABELS = {
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
    "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
    "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
}

def init_model():
    """Initializes and loads the model into memory."""
    global processor, model
    if model is None:
        print(f"Loading model into memory from {CACHE_DIR}...")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        model = SiglipForImageClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        print("Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive image bytes and return the prediction.
    LabVIEW will POST the raw image data here.
    """
    try:
        # Read the raw image data from the POST request
        image_bytes = request.data
        if not image_bytes:
            return jsonify({"error": "No image data received"}), 400

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Inference
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_id = torch.argmax(outputs.logits, dim=1).item()
        
        result = LABELS.get(str(predicted_id), "Unknown")
        print(f"Prediction: {result}")
        return jsonify({"prediction": result})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ready", "model": MODEL_NAME})

if __name__ == "__main__":
    init_model()
    # Run server on localhost port 5000
    app.run(host='127.0.0.1', port=5000)
