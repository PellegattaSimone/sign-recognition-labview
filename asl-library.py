import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# Configuration
MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Global variables
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
    if model is None or processor is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        model = SiglipForImageClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    return True

def classify_sign(image_array):
    """
    Predicts the sign language alphabet from a LabVIEW U32 Color Array.
    
    :param image_array: 2D NumPy array of U32 (0xXXRRGGBB)
    :return: String predicted character
    """
    global processor, model
    if model is None: init_model()

    # Ensure input is U32 for bit manipulation
    arr = np.array(image_array, dtype=np.uint32)

    # Extract R, G, B channels from LabVIEW's U32 format (0xXXRRGGBB)
    r = ((arr >> 16) & 0xFF).astype(np.uint8)
    g = ((arr >> 8) & 0xFF).astype(np.uint8)
    b = (arr & 0xFF).astype(np.uint8)
    
    # Stack into (Height, Width, 3) RGB image
    rgb_stack = np.stack([r, g, b], axis=-1)
    image = Image.fromarray(rgb_stack)
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
    
    return LABELS.get(str(predicted_id), "Unknown")

if __name__ == "__main__":
    init_model()
    print("Model ready for LabVIEW.")
