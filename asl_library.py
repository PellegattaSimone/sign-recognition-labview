import requests
import numpy as np
from PIL import Image
import io

# The URL of the 64-bit Flask server
SERVER_URL = "http://127.0.0.1:5000/predict"

def classify_sign(image_array):
    """
    Bridge function for LabVIEW 32-bit.
    Sends the image to the 64-bit server and returns the result.
    """
    try:
        # 1. Convert LabVIEW U32 array to a standard RGB Image
        arr = np.array(image_array, dtype=np.uint32)
        r = ((arr >> 16) & 0xFF).astype(np.uint8)
        g = ((arr >> 8) & 0xFF).astype(np.uint8)
        b = (arr & 0xFF).astype(np.uint8)
        image = Image.fromarray(np.stack([r, g, b], axis=-1))

        # 2. Save image to a memory buffer (PNG format)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        byte_data = buf.getvalue()

        # 3. Send to the 64-bit server
        response = requests.post(SERVER_URL, data=byte_data)
        
        if response.status_code == 200:
            return response.json().get("prediction", "Error")
        else:
            return f"Server Error: {response.status_code}"

    except Exception as e:
        return f"Bridge Error: {str(e)}"
