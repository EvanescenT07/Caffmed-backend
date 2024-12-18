import io
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf

# Constants
THRESHOLD = 0.85  # Update with your threshold
MODEL_PATH = 'brain_tumorV2.h5'  # Update with your model path

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")


def load_uploaded_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((244, 244))
        img_array = img.convert('RGB')
        img_array = np.array(img_array) / 255.0
        return img_array
    except Exception as e:
        return None, str(e)


def predict_image(image_bytes):
    try:
        img_array = load_uploaded_image(image_bytes)
        if isinstance(img_array, tuple):  # Error occurred
            return {
                "class": "Error",
                "confidence": 0.0,
                "error": img_array[1]
            }

        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        class_names = ['glioma detected', 'meningioma detected',
                       'pituitary detected', 'no tumor detected']

        probability = float(np.max(prediction))
        if probability < THRESHOLD:
            return {
                "class": "Not a Brain X-Ray image",
                "confidence": 0,
                "error": None
            }

        predicted_class = class_names[np.argmax(prediction)]
        return {
            "class": predicted_class,
            "confidence": probability,
            "error": None
        }
    except Exception as e:
        return {
            "class": "Error",
            "confidence": 0,
            "error": str(e)
        }

# Test usage of the model
def test_model():
    try:
        # Update with your test image path
        test_path = f"D:\Download/fde1484d-6647-4931-8867-a84cf27340c6.jpg"
        image_bytes = open(test_path, "rb").read()

        result = predict_image(image_bytes)
        print(f"Prediction: {result['class']}")
        print(
            f"Confidence: {result['confidence']} / {result['confidence']:.2%}")
        if result['error']:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_model()
