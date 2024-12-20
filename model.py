from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
from dotenv import load_dotenv

def create_app() :
    app = Flask(__name__)
    CORS(app)

    # Load environment variables
    load_dotenv()

    # Const
    THRESHOLD = 0.85
    MODEL_PATH = 'brain_tumorV2.h5'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_FILE_SIZE = 5 * 1024 * 1024

    # Load the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = ['Glioma detected', 'Meningioma detected',
                    'Pituitary detected', 'No Tumor detected']
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
            # Return a tuple (None, error message) on failure
            return None, str(e)


    def predict_image(image_bytes):
        try:
            img_array = load_uploaded_image(image_bytes)
            # Check if an error occurred during image loading
            if isinstance(img_array, tuple):
                return {
                    "class": "Error",
                    "confidence": 0.0,
                    "error": img_array[1]
                }

            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)

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
                "confidence": float(probability),
                "error": None
            }
        except Exception as e:
            return {
                "class": "Error Predicting Image",
                "confidence": 0,
                "error": str(e)
            }


    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    @app.route('/api/model', methods=['POST'])
    def model_check():
        return (jsonify({'result': predict_image(request.data), "model_loaded": "brain_tumorV2.h5" in os.listdir()}))


    @app.route('/api/predict', methods=['POST'])
    def predict():
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        image_file = request.files['image']

        # Check if filename is empty
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400

        # Check file type
        if not allowed_file(image_file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Only PNG, JPG, JPEG files are allowed"
            }), 400

        # Check file size
        image_file.seek(0, os.SEEK_END)
        size = image_file.tell()
        if size > MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "error": "File size too large. Maximum size is 10MB"
            }), 400
        image_file.seek(0)

        try:
            image_bytes = image_file.read()
            result = predict_image(image_bytes)

            # Format response to match frontend expectations
            return jsonify({
                "success": True,
                "predicted": result["class"],
                "prediction": result["confidence"],
                "error": result["error"]
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": error.description or "File size too large. Maximum size is 10MB"
        }), 413


    # Configure maximum content length
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    return app

app = create_app()
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port)
