from flask import Flask, render_template, request, jsonify
import os
import numpy as np

# Import Keras directly (Keras 3.x)
import keras
from keras.preprocessing import image
# Import preprocess_input specifically for VGG16
from keras.applications.vgg16 import preprocess_input # <--- ADD THIS LINE

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration for your model ---
# Path to your trained model file. Make sure this path is correct relative to app.py
MODEL_PATH = 'models/brain_tumor_vgg16_transfer_learning_model2.h5'
# List of class names in the order your model predicts them.
CLASS_NAMES = ['glioma','meningioma','notumor','pituitary'] # Corrected based on your training output

# The target size (width, height) that your model expects for input images.
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Load your trained model globally when the Flask app starts.
try:
    # Use keras.models.load_model for Keras 3.x models
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails, to handle gracefully

# Define the route for the home page.
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for making predictions.
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        try:
            filename = "uploaded_image_" + str(np.random.randint(100000)) + ".jpg"
            filepath = os.path.join('static', filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply VGG16 specific preprocessing
            img_array = preprocess_input(img_array) # <--- ADD THIS LINE AND REMOVE THE OLD NORMALIZATION

            predictions = model.predict(img_array)[0]
            
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[predicted_class_index]) * 100

            all_probabilities = {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(predictions)}

            return jsonify({
                "prediction": predicted_class_name,
                "confidence": f"{confidence:.2f}%",
                "all_probabilities": all_probabilities,
                "image_url": f"/{filepath}"
            })

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"}), 500
        
    return jsonify({"error": "Something went wrong."}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
        
    app.run(debug=True)
