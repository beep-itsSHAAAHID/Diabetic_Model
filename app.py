import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to my Flask server!'

# Get the absolute path to the directory containing the Flask app
app_dir = os.path.abspath(os.path.dirname(__file__))

# Define the paths to the TensorFlow Lite model and label file
model_path = os.path.join(app_dir, 'model.tflite')
label_path = os.path.join(app_dir, 'labels.txt')  # Adjust the filename if necessary

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load the labels
with open(label_path, 'r') as f:
    labels = f.read().splitlines()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define image size expected by the model
input_shape = input_details[0]['shape'][1:3]

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to expected input shape
    image = image.resize(input_shape)
    # Normalize image
    image = np.array(image) / 255.0
    # Expand dimensions to match model input
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# Function to perform inference
def run_inference(image):
    # Preprocess image
    image = preprocess_image(image)
    
    # Convert image to UINT8 data type
    image = (image * 255).astype(np.uint8)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    # Convert image bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform inference
    result = run_inference(image)
    
    # Get the predicted class label
    predicted_label = labels[np.argmax(result)]
    
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
