from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
import subprocess
from flask_cors import CORS, cross_origin
import base64
import torch
import tensorflow as tf

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origins="http://127.0.0.1:5000", methods=["POST", "OPTIONS"])
def upload_file():
    print("TensorFlow installed:", tf.__version__)  # If this works, TensorFlow is installed.
    print("PyTorch installed:", torch.__version__)  # If this works, PyTorch is installed.
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({"message": "Preflight check passed"}), 200

    if 'file' not in request.files:
        return "File not found in request", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    print('Processing your data...')

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_data.csv')
        file.save(file_path)

        try:
            # Run the BERT script
            subprocess.run(["python", "BERT.py"], check=True)

            # Run the ML script
            subprocess.run(["python", "ml_pro.py"], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"An error occurred while running scripts: {e}"}), 500

        # Encode images as base64 strings
        with open("templates/review_wordcloud.png", "rb") as image_file:
            wordcloud_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        with open("templates/mds_visualization.png", "rb") as image_file:
            mds_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Return an array of both images as base64 strings
        return jsonify({
            "wordcloud": f"data:image/png;base64,{wordcloud_base64}",
            "mds": f"data:image/png;base64,{mds_base64}"
        }), 200

    else:
        return jsonify({"error": "Invalid file type"}), 400
    

@app.route('/submit_tip', methods=['POST'])
def submit_tip():
    user_input = request.form['user_input']
    # Handle the user input as needed
    return redirect(url_for('tips'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)