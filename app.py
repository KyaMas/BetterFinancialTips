from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
import subprocess
from flask_cors import CORS, cross_origin
import base64
from openai import OpenAI
from ml_pro import category_features

client = OpenAI(
    api_key='follow instruction in the readme file to use API key'
)

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
        with open("static/images/review_wordcloud.png", "rb") as image_file:
            wordcloud_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        with open("static/images/mds_visualization.png", "rb") as image_file:
            mds_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Return an array of both images as base64 strings
        return jsonify({
            "wordcloud": f"data:image/png;base64,{wordcloud_base64}",
            "mds": f"data:image/png;base64,{mds_base64}"
        }), 200

    else:
        return jsonify({"error": "Invalid file type"}), 400
    
@app.route("/tips", methods=["GET", "POST"])
def generate_tip():
    if request.method == "POST":
        # Step 1: Retrieve MDS values from ml_pro.py
        try:
            ml_feature = category_features()  # Call the function to get the feature
        except Exception as e:
            return jsonify({"error": f"Failed to retrieve processed feature: {str(e)}"}), 500

        # Step 2: Get user input from the form
        user_prompt = request.form["user_input"]

        # Step 3: Use MDS feature and user input to generate the response
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.6,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a financial assistant designed to provide users with spending and budgeting tips based on MDS values on the following categories: "
                            f"Electronics, Clothing, Household, Entertainment, Necessities, Sports {ml_feature}. "
                            "Provide responses in a friendly, engaging tone. Ensure your reply is clear and easy to understand. "
                            "Output should not exceed 100 words."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )

            # Step 4: Return the generated message as a JSON response
            return jsonify({"message": response.choices[0].message.content}), 200

        except Exception as e:
            return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

    # If method is GET, return an error as it's not intended
    return jsonify({"error": "Invalid request method. Please use POST."}), 405



if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)