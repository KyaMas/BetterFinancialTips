from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
import subprocess
from flask_cors import CORS, cross_origin
import base64
from openai import OpenAI
from ml_pro import category_features

client = OpenAI(
    api_key='PLACEHOLDER FOR API KEY - check README.md for instructions'
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
    
@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route("/submit_tip", methods=["POST"])
def generate_tip():
    try:
        # Load and encode the mds image in base64
        with open("static/images/mds_visualization.png", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    except Exception as e:
        print(f"Error in encoding image: {e}")
        return jsonify({"error": f"Failed to load and encode image: {str(e)}"}), 500

    # Get user input from the request body, use default if not provided
    user_prompt = request.json.get("user_input", "Provide me with a default financial tip.")

    # Use base64 image and user input to generate the response
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a financial assistant designed to provide users with e-commerce spending and budgeting tips based on MDS values on the following categories: "
                        f"Electronics, Clothing, Household, Entertainment, Necessities, Sports. "
                        f"Provide responses in a friendly, engaging tone. Ensure your reply is clear and easy to understand. "
                        f"Output should not exceed 100 words."
                        f"Prioritize based on the provided base64-encoded image to generate personalized financial tips. "
                        f"The image is encoded in base64 format. Use it as visual data to derive insights. The image is an MDS visualization of spending patterns, in the following cetegories: total_spending, average_spending, purchase_count."
                        f"The base64 image string is:\n\n{base64_image}."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )

        # Return the response
        return jsonify({"message": response.choices[0].message.content}), 200

    except Exception as e:
        print(f"Error in generating response: {e}")
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500



if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)