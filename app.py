from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "File not found in request", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if file and file.filename.endswith('.csv'):
        file_path = file.save(os.path.join(app.config['uploads'], 'raw_data.csv'))
        file.save(file_path)

        # Run the BERT script
        subprocess.run(["python", "BERT.py"], check=True)

        # Run the ML script
        subprocess.run(["python", "ml_pro.py"], check=True)

        return "Processing complete", 200
    else:
        return "Invalid file type", 400

    



@app.route('/submit_tip', methods=['POST'])
def submit_tip():
    user_input = request.form['user_input']
    # Handle the user input as needed
    return redirect(url_for('tips'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)