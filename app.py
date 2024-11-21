from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('error'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('error'))
    if file and file.filename.endswith('.csv'):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('success'))
    else:
        return redirect(url_for('error'))

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/submit_tip', methods=['POST'])
def submit_tip():
    user_input = request.form['user_input']
    # Handle the user input as needed
    return redirect(url_for('tips'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)