from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model import predict_plate

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', plate_number='')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        plate_number = predict_plate(filepath)
        return render_template('index.html', plate_number=plate_number)

if __name__ == '__main__':
    app.run(debug=True)
