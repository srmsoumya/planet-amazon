from predict import load_model, predict as evaluate
import os
import json
import torch

from flask import (Flask, flash, redirect, render_template, request,
                   jsonify, send_from_directory, url_for)

# TODO: set from environment variables
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Mjolnir'
app.config['UPLOAD_FOLDER'] = '/tmp/'


class Arguments:
    'Set the arguments required by the model'
    device = torch.device('cpu')  # Run it in CPU mode by default
    cp_file = 'cp_best.pt.tar'
    drop_rate = 0.0
    img_path = None


args = Arguments()

# Load the model
model = load_model(args)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file was uploaded')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No file was uploaded')
            return redirect(request.url)

        if image_file:
            passed = False
            try:
                filename = image_file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                return redirect(url_for('error'))


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    img_url = url_for('images', filename=filename)
    args.img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = evaluate(model, args)

    return render_template('predict.html',
                           result=result[0],
                           img_url=img_url)


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
