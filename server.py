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

# TODO: Check for pre-defined set of extensions
def check_image_file(request):
    import pdb; pdb.set_trace()
    if 'image' not in request.files:
        flash('No file was uploaded')
        return redirect(request.url)

    image_file = request.files['image']

    if image_file.filename == '':
        flash('No file was uploaded')
        return redirect(request.url)

    return image_file


def save_image(image_file):
    '''Save the image inside config directory.'''
    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    return filename


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        image_file = check_image_file(request)
        import pdb; pdb.set_trace()
        if image_file:
            try:
                filename = save_image(image_file)
                passed = True
            except Exception:
                passed = False

            if passed:
                img_url = url_for('images', filename=filename)
                args.img_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)
                result = evaluate(model, args)

                _format = request.args.get('format')
                if _format == 'json':
                    return jsonify(str(result))
                else:
                    return render_template('predict.html',
                                           result=result,
                                           img_url=img_url)

            else:
                return redirect(url_for('error'))


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(Exception)
def error(error):
    return render_template('error.html'), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
