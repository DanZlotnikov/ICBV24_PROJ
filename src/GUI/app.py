import time

from flask import Flask, render_template, request, redirect, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
from urllib.parse import urlparse

from src.image_completion.image_completion import remove_rectangle

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = '../uploads/'
app.config['PROCESSED_FOLDER'] = '../processed/'

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

x, x_delta, y, y_delta = 0,0,0,0

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Here we render the same template but pass the filename to display the image
        return render_template('index.html', uploaded_image=filename)


@app.route('/process-rect', methods=['POST'])
def process_rect():
    data = request.get_json()
    print(data)
    # Here you would add your logic to process the rectangle
    # For example, you could pass these coordinates to a function that processes the image
    x, x_delta, y, y_delta = data['startX'], data['w'], data['startY'], data['h']
    print( x, x_delta, y, y_delta)

    x, x_delta, y, y_delta = int(x), int(x_delta), int(y), int(y_delta)
    imageName = data['Name']
    remove_rectangle(imageName,x, x_delta, y, y_delta)
    return 'Rectangle processed', 200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/complete-image', methods=['POST'])
def complete_image():
    method = request.form.get('method')
    print(method)
    if method == 'Single':
        pass
    elif method == 'Full':
        pass
    elif method == 'Tiling':
        pass

    return 'Rectangle processed', 200

@app.route('/process-image', methods=['POST'])
def process_image():
    image_path = request.form['imageSrc']
    parsed_url = urlparse(image_path)
    image_name = os.path.basename(parsed_url.path)

    if request.method == 'POST':
        # Capture the form data
        method = request.form.get('method')
        scale = request.form.get('scale')
        time.sleep(1)
        before_image_name = image_name  # Extract this from the form as you have been
        after_image_name = 'processed_' + image_name  # Assuming you name the processed file this way

        return render_template('before_vs_after.html',
                               before_image_name=before_image_name,
                               after_image_name=after_image_name)


@app.route('/')
def index():
    # Ensure the template can handle being called without parameters initially
    return render_template('index.html', method=None, scale=None)


if __name__ == '__main__':
    app.run(debug=True)

