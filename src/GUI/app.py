from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

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
    # Here you would add your logic to process the rectangle
    # For example, you could pass these coordinates to a function that processes the image
    x,x_delta,y,y_delta = data['startX'],data['startX']+data['w'],data['startY'],data['startY']+data['h']
    x,x_delta,y,y_delta = int(x), int(x_delta), int(y), int(y_delta)
    print(x,x_delta,y,y_delta)
    return 'Rectangle processed', 200



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
