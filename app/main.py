from flask import Flask, request, redirect, url_for, render_template, send_file
import os
from werkzeug.utils import secure_filename
import subprocess

UPLOAD_FOLDER = '/app/uploads'
OUTPUT_FOLDER = '/app/outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run the detection script
            output_filename = filename.rsplit('.', 1)[0] + '_output.avi'
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            command = ['python3', 'detect_from_video.py', '-i', filepath, '-o', output_filepath]
            result = subprocess.run(command, capture_output=True, text=True)

            # Debug output
            print(result.stdout)
            print(result.stderr)

            if os.path.exists(output_filepath):
                return render_template('index.html', filename=output_filename)
            else:
                return render_template('index.html', filename=None, error="Processing failed.")
    return render_template('index.html', filename=None)

@app.route('/results/<filename>')
def results(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)