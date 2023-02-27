import os
import sys

from data import *

from flask import Flask, redirect, render_template, url_for, request, flash
from werkzeug.utils import secure_filename

STATIC_FOLDER = 'app/static/'
SAMPLE_RESULTS_JSON = "app/static/results.json"
STATIC_UPLOAD_FOLDER = 'media/uploads/'

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'super secret key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1000 # max content size 50 MB
app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_FOLDER, STATIC_UPLOAD_FOLDER)

# Default data
sample_files = get_sample_video_paths()
sample_results = load_json(SAMPLE_RESULTS_JSON)
video_player = {
    'src': "media/sample-detected/Detected-This is not Morgan Freeman  -  A Deepfake Singularity.mp4",
    'filename': "This is not Morgan Freeman  -  A Deepfake Singularity.mp4"
}
upload_video = {
    'status': False,
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about-us')
def aboutus():
    return render_template('about-us.html')

@app.route('/detect', methods=["GET", "POST"])
def detect():
    return render_template('detect.html', 
                           sample_files=sample_files,
                           sample_results=sample_results,
                           video_player=video_player,
                           )

@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        file = request.files['videoToUpload']
        if file.filename == '':
            flash("No selected file", "Error:")
            return redirect(request.url)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(save_path)
        video_player['src'] = os.path.join(STATIC_UPLOAD_FOLDER, secure_filename(file.filename))
        video_player['filename'] = file.filename
        flash("Video successfully uploaded.","Info:")
    return redirect(url_for('detect'))

@app.errorhandler(413)
def largefile_error(e):
    print(e)
    flash("File size too large: Please choose a video less than 50 MB.", "Error!")
    return redirect(url_for('detect'))

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=5001, 
        debug=True
    )