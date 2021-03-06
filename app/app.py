import os
import sys

from data import *

print(os.getcwd())

from flask import Flask, render_template, url_for

app = Flask(__name__, template_folder='templates', static_folder='static')

sample_files = get_sample_video_paths()
defaults = {
    'video_static_src': "media/sample-detected/Detected-This is not Morgan Freeman  -  A Deepfake Singularity.mp4",
    'video_filename': "This is not Morgan Freeman  -  A Deepfake Singularity.mp4"
}
results = load_json("app/static/results.json")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about-us')
def aboutus():
    return render_template('about-us.html')

@app.route('/detect')
def detect():
    return render_template('detect.html', 
                           sample_files=sample_files,
                           defaults=defaults,
                           results = results
                           )

# @app.route('/result', method=['GET'])
# def return_result():
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)