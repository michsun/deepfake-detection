import os
import sys

print(os.getcwd())

from flask import Flask, render_template, url_for

app = Flask(__name__, 
template_folder='templates', 
static_folder='static')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about-us')
def aboutus():
    return render_template('about-us.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)