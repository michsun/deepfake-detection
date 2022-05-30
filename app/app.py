from flask import Flask, render_template, url_for



app = Flask(__name__)


#@app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

#@app.route('/video')
#def video():
#    return render_template('video.html')

#@app.route('/index')
#def index():
#    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)