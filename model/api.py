import json
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, jsonify, request, make_response

import argparse
import json
import time
from tqdm import tqdm

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello, world!</h1>'

@app.route('/analyze', methods=['POST'])
def analyze():

    return 