import os
from data import *

results = load_json("app/static/results.json")

if __name__ == "__main__":
    for k,v in results.items():
        root, filename = os.path.split(k)
        print(filename)
        print("Prediction :", v['prediction'])
        print("Average confidence:", v['prediction_confidence'])
        print()