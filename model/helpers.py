import json
import numpy as np
import os

from typing import List

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(data, path):
    with open(path,'w') as fp:
        json.dump(data, fp, cls=NpEncoder)
        
def load_json(path):
    with open(path,'r') as fp:
        data = json.load(fp)
    return data

def time_delta_str(seconds) -> str:
    h = int(seconds/(60*60))
    m = int( (seconds - (h * 60 * 60)) / 60 )
    s = round( seconds - (m * 60), 2 )
    delta = "{} hours(s) {} minute(s) {} second(s)".format(h, m, s)
    return delta

def iterate_files(directory: str) -> List:
    """Iterates over the files in the given directory and returns a list of 
    found files."""
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fullpath = os.path.join(directory, filename)
        if not os.path.isdir(fullpath):
            files.append(fullpath)
    return files