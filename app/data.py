import os
import json

# Locations of the sample videos
SAMPLE_VIDEO_DIR = "app/static/media/sample-original/Youtube"
SAMPLE_DETECTED_DIR = "app/static/media/detected/"

def get_sample_video_paths():
    filenames = []
    directory = os.fsencode(SAMPLE_VIDEO_DIR)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
    filenames = sorted(filenames)
    return filenames

def load_json(path):
    with open(path,'r') as fp:
        data = json.load(fp)
    return data

