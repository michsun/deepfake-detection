from matplotlib.font_manager import json_dump
import tensorflow as tf

from tensorflow import keras
from keras import backend as K

import concurrent.futures
import cv2
import json
import numpy as np
import os
import subprocess
import time
from tqdm import tqdm

from extractors import *
from helpers import *

MODEL_PATH = "model/weights/efficientnetb7-v7-1.hdf5"
MODEL_DIM = (256, 256)

OUTPUT_PATH = "app/static/sample-media/"

class Predictor:
    
    def __init__(self):
        self.model = self._init_model()
        # self.session = K.get_session()
        # self.graph = tf.get_default_graph()
        # self.graph.finalize()
        
    def _init_model(self, verbose=1):
        tic = time.time()
        model = keras.models.load_model(MODEL_PATH)
        toc = time.time()
        if verbose == 1:
            print("Model loading complete in", time_delta_str(toc - tic))
        return model
    
    def _preprocessing(self, image: np.array):
        image = cv2.resize(image, MODEL_DIM, cv2.INTER_AREA)
        # Convert shape to (None, 256, 256, 3)
        batch_shape = []
        batch_shape.append(image)
        image = np.asarray(batch_shape)
        return image
    
    def predict(self, image: np.array):
        # TODO: Image is approapriate (3 channels etc.)
        image = self._preprocessing(image)
        # with self.session.as_default():
            # with self.graph.as_default():
        pred = self.model.predict(image)[0][0]
        return pred
    
    def worker_predict(self, image: np.array):
        work_model = keras.models.load_model(MODEL_PATH)
        image = self._preprocessing(image)
        pred = work_model.predict(image)[0][0]
        return pred
    
class VideoDeepfakeDetector:
    
    def __init__(self, video_source : str,
                 model: Predictor,
                 face_detector="retinaface",
                 face_detector_thresh:float=0.5,
                 detect_per_second: float = 10.,
                 output_dir: str = ".",
                 ):
        self.video_source = video_source
        self.capture = self._init_capture()
        self.face_detector = face_detector
        self.face_extractor = FaceExtractor(model=face_detector, thresh=face_detector_thresh)
        self.model = model
        # Video details
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Video out details
        _, filename = os.path.split(video_source)
        filename, _ = os.path.splitext(filename)
        self.output_path = os.path.join(output_dir, "Detected-"+filename+".mp4")
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)
        codec = cv2.VideoWriter_fourcc(*'H264') # MJPG, MP4V, H264
        self.output = cv2.VideoWriter(self.output_path, codec, self.fps, (self.frame_width, self.frame_height))
        self.cmap = DetectorCmap()
        self.display_scale = get_display_scale(self.frame_width, self.frame_height)
        if detect_per_second == 0:
            print("Cannot detect 0 frames per second. Setting default detect_per_second=10")
            detect_per_second=10
        self.frame_skip = round(round(self.fps) / detect_per_second)
        self.results = self._init_results()
        
    def _init_capture(self):
        capture = cv2.VideoCapture(self.video_source)
        if not capture.isOpened():
            raise Exception(f"{self.video_source} is not a valid video.")
        return capture
    
    def _init_results(self):
        results = {}
        results['original_source'] = self.video_source
        results['output_source'] = self.output_path
        results['face_detector'] = self.face_detector
        results['frames_skipped'] = self.frame_skip
        results['frame_details'] = []
        return results
    
    def run_task_by_task(self, workers=8):
        # Display params
        box_width = int(1 * self.display_scale)
        font_scale = 0.6 * self.display_scale
        
        tic = time.time()
        
        ### 1. GETTING INITIAL FRAMES <-- BOTTLENECK ###
        frame_indexes = np.arange(0,self.total_frames, self.frame_skip)
        # Get frames from indices
        frames_to_run = []
        for i in tqdm(range(len(frame_indexes)), desc="Getting video frames"):
            self.capture.set(1,frame_indexes[i])
            success, frame = self.capture.read()
            if success:
                frames_to_run.append(frame)
            else:
                print("Error in frame count")
                return
        self.capture.set(1,0)
        
        ### 2. DETECT FACES ###
        def detect_and_crop_frames(index):
            image = frames_to_run[index]
            frame_loc = frame_indexes[index]
            
            frame_detail = {}
            frame_detail['idx'] = frame_loc
            frame_detail['face_details'] = []
            frame_detail['cropped_frames'] = []
            
            detected_faces = self.face_extractor.detect_faces(image, sort_confidence=True)
            for face in detected_faces:
                x, y, w, h = face['box']
                x, y, w, h = add_padding(x, y, w, h, self.frame_width, self.frame_height, 0.3)
                # 1. Crop the frame
                cropped_frame = frame[y:y+h, x:x+h]
                # 2. Predict each frame if not skipped
                frame_detail['face_details'].append({
                    'box': tuple((x, y, w, h)),
                    'face_confidence': face['confidence'],
                })
                frame_detail['cropped_frames'].append(cropped_frame)
            return frame_detail
        
        indexes = np.arange(0,len(frames_to_run),1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            frame_details = list(tqdm(executor.map(detect_and_crop_frames, indexes), desc="Detecting faces ", total=len(frames_to_run)))
        # print("Completed detecting faces and prediction in", time_delta_str(toc-tic))
        
        ### 3. PREDICTING DEEPFAKE ###
        for frame in tqdm(frame_details, desc="Predicting deepfakes", total=len(frame_details)):
            for i, face in enumerate(frame['cropped_frames']):
                deepfake_pred = self.model.predict(face)
                frame['face_details'][i]['deepfake_prediction'] = deepfake_pred
        
        # Sort frame details by 'idx'
        frame_details =  sorted(frame_details, key=lambda x:x['idx'])
        
        ### 4. WRITE VIDEO ###
        frame_index = 0
        face_details = []
        for i in tqdm(range(self.total_frames), desc="Writing video "):
            success, frame = self.capture.read()
            if not success:
                print("Error in frame count")
                return
            
            if frame_index < len(frame_details) and frame_details[frame_index]['idx'] == i:
                face_details = []
                face_details = frame_details[frame_index]['face_details']
                frame_index += 1

            for face in face_details:
                x, y, w, h = face['box']
                box_colour = self.cmap.get_rgb(face['deepfake_prediction'])
                text_colour = (0,0,0)
                label = "Deepfake:{:.1f}%".format(face['deepfake_prediction']*100)

                cv2.rectangle(frame, (x,y), (x+w, y+h), box_colour, box_width)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,  font_scale, 1)
                cv2.rectangle(frame, (x, y ), (x + tw, y+th+int(4*self.display_scale)), box_colour, -1)
                cv2.putText(frame, label, (x, y+th+int(2*self.display_scale)), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_colour, 1)
            self.output.write(frame)
        
            self.results['frame_details'].append({
                'idx': i,
                'details': face_details
            })

        toc = time.time()
        print(f"Complete writing {self.output_path} in {time_delta_str(toc-tic)}.")
        self.results['time'] = toc-tic
        return self.results
        
    def run_frame_by_frame(self):
        """
        Predicts whether it is a deepfake on a frame by frame basis.
        """
        # Display params
        box_width = int(1 * self.display_scale)
        font_scale = 0.6 * self.display_scale
        
        curr_frame_skip = self.frame_skip
        face_details = []
        detected_faces = []
        
        tic = time.time()
        for i in tqdm(range(self.total_frames), desc="Detecting deepfake "):
            success, frame = self.capture.read()
            if not success:
                print("Error in frame count")
                return
            
            if curr_frame_skip == self.frame_skip:
                detected_faces = self.face_extractor.detect_faces(frame, sort_confidence=True)
                face_details = []
                for face in detected_faces:
                    x, y, w, h = face['box']
                    x, y, w, h = add_padding(x, y, w, h, self.frame_width, self.frame_height, 0.3)
                    
                    # 1. Crop the frame
                    cropped_frame = frame[y:y+h, x:x+h]
                    # 2. Predict each frame if not skipped
                    deepfake_pred = self.model.predict(cropped_frame)
                    face_details.append({
                        'box': tuple((x, y, w, h)),
                        'face_confidence': face['confidence'],
                        'deepfake_prediction': deepfake_pred
                    })
                curr_frame_skip = 0

            for face in face_details:
                x, y, w, h = face['box']
                box_colour = self.cmap.get_rgb(face['deepfake_prediction'])
                text_colour = (0,0,0)
                label = "Deepfake: {:.1f} %".format(face['deepfake_prediction']*100)
                
                cv2.rectangle(frame, (x,y), (x+w, y+h), box_colour, box_width)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,  font_scale, 1)
                cv2.rectangle(frame, (x, y ), (x + tw, y+th+int(4*self.display_scale)), box_colour, -1)
                cv2.putText(frame, label, (x, y+th+int(2*self.display_scale)), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_colour, 1)
            self.output.write(frame)
            
            self.results['frame_details'].append({
                'idx': i,
                'details': face_details
            })
            curr_frame_skip += 1
        
        toc = time.time()
        print(f"Complete writing {self.output_path} in {time_delta_str(toc-tic)}.")
        self.results['time'] = toc-tic
        self.predict_average_primary_face()
        print("Prediction complete.")
        return self.results
    
    def predict_average_primary_face(self, thresh=0.5):
        frame_indices = np.arange(0,self.total_frames, self.frame_skip)
        # print(self.results['frame_details'][0])

        primary_face_predictions = []
        for index in frame_indices:
            frame = self.results['frame_details'][index]
            if len(frame['details']) > 0:
                primary_face_predictions.append(frame['details'][0]['deepfake_prediction'])
        print(primary_face_predictions)
        confidence = sum(primary_face_predictions) / len(primary_face_predictions)
        if confidence > thresh:
            self.results['prediction'] = "FAKE"
        else:
            self.results['prediction'] = "REAL"
        self.results['prediction_confidence'] = confidence
        return self.results
            
        
        
        

def get_display_scale(frame_width, frame_height):
    max_dim = max(frame_width, frame_height)
    if max_dim >= 3840: # 4k
        return 3
    if max_dim >= 1920: # 1080p
        return 2
    if max_dim >= 1280: # 720p
        return 1.5
    else:
        return 1


SAMPLE_VIDEO_DIR = "app/static/media/sample-original/"
SAMPLE_DETECTED_DIR = "app/static/media/sample-detected/"

def get_sample_video_paths():
    filenames = []
    directory = os.fsencode(SAMPLE_VIDEO_DIR)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
    filenames = sorted(filenames)
    return filenames

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
    
class FFMpegSubprocess:
    
    def __init__(self):
        pass
    
    def path_string(self, path):
        return path.replace(" ", "\ ").replace("?", "\?").replace("&", "\&").replace("(", "\(").replace(")", "\)").replace("*", "\*").replace("<", "\<").replace(">", "\>").replace("[","\[").replace("]","\]")
    
    def merge_video_audio(self, video_source, audio_source, output_video):
        root, _ = os.path.split(output_video)
        temp_output_video = os.path.join(root, "temp.mp4")
        cmd = ['ffmpeg',
               '-i',self.path_string(video_source),
               '-i',self.path_string(audio_source), 
               '-c', 'copy'
               '-map','0:0',
               '-map','1:1',
               '-shortest',
               '-qscale','0',
               '-vcodec', 'mpeg4',
                self.path_string(temp_output_video) ]
        os.system(' '.join(cmd))
        while not os.path.exists(temp_output_video):
            time.sleep(1)
        cmd = [
            'ffmpeg',
            '-i', self.path_string(temp_output_video),
            '-vcodec', 'libx264',
            self.path_string(output_video)
        ]
        os.system(' '.join(cmd))
        # subprocess.Popen(cmd)
        while not os.path.exists(output_video):
            time.sleep(1)
        os.remove(temp_output_video)


def main(filepaths, outputdir, save_path):    
    results = {}
    model = Predictor()
    for original_file in filepaths:
        detector = VideoDeepfakeDetector(video_source=original_file,
                                     model=model,
                                     face_detector="retinaface",
                                     face_detector_thresh=0.8,
                                     detect_per_second=5,
                                     output_dir = outputdir
                                     )
        results[original_file] = detector.run_frame_by_frame()
    save_json(results, save_path)
        
if __name__ == "__main__":
    n_gpus = len(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", n_gpus)
    if n_gpus == 0:
        print("NO GPU SETUP")
        import sys
        sys.exit(0)
    
    sample_detected_path = "app/static/media/sample-detected/"
    json_save_path = "app/static/results.json"
    queue = [
        "app/static/media/sample-original/'We will defend ourselves', says Ukrainian president Volodymyr Zelenskiy [ real ].mp4",
        "app/static/media/sample-original/Bill Hader channels Tom Cruise [DeepFake].mp4",
        "app/static/media/sample-original/Deepfake Queen- 2020 Alternative Christmas Message.mp4",
        "app/static/media/sample-original/Deepfake video of Volodymyr Zelensky surrendering surfaces on social media.mp4",
        "app/static/media/sample-original/Robert Downey Jr and Tom Holland in Back to the future - This is heavy! [ deepfake ].mp4",
        "app/static/media/sample-original/This is not Morgan Freeman  -  A Deepfake Singularity.mp4",
        "app/static/media/sample-original/Ukraine president Zelensky says Russia has marked him 'as target No.1' and his family 'No.2' [ real ].mp4",
        "app/static/media/sample-original/Very realistic Tom Cruise Deepfake | AI Tom Cruise.mp4",
        "app/static/media/sample-original/You Won’t Believe What Obama Says In This Video!.mp4"
    ]
    main(queue, sample_detected_path, json_save_path)
    
    # ffmpeg = FFMpegSubprocess()
    # audio_source = queue[0]
    # video_source = "app/static/sample-media/Detected-Deepfake video of Volodymyr Zelensky surrendering surfaces on social media.mp4"
    
    # sample_detected_path = "app/static/media/sample-detected/"
    # _, filename = os.path.split(audio_source)
    # filename, _ = os.path.splitext(filename)
    # final_video = os.path.join(sample_detected_path, filename+'.mp4')
    
    # ffmpeg.merge_video_audio(video_source, audio_source, final_video)
    