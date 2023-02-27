import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from typing import List, Tuple, Union

AVAILABLE_MODELS = ["mtcnn", "retinaface"]

class DetectorCmap:
    """
    Custom cmap. 
    """
    def __init__(self, l=None):
        if l is None:
            self.cmap = self.set_default_cmap()
        else:
            self.cmap = self.set_cmap(l)

    def set_default_cmap(self):
        c = ["red", "orange", "yellow", "greenyellow", "limegreen"]
        v = [0, .25, .5, .76 , 1.]
        l = list(zip(v,c))
        cmap = LinearSegmentedColormap.from_list("ryg", l, N=1000)
        return cmap
    
    def set_cmap(self, l):
        cmap = LinearSegmentedColormap.from_list("custom", l, N=1000)
        return cmap
    
    def get_rgb(self, k:float):
        if k < 0 or k > 1:
            raise ValueError("k value should be between 0 and 1.")
        rgba = self.cmap(k)
        r, g, b = int(rgba[0]*256), int(rgba[1]*256), int(rgba[2]*256)
        return r, g, b
    
    def get_bgr(self, k: float):
        if k < 0 or k > 1:
            raise ValueError("k value should be between 0 and 1.")
        rgba = self.cmap(k)
        r, g, b = int(rgba[0]*256), int(rgba[1]*256), int(rgba[2]*256)
        return b, g, r

    def display_cmap(self):
        fig, ax = plt.subplots(1,1,figsize=(6,1))
        ax.imshow(np.array([np.arange(1000) for i in range(100)]), cmap=self.cmap)
        ax.get_yaxis().set_visible(False)
        ax.set_xticks([0,500,1000])
        plt.show()
        

class FaceExtractor:
    
    def __init__(self, model : str, thresh:float=0.5):
        if model not in AVAILABLE_MODELS:
            raise Exception(f"{model} not valid.")
        self.model = model
        self.detector = self._initialise_detector()
        self.cmap = DetectorCmap()
        self.thresh = thresh
        
    def _initialise_detector(self) -> None:
        if self.model == "mtcnn":
            from mtcnn import MTCNN
            return MTCNN()
        elif self.model == "retinaface":
            from retinaface import RetinaFace
            return RetinaFace
    
    def draw_face_on_image(self, img: Union[str, np.ndarray], padding=0.3, width=2, **kwargs):
        """
        Detects a face and draws a rectangle on the image.
        """
        if type(img) == str:
            img = open_image(img)
        
        # Image size
        img_y = img.shape[0]
        img_x = img.shape[1]
        
        faces = self.detect_faces(img, **kwargs)
        for face in faces:
            if self.model == "mtcnn":
                x, y, w, h = face['box']
            elif self.model == "retinaface":
                x, y, w, h = face['box']
            if padding:
                x, y, w, h = add_padding(x,y,w,h, img_x, img_y, padding)
            confidence = face['confidence']
            box_colour = self.cmap.get_rgb(confidence)
            # Draws the rectangle on the img
            cv2.rectangle(img, (x,y), (x+w, y+h), box_colour , width)
        return img
    
    def crop_detected_face(self, img: Union[str, np.ndarray], padding:float=0.3, **kwargs) -> List[np.ndarray]:
        """
        Detects faces from an image and returns the cropped images of the faces.
        """
        if type(img) == str:
            img = open_image(img)
            
        img_y = img.shape[0]
        img_x = img.shape[1]
        cropped_images = []
        faces = self.detect_faces(img, **kwargs)
        for face in faces:
            if face['confidence'] > self.thresh:
                x, y, w, h = face['box']
                if padding:
                    x, y, w, h = add_padding(x,y,w,h, img_x, img_y, padding)
                cropped_images.append(img[y:y+h, x:x+h])
        return cropped_images
    
    def crop_best_face(self, img: Union[str, np.ndarray], padding:float=0.3) -> Tuple[np.ndarray, float]:
        """
        Detects faces from the given image and returns the face with the best 
        confidence level, and the confidence level, that is above the threshold.
        """
        if type(img) == str:
            img = open_image(img)
        
        img_y = img.shape[0]
        img_x = img.shape[1]
        
        best_confidence = 0
        best_face = None
        faces = self.detect_faces(img)
        for face in faces:
            if face['confidence'] > self.thresh and face['confidence'] > best_confidence:
                best_confidence = face['confidence']
                x, y, w, h = face['box']
                if padding:
                    x, y, w, h = add_padding(x,y,w,h, img_x, img_y, padding)
                best_face = img[y:y+h, x:x+h]

        return best_face, best_confidence
    
    def detect_faces(self, img: np.ndarray, 
                     sort_by: str = None,
                     face_limit: int = -1, 
                     min_area: float = 0.0
                     ) -> List:
        """Detects the faces using the specified detector."""
        SORT_METHODS = ["conf", "area"]
        if sort_by is not None and sort_by not in SORT_METHODS:
            raise ValueError(f"`{sort_by}` is not a valid `sort_by` argument")
        
        faces = []
        results = self.detector.detect_faces(img)
        
        for res in results:
            if self.model == "mtcnn":
                box = tuple(res['box'])
                confidence = res['confidence']
            elif self.model == "retinaface":
                # If 'res' is not a face key in the results output
                if type(res) is not str: 
                    return []
                res = results[res]
                x, y, xmax, ymax = tuple(res['facial_area'])
                w, h = xmax-x, ymax-y
                box = tuple((x, y, w, h))
                confidence = res['score']
            if min_area > 0.0:
                w, h = box[2], box[3]
                curr_area = (w * h) / (img.shape[0] * img.shape[1])
                if curr_area < min_area:
                    continue
            if confidence > self.thresh:
                faces.append({
                    'box': box,
                    'confidence': confidence
                })
        if sort_by == "conf":
            faces = sorted(faces, key=lambda x:x['confidence'], reverse=True)
        elif sort_by == "area":
            faces = sorted(faces, key=lambda x:x['box'][2]*x['box'][3], reverse=True)
        if face_limit > 0:
            faces = faces[:face_limit]
        return faces

def add_padding(x:int, y:int, w:int, h:int, xlim: int, ylim: int, 
                                amount:float=0.3) -> tuple:
    """
    Returns the dimensions after a padding amount has been added.
    Defaults to add 30% padding.
    """
    # Adds paddding
    new_w = int(w * (1 + amount)) 
    new_h = int(h * (1 + amount))      
    new_x = x - int( (new_w - w)/2 )
    new_y = y - int( (new_h - h)/2 )
    # Check limits
    if new_w > xlim:
        new_w = xlim
        new_x = 0
    if new_h > ylim:
        new_h = ylim
        new_y = 0
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    return new_x, new_y, new_w, new_h

def open_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found.")
    return img


class FrameExtractor:
        
    def __init__(self, video_source: str, verbose=False):
        self.source = video_source
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise Exception(f"{self.source} is not a valid video.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.verbose = verbose
            
    def get_frames_evenly(self, n_frames: int) -> List:
        if n_frames <= 0:
            raise ValueError(f"'n_frames' must be 1 or more.")
        if n_frames < self.frame_count:
            step = int(self.frame_count/n_frames) # number of frames to skip
            frame_locations = np.arange(1, self.frame_count+1, step)
            frame_locations[:n_frames]
        else: # Get all frames from video
            frame_locations = np.arange(1,self.frame_count,1)
        return self.get_frames(frame_locations)
        
    def get_frames_per_second(self, fps: float) -> List:
        if fps == 0:
            if self.verbose:
                print("Cannot extract 0 frames per second.")
            return None
        frame_skip = round(round(self.fps)/fps)
        frame_locations = np.arange(0, self.frame_count, frame_skip)
        return self.get_frames(frame_locations)
        
    def get_frames(self, frame_locs: List[int]) -> List:
        """Returns a list of video frames as images or numpy arrays given the
        index position of the frame in the video."""
        # Reset vidcapture
        self.capture.set(1,0)
        frames = []
        n_success = 0
        for i in frame_locs:
            self.capture.set(1,i)
            success, image = self.capture.read()
            if success:
                n_success += 1
                frames.append({
                    'loc': i,
                    'frame': image
                })
        if self.verbose:
            print(f"Successfully extracted {n_success} frames from {self.source}")        
        return frames
        