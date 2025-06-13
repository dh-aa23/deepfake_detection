import os
import torch
import numpy as np
from ultralytics import YOLO
import cv2
IMG_SIZE=299
NUM_FEATURES=2048
MAX_SEQ_LEN=100
class Preprocess:
    def __init__(self, model_path, video_path, feature_extractor, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(self.device)  # Force model to the desired device
        self.video_path = video_path
        self.feature_extractor = feature_extractor
    def crop_face(self,frame, target_size=(IMG_SIZE, IMG_SIZE)):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = self.model.predict(frame, verbose=False, device=self.device) 
        faces = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Optional: filter by class (0 = person for COCO YOLOv8 model)
                if cls == 0 and conf > 0.5:
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face = cv2.resize(face, target_size)
                    faces.append(face)

        if faces:
            return faces[0]  # Return the first detected face
        else:
            return cv2.resize(frame, target_size)  # fallback

    def load_video(self,max_frames=0,target_size=(IMG_SIZE,IMG_SIZE)):
        capt=cv2.VideoCapture(self.video_path)
        frames=[]
        frame_count=0
        while True:
            ret, frame = capt.read()
            if not ret or frame is None:
                break
            if max_frames and frame_count >= max_frames:
                break
            # print(frame.shape)
            faces = self.crop_face(frame,target_size)
            frames.append(faces)
            # print(faces.shape)
            frame_count += 1
        capt.release()
        return np.array(frames)


    def prepare_single_video(self, max_seq_len=MAX_SEQ_LEN):
        video_full_path = self.video_path
        frames = self.load_video(max_frames=max_seq_len)

        if len(frames) == 0:
            print(f"[WARNING] No valid frames found in video: {video_full_path}")
            return None, None

        length = min(len(frames), max_seq_len)

        # Trim or pad frames to match max_seq_len
        selected_frames = frames[:length]
        selected_frames = np.array(selected_frames)

        # Extract features
        features = self.feature_extractor.predict(selected_frames, verbose=0)  # shape: (length, NUM_FEATURES)

        # Prepare containers
        frame_mask = np.zeros((1, max_seq_len), dtype=bool)
        frame_features = np.zeros((1, max_seq_len, NUM_FEATURES), dtype=np.float32)

        frame_features[0, :length, :] = features
        frame_mask[0, :length] = True

        return frame_features, frame_mask
