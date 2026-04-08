import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import time
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque

class HandTracker:
    def __init__(self):
        self.model_path = "hand_landmarker.task"
        self.model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        self._download_model()

        BaseOptions = mp_python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        self.connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        self.history = deque(maxlen=10)

    def _download_model(self):
        if not os.path.exists(self.model_path):
            print("Downloading model...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            print("Model ready.")

    def draw_hand(self, frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for s, e in self.connections:
            cv2.line(frame, pts[s], pts[e], (255, 200, 0), 2)

        for p in pts:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)
        return pts
    def draw_bbox(self, frame, landmarks, label):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    def draw_finger_status(self, frame, fingers):
        for i, val in enumerate(fingers):
            color = (0,255,0) if val else (0,0,255)
            cv2.circle(frame, (20 + i*30, 60), 10, color, -1)

    def draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (250,140), (0,0,0), -1)
        return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    def count_fingers(self, lm):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # thumb
        fingers.append(1 if lm[4].x < lm[3].x else 0)

        for t in tips[1:]:
            fingers.append(1 if lm[t].y < lm[t-2].y else 0)

        return fingers, sum(fingers)

    def recognize(self, fingers):
        patterns = {
            (0,0,0,0,0): "Fist",
            (1,1,1,1,1): "Open",
            (0,1,0,0,0): "One",
            (0,1,1,0,0): "Peace",
            (1,0,0,0,1): "Rock"
        }
        return patterns.get(tuple(fingers), "Unknown")

    def smooth_gesture(self, gesture):
        self.history.append(gesture)
        return max(set(self.history), key=self.history.count)
    def process(self, frame, timestamp):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, timestamp)

        frame = self.draw_overlay(frame)

        output = []

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                pts = self.draw_hand(frame, hand)
                fingers, count = self.count_fingers(hand)

                gesture = self.recognize(fingers)
                gesture = self.smooth_gesture(gesture)

                self.draw_bbox(frame, hand, gesture)
                self.draw_finger_status(frame, fingers)

                output.append((count, gesture))

        return frame, output, len(result.hand_landmarks or [])
class HandTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Controller")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")

        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()

        self.running = True
        self.prev_time = 0

        
        self.label = tk.Label(root)
        self.label.pack()

        
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start", command=self.start, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop", command=self.stop, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", command=self.quit, width=10).pack(side=tk.LEFT, padx=5)

        
        self.status = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        self.update_frame()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def quit(self):
        self.cap.release()
        self.root.destroy()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (800, 600))

                timestamp = int(time.time() * 1000)
                frame, data, num_hands = self.tracker.process(frame, timestamp)
                
                curr_time = time.time()
                fps = int(1 / (curr_time - self.prev_time)) if self.prev_time else 0
                self.prev_time = curr_time
         
                y = 30
                for count, gesture in data:
                    cv2.putText(frame, f"{gesture} ({count})", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    y += 30

                cv2.putText(frame, f"Hands: {num_hands}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"FPS: {fps}", (10, y+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
       
                self.status.config(text=f"Hands: {num_hands} | FPS: {fps}")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerGUI(root)
    root.mainloop()
