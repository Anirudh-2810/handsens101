import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import urllib.request
import sys

class ModernVisionUI:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.model_path = "hand_landmarker.task"
        self._fetch_model()
        
        # Modern MediaPipe Tasks API Initialization
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = HandLandmarker.create_from_options(options)
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.ploc_x, self.ploc_y = 0, 0
        self.smooth = 5

    def _fetch_model(self):
        """Fetches the required Edge ML model directly from Google storage."""
        if not os.path.exists(self.model_path):
            print("System: Fetching ML Model dependencies...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def execute(self):
        if not self.cap.isOpened():
            sys.exit("CRITICAL EXCEPTION: HID Camera stream inaccessible. Check OS permissions.")

        print("System Online: Tracking Active.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: 
                continue

            # Matrix transformations for natural UX
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Neural network inference
            results = self.detector.detect(mp_image)

            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                idx_tip = hand[8]
                thumb_tip = hand[4]

                # Spatial Interpolation (Camera Space -> Monitor Space)
                tx = np.interp(idx_tip.x, [0.1, 0.9], [0, self.screen_w])
                ty = np.interp(idx_tip.y, [0.1, 0.9], [0, self.screen_h])

                # Exponential Moving Average for signal smoothing
                cloc_x = self.ploc_x + (tx - self.ploc_x) / self.smooth
                cloc_y = self.ploc_y + (ty - self.ploc_y) / self.smooth

                pyautogui.moveTo(cloc_x, cloc_y)
                self.ploc_x, self.ploc_y = cloc_x, cloc_y

                # Vector Calculus: Euclidean distance for click detection
                dist = np.linalg.norm(np.array([idx_tip.x, idx_tip.y]) - np.array([thumb_tip.x, thumb_tip.y]))
                if dist < 0.05:
                    pyautogui.click()
                    pyautogui.sleep(0.2) # Debounce

            # Output rendering
            cv2.imshow("Vision Interface", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        # Resource deallocation
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ModernVisionUI()
    app.execute()
