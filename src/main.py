import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os

# SYSTEM CALIBRATION
pyautogui.PAUSE = 0 
pyautogui.FAILSAFE = False

class JarvisUltimaPro:
    def __init__(self):
        self.model_path = "hand_landmarker.task"
        self._check_model()
        
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1, 
            min_hand_detection_confidence=0.85 # Increased for noise reduction
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        
        self.cap = cv2.VideoCapture(0)
        self.screen_w, self.screen_h = pyautogui.size()
        
        # FILTERING & STATE
        self.smooth = 5.0 # Weighted smoothing factor
        self.p_loc = np.array([0.0, 0.0])
        self.is_dragging = False
        self.prev_scroll_y = 0

    def _check_model(self):
        if not os.path.exists(self.model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = self.detector.detect(mp_image)

            if res.hand_landmarks:
                lms = res.hand_landmarks[0]
                
                # 1. LANDMARK EXTRACTION
                thumb = np.array([lms[4].x, lms[4].y])
                index = np.array([lms[8].x, lms[8].y])
                middle = np.array([lms[12].x, lms[12].y])
                
                # Distances
                pinch_dist = np.linalg.norm(index - thumb)
                scroll_dist = np.linalg.norm(index - middle)

                # 2. STATE ENGINE (With Hysteresis)
                state_text = "Tracking"
                box_color = (255, 255, 255)

                # A. DRAG LOGIC (Prioritized)
                if pinch_dist < 0.035:
                    state_text = "Clicked"
                    box_color = (0, 255, 0)
                    if not self.is_dragging:
                        pyautogui.mouseDown()
                        self.is_dragging = True
                
                elif self.is_dragging and pinch_dist > 0.06:
                    pyautogui.mouseUp()
                    self.is_dragging = False

                # B. SCROLL LOGIC (Requires Index & Middle together)
                elif scroll_dist < 0.03 and not self.is_dragging:
                    state_text = "Scrolling"
                    box_color = (255, 165, 0)
                    curr_y = index[1] * h
                    if self.prev_scroll_y != 0:
                        dy = self.prev_scroll_y - curr_y
                        if abs(dy) > 3: pyautogui.scroll(int(dy * 2))
                    self.prev_scroll_y = curr_y

                # C. MOVE LOGIC (Only if scroll_dist is high enough)
                else:
                    state_text = "Moving"
                    box_color = (0, 255, 255)
                    self.prev_scroll_y = 0
                    
                    # Interpolation with Deadzone [0.2, 0.8]
                    tx = np.interp(index[0], [0.2, 0.8], [0, self.screen_w])
                    ty = np.interp(index[1], [0.2, 0.8], [0, self.screen_h])
                    
                    # Weighted Moving Average
                    curr_loc = self.p_loc + (np.array([tx, ty]) - self.p_loc) / self.smooth
                    pyautogui.moveTo(curr_loc[0], curr_loc[1])
                    self.p_loc = curr_loc

                # 3. POLISHED UI RENDERING
                x_coords = [int(lm.x * w) for lm in lms]
                y_coords = [int(lm.y * h) for lm in lms]
                x_min, x_max = min(x_coords)-20, max(x_coords)+20
                y_min, y_max = min(y_coords)-20, max(y_coords)+20
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.rectangle(frame, (x_min, y_min-30), (x_min+150, y_min), box_color, -1)
                cv2.putText(frame, state_text, (x_min+5, y_min-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            cv2.imshow("Jarvis Stabilized V7", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if self.is_dragging: pyautogui.mouseUp()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    JarvisUltimaPro().run()
