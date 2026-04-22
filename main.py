import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import sys

class JarvisInterface:
    def __init__(self):
        # 1. Bootstrapping & Hardware Check
        pyautogui.FAILSAFE = False  # Prevents crash if cursor hits corners
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not access webcam. Ensure no other app is using it.")
            sys.exit()

        # 2. Initialize MediaPipe Engine
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 3. Screen Dimensions & Smoothing Setup
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_factor = 7  # Higher = smoother but slower; Lower = faster but shakier
        self.ploc_x, self.ploc_y = 0, 0
        self.cloc_x, self.cloc_y = 0, 0
        
        # 4. Define Active Zone (Padding)
        # We use a box in the center of the camera so you don't have to reach across the room
        self.frame_pad = 100 

    def start_system(self):
        print("Jarvis Vision UI: Online")
        print("Controls: Index Finger = Move | Pinch (Index + Thumb) = Click | 'q' = Quit")
        
        while True:
            success, img = self.cap.read()
            if not success: break

            # Flip image for mirror effect
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Draw visual feedback on the window
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get Landmarks: 8 (Index Tip), 4 (Thumb Tip)
                    index_tip = hand_lms.landmark[8]
                    thumb_tip = hand_lms.landmark[4]

                    # Map Camera Coordinates to Screen Coordinates
                    # Use np.interp to scale the 'Active Zone' to full screen size
                    tx = np.interp(index_tip.x * w, (self.frame_pad, w - self.frame_pad), (0, self.screen_w))
                    ty = np.interp(index_tip.y * h, (self.frame_pad, h - self.frame_pad), (0, self.screen_h))

                    # Apply Smoothing (Current Location = Previous + (Target - Previous) / Smoothing)
                    self.cloc_x = self.ploc_x + (tx - self.ploc_x) / self.smooth_factor
                    self.cloc_y = self.ploc_y + (ty - self.ploc_y) / self.smooth_factor

                    # Move Cursor
                    pyautogui.moveTo(self.cloc_x, self.cloc_y)
                    self.ploc_x, self.ploc_y = self.cloc_x, self.cloc_y

                    # Detect Click (Pinch Gesture)
                    # Calculate distance between thumb and index tips
                    dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    
                    if dist < 0.05: # Threshold for a pinch
                        pyautogui.click()
                        cv2.circle(img, (int(index_tip.x * w), int(index_tip.y * h)), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.sleep(0.1) # Debounce

            # Visual UI Window
            cv2.rectangle(img, (self.frame_pad, self.frame_pad), (w-self.frame_pad, h-self.frame_pad), (255, 0, 255), 2)
            cv2.imshow("Jarvis Vision", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ui = JarvisInterface()
    ui.start_system()
