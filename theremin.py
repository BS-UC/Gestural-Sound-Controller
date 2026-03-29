import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyaudio
import time

# --- Hand Connections (Hardcoded for stability) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky and palm
]

# --- Configuration ---
# Audio settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32

# Cello pitch range (C2 to A5)
MIN_FREQ = 65.41
MAX_FREQ = 880.00

# Hand tracking settings
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.5
MODEL_ASSET_PATH = 'hand_landmarker.task'

# Control mapping
MAX_HAND_TRAVEL_PITCH = 300
MAX_HAND_TRAVEL_VOLUME = 300
SMOOTHING_FACTOR = 0.05 # EMA factor: Lower value = more smoothing.

# --- Audio Generation ---
class AudioGenerator:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=SAMPLE_RATE,
                                  output=True,
                                  frames_per_buffer=BUFFER_SIZE,
                                  stream_callback=self.audio_callback)
        self.frequency = MIN_FREQ
        self.amplitude = 0.0
        self.phase = 0

    def audio_callback(self, in_data, frame_count, time_info, status):
        t = (self.phase + np.arange(frame_count)) / SAMPLE_RATE
        wave = self.amplitude * (2 * (t * self.frequency - np.floor(0.5 + t * self.frequency)))
        self.phase += frame_count
        return (wave.astype(np.float32).tobytes(), pyaudio.paContinue)

    def set_frequency(self, freq):
        self.frequency = max(MIN_FREQ, min(freq, MAX_FREQ))

    def set_amplitude(self, amp):
        self.amplitude = max(0.0, min(amp, 1.0))

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# --- Hand Tracking ---
class HandTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.landmarker.detect(mp_image)
        
        if draw and self.results.hand_landmarks:
            height, width, _ = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]
                    cv2.line(img, (int(start_point.x * width), int(start_point.y * height)),
                             (int(end_point.x * width), int(end_point.y * height)),
                             (0, 255, 0), 2)
                for landmark in hand_landmarks:
                    cv2.circle(img, (int(landmark.x * width), int(landmark.y * height)), 5, (0, 0, 255), -1)
        return img

    def get_hand_positions(self, img_shape):
        left_hand, right_hand = None, None
        if self.results and self.results.hand_landmarks:
            for i, hand_landmarks in enumerate(self.results.hand_landmarks):
                handedness = self.results.handedness[i][0].category_name
                if handedness == 'Left':
                    left_hand = hand_landmarks[0]
                else:
                    right_hand = hand_landmarks[0]
        return left_hand, right_hand

# --- Main Application ---
def main():
    try:
        with open(MODEL_ASSET_PATH, 'rb') as f:
            pass
    except FileNotFoundError:
        print(f"Model file '{MODEL_ASSET_PATH}' not found. Downloading...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, MODEL_ASSET_PATH)
        print("Model downloaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hand_tracker = HandTracker()
    audio_gen = AudioGenerator()

    # --- Auto-Calibration ---
    left_start_y, right_start_y = None, None
    calibration_done = False
    stable_time_start = None
    last_left_pos, last_right_pos = None, None
    STABILITY_THRESHOLD = 0.01
    CALIBRATION_TIME = 3.0

    print("Calibration phase. Hold both hands steady in a comfortable, neutral position.")

    while not calibration_done:
        success, img = cap.read()
        if not success: continue
        
        img = hand_tracker.find_hands(img)
        left_hand, right_hand = hand_tracker.get_hand_positions((height, width))
        display_text = "Hold hands steady to calibrate..."

        if left_hand and right_hand:
            current_left_pos = (left_hand.x, left_hand.y)
            current_right_pos = (right_hand.x, right_hand.y)

            if last_left_pos and last_right_pos:
                left_dist = np.sqrt(sum([(a - b)**2 for a, b in zip(current_left_pos, last_left_pos)]))
                right_dist = np.sqrt(sum([(a - b)**2 for a, b in zip(current_right_pos, last_right_pos)]))

                if left_dist < STABILITY_THRESHOLD and right_dist < STABILITY_THRESHOLD:
                    if stable_time_start is None:
                        stable_time_start = time.time()
                    
                    elapsed_time = time.time() - stable_time_start
                    display_text = f"Calibrating... {int(CALIBRATION_TIME - elapsed_time) + 1}"

                    if elapsed_time >= CALIBRATION_TIME:
                        left_start_y = left_hand.y * height
                        right_start_y = right_hand.y * height
                        calibration_done = True
                        print(f"Calibration successful! Left start Y: {left_start_y:.2f}, Right start Y: {right_start_y:.2f}")
                        cv2.destroyWindow("Calibration")
                        time.sleep(1)
                        continue
                else:
                    stable_time_start = None
                    display_text = "Hold hands steady to calibrate..."
            
            last_left_pos, last_right_pos = current_left_pos, current_right_pos
        else:
            stable_time_start, last_left_pos, last_right_pos = None, None, None
            display_text = "Show both hands to calibrate..."

        cv2.putText(img, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Calibration", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    audio_gen.start()

    # --- Main Loop ---
    smoothed_pitch = MIN_FREQ
    smoothed_volume = 0.0

    try:
        while True:
            success, img = cap.read()
            if not success: break

            img = hand_tracker.find_hands(img)
            left_hand, right_hand = hand_tracker.get_hand_positions((height, width))

            current_pitch_hand_y = left_start_y
            current_volume_hand_y = right_start_y

            if left_hand:
                current_pitch_hand_y = left_hand.y * height
                cv2.line(img, (int(left_hand.x * width), int(left_start_y)), (int(left_hand.x * width), int(current_pitch_hand_y)), (255, 0, 0), 3)

            if right_hand:
                current_volume_hand_y = right_hand.y * height
                cv2.line(img, (int(right_hand.x * width), int(right_start_y)), (int(right_hand.x * width), int(current_volume_hand_y)), (0, 255, 0), 3)

            # --- Calculate Raw Values ---
            pitch_delta = left_start_y - current_pitch_hand_y
            volume_delta = right_start_y - current_volume_hand_y
            
            raw_pitch = MIN_FREQ + (pitch_delta / MAX_HAND_TRAVEL_PITCH) * (MAX_FREQ - MIN_FREQ)
            raw_volume = max(0, volume_delta / MAX_HAND_TRAVEL_VOLUME)
            raw_volume = min(1, raw_volume * 1.5)
            
            # --- Apply Smoothing (EMA) ---
            smoothed_pitch = (SMOOTHING_FACTOR * raw_pitch) + ((1 - SMOOTHING_FACTOR) * smoothed_pitch)
            smoothed_volume = (SMOOTHING_FACTOR * raw_volume) + ((1 - SMOOTHING_FACTOR) * smoothed_volume)

            audio_gen.set_frequency(smoothed_pitch)
            audio_gen.set_amplitude(smoothed_volume)

            cv2.putText(img, f"Pitch: {audio_gen.frequency:.2f} Hz", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Volume: {audio_gen.amplitude:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Digital Theremin", img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        print("Stopping...")
        audio_gen.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()