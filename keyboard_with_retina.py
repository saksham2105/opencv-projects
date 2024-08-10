import cv2
import numpy as np
import mediapipe as mp
import math
from pynput.keyboard import Controller, Key
import pygame

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the keyboard controller
keyboard = Controller()

keys = [
    ('Q', (0.05, 0.1), (0.08, 0.08)),
    ('W', (0.14, 0.1), (0.08, 0.08)),
    ('E', (0.23, 0.1), (0.08, 0.08)),
    ('R', (0.32, 0.1), (0.08, 0.08)),
    ('T', (0.41, 0.1), (0.08, 0.08)),
    ('Y', (0.50, 0.1), (0.08, 0.08)),
    ('U', (0.59, 0.1), (0.08, 0.08)),
    ('I', (0.68, 0.1), (0.08, 0.08)),
    ('O', (0.77, 0.1), (0.08, 0.08)),
    ('P', (0.86, 0.1), (0.08, 0.08)),
    ('A', (0.10, 0.22), (0.08, 0.08)),
    ('S', (0.19, 0.22), (0.08, 0.08)),
    ('D', (0.28, 0.22), (0.08, 0.08)),
    ('F', (0.37, 0.22), (0.08, 0.08)),
    ('G', (0.46, 0.22), (0.08, 0.08)),
    ('H', (0.55, 0.22), (0.08, 0.08)),
    ('J', (0.64, 0.22), (0.08, 0.08)),
    ('K', (0.73, 0.22), (0.08, 0.08)),
    ('L', (0.82, 0.22), (0.08, 0.08)),
    ('Z', (0.15, 0.34), (0.08, 0.08)),
    ('X', (0.24, 0.34), (0.08, 0.08)),
    ('C', (0.33, 0.34), (0.08, 0.08)),
    ('V', (0.42, 0.34), (0.08, 0.08)),
    ('B', (0.51, 0.34), (0.08, 0.08)),
    ('N', (0.60, 0.34), (0.08, 0.08)),
    ('M', (0.69, 0.34), (0.08, 0.08)),
    (' ', (0.10, 0.48), (0.80, 0.08)),  # Space key with larger width
    ('enter', (0.85, 0.22), (0.12, 0.30)),  # Enter key
    ('Left', (0.40, 0.75), (0.08, 0.08)),  # Left Arrow (moved to the left)
    ('Right', (0.50, 0.75), (0.08, 0.08)),  # Right Arrow (moved to the left)
    ('Up', (0.45, 0.65), (0.08, 0.08)),  # Up Arrow (moved to the left)
    ('Down', (0.45, 0.85), (0.08, 0.08)),  # 
]

# Initialize pygame for sound playback
pygame.init()
pygame.mixer.init()
sound_effect = pygame.mixer.Sound('mechanical_keyboard_effect.mp3')
enter_space = pygame.mixer.Sound('enter_space.wav')

# Start capturing video
cap = cv2.VideoCapture(0)

def distance_between_points(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=15):
    """Draw a rounded rectangle on the image."""
    (x1, y1) = top_left
    (x2, y2) = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)

def draw_keyboard(frame, frame_width, frame_height, hover_key=None):
    """Draw the virtual keyboard on the frame with an optional hover effect."""
    keyboard_frame = np.zeros_like(frame, dtype=np.uint8)  # Create a blank image

    # Draw the keyboard border
    border_thickness = 5
    border_color = (255, 255, 255)  # White border
    keyboard_x1 = int(0.05 * frame_width) - 50
    keyboard_y1 = int(0.05 * frame_height)
    keyboard_x2 = int(0.95 * frame_width) + 60
    keyboard_y2 = int(0.85 * frame_height + 70)
    cv2.rectangle(keyboard_frame, (keyboard_x1, keyboard_y1), (keyboard_x2, keyboard_y2), border_color, border_thickness)

    for key, (rel_x, rel_y), (rel_w, rel_h) in keys:
        x = int(rel_x * frame_width)
        y = int(rel_y * frame_height)
        w = int(rel_w * frame_width)
        h = int(rel_h * frame_height)

        if hover_key == key:
            # Apply zoom effect for hovered key
            x, y, w, h = int(x), int(y), int(w * 1.2), int(h * 1.2)  # Increase size for zoom effect
            x -= int((w * 0.2) / 2)  # Adjust position to center the zoomed-in key
            y -= int((h * 0.2) / 2)

        draw_rounded_rectangle(keyboard_frame, (x, y), (x + w, y + h), (200, 200, 200), radius=10)  # Semi-transparent rectangle

        # Draw text on keys
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(keyboard_frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return keyboard_frame

def get_key_at_position(x, y):
    """Determine which key is at the given (x, y) position."""
    for key, (rel_x, rel_y), (rel_w, rel_h) in keys:
        key_x = int(rel_x * frame_width)
        key_y = int(rel_y * frame_height)
        key_w = int(rel_w * frame_width)
        key_h = int(rel_h * frame_height)
        if key_x <= x <= key_x + key_w and key_y <= y <= key_y + key_h:
            return key
    return None

def estimate_gaze_direction(eye_landmarks, frame_width, frame_height):
    """Estimate gaze direction based on eye landmarks."""
    if not eye_landmarks:
        return None
    
    # Assuming eye_landmarks contain the points of the eye
    # Simplified example using eye center
    left_eye = eye_landmarks[0:6]  # Example of left eye landmarks
    right_eye = eye_landmarks[6:12]  # Example of right eye landmarks

    # Calculate the center of the eye
    left_eye_center = np.mean([landmark for landmark in left_eye], axis=0)
    right_eye_center = np.mean([landmark for landmark in right_eye], axis=0)

    # Average gaze direction
    gaze_center_x = int((left_eye_center[0] + right_eye_center[0]) / 2 * frame_width)
    gaze_center_y = int((left_eye_center[1] + right_eye_center[1]) / 2 * frame_height)

    return gaze_center_x, gaze_center_y

def draw_circle_on_eye(img, eye_landmarks, frame_width, frame_height):
    """Draw a small circle on the retina based on eye landmarks."""
    if not eye_landmarks:
        return img

    # Assuming eye_landmarks contains the points of the eye
    left_eye = eye_landmarks[0:6]  # Example of left eye landmarks
    right_eye = eye_landmarks[6:12]  # Example of right eye landmarks

    # Calculate the center of the eye
    left_eye_center = np.mean([landmark for landmark in left_eye], axis=0)
    right_eye_center = np.mean([landmark for landmark in right_eye], axis=0)

    # Average eye center
    eye_center_x = int((left_eye_center[0] + right_eye_center[0]) / 2 * frame_width)
    eye_center_y = int((left_eye_center[1] + right_eye_center[1]) / 2 * frame_height)

    # Draw a small circle on the retina
    circle_radius = 5
    circle_color = (0, 0, 255)  # Red color
    cv2.circle(img, (eye_center_x, eye_center_y), circle_radius, circle_color, -1)

    return img

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Process the frame to detect hands and face
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    hovered_key = None  # Track the currently hovered key

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Extract landmarks for eyes (example indices, adjust as needed)
            eye_landmarks = [face_landmarks.landmark[i] for i in [33, 133, 160, 158, 153, 144, 362, 263, 387, 385, 380, 374]]
            eye_landmarks = [(lm.x, lm.y) for lm in eye_landmarks]

            # Draw a small circle on the retina
            frame = draw_circle_on_eye(frame, eye_landmarks, frame_width, frame_height)

            # Estimate gaze direction
            gaze_direction = estimate_gaze_direction(eye_landmarks, frame_width, frame_height)

            if gaze_direction:
                # Determine which key is being hovered over
                hovered_key = get_key_at_position(gaze_direction[0], gaze_direction[1])
                if hovered_key:
                    # Highlight the key
                    key_pos = next(((k, pos, size) for k, pos, size in keys if k == hovered_key), None)
                    if key_pos:
                        key_x = int(key_pos[1][0] * frame_width)
                        key_y = int(key_pos[1][1] * frame_height)
                        key_w = int(key_pos[2][0] * frame_width)
                        key_h = int(key_pos[2][1] * frame_height)
                        if hovered_key == ' ' or hovered_key == 'enter':
                            # Highlight rectangle keys with a different color
                            draw_rounded_rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), (0, 255, 255), radius=10)  # Semi-transparent highlight
                        else:
                            draw_rounded_rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), (0, 255, 255), radius=10)  # Semi-transparent highlight
                        
                        # Simulate key press with sound effect
                        if hovered_key == 'enter':
                            keyboard.press(Key.enter)
                            keyboard.release(Key.enter)
                            enter_space.play()
                        elif hovered_key in {'Left', 'Right', 'Up', 'Down'}:
                            # Handle arrow keys
                            if hovered_key == 'Left':
                                keyboard.press(Key.left)
                                keyboard.release(Key.left)
                            elif hovered_key == 'Right':
                                keyboard.press(Key.right)
                                keyboard.release(Key.right)
                            elif hovered_key == 'Up':
                                keyboard.press(Key.up)
                                keyboard.release(Key.up)
                            elif hovered_key == 'Down':
                                keyboard.press(Key.down)
                                keyboard.release(Key.down)
                            sound_effect.play()
                        else:
                            keyboard.press(hovered_key)
                            keyboard.release(hovered_key)
                            if hovered_key == ' ':
                                enter_space.play()
                            else:
                                sound_effect.play()

    # Draw the virtual keyboard on the frame with the zoom effect for the hovered key
    keyboard_frame = draw_keyboard(frame, frame_width, frame_height, hover_key=hovered_key)
    
    # Combine the keyboard frame with the original frame
    combined_frame = cv2.addWeighted(frame, 1.0, keyboard_frame, 0.6, 0)  # Adjust weights for transparency

    # Display the resulting frame
    cv2.imshow('Jarvis-like Stylish Keyboard', combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()  # Quit pygame when done
