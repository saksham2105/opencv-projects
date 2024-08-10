import cv2
import numpy as np
import mediapipe as mp
import math
from pynput.mouse import Button, Controller

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the mouse controller
mouse = Controller()

# Start capturing video
cap = cv2.VideoCapture(0)

def distance_between_points(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

mouse_down = False  # To track if the mouse button is currently held down

# Create a canvas to draw on with the same dimensions as the frame
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Same size as the frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to detect hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for the index finger and thumb
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert landmarks to pixel coordinates
            index_finger_tip_x = int(index_finger_tip.x * frame_width)
            index_finger_tip_y = int(index_finger_tip.y * frame_height)
            thumb_tip_x = int(thumb_tip.x * frame_width)
            thumb_tip_y = int(thumb_tip.y * frame_height)

            # Draw a circle on the index finger tip on the original frame
            cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 10, (0, 255, 0), -1)  # Larger circle for visibility

            # Move the mouse cursor based on the index finger position
            # Adjust these scaling factors as needed
            screen_x = int(index_finger_tip_x * 1920 / frame_width)
            screen_y = int(index_finger_tip_y * 1080 / frame_height)
            mouse.position = (screen_x, screen_y)

            # Draw a darker red dot on the canvas at the index finger position
            if mouse_down == False:
                cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), 10, (0, 0, 100), -1)  # Darker red dot

            # Calculate the distance between index finger tip and thumb tip
            pinch_distance = distance_between_points(
                (index_finger_tip_x, index_finger_tip_y),
                (thumb_tip_x, thumb_tip_y)
            )

            # Define a threshold for pinching
            pinch_threshold = 30

            # Draw a circle around the pinch if detected
            if pinch_distance < pinch_threshold:
                cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 20, (255, 0, 0), 2)
                cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 20, (255, 0, 0), 2)

                # Hold down the mouse button if not already held
                if not mouse_down:
                    mouse.press(Button.left)  # Hold down the mouse button
                    mouse_down = True
            else:
                # Release the mouse button if it was held
                if mouse_down:
                    mouse.release(Button.left)  # Release the mouse button
                    mouse_down = False

    # Combine the canvas with the original frame
    # Increase the weight of the canvas to make drawings more prominent
    combined_frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)  # Blend the canvas on top of the frame

    # Display the resulting frame
    cv2.imshow('Drawing with Finger Tracking', combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()