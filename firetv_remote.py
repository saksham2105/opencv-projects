import cv2
import numpy as np
import mediapipe as mp
import subprocess

# Load the Fire TV remote image (ensure this image is suitable for overlaying)
remote_image_path = 'firetv_remote.jpg'
remote_image = cv2.imread(remote_image_path)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Check if the image was loaded successfully
if remote_image is None:
    print(f"Error: Unable to load image at path {remote_image_path}")
    exit()

# Create an alpha channel for the remote image to simulate transparency
alpha_channel = np.ones((remote_image.shape[0], remote_image.shape[1]), dtype=np.uint8) * 255  # Fully opaque

# Adjust alpha_channel to create a semi-transparent effect
alpha_channel = cv2.addWeighted(alpha_channel, 0.5, np.zeros_like(alpha_channel), 0, 0)  # 50% transparency

# Merge the remote image with the alpha channel
remote_image_bgra = cv2.merge((remote_image, alpha_channel))

# Start capturing video
cap = cv2.VideoCapture(1)

def press_back_button():
    """Send ADB command to press the Back button."""
    try:
        subprocess.run(["adb", "shell", "input", "keyevent", "4"], check=True)
        print("Back button command sent successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error sending ADB command: {e}")

def press_left_button():
    """Send ADB command to press the Back button."""
    try:
        subprocess.run(["adb", "shell", "input", "keyevent", "21"], check=True)
        print("Back button command sent successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error sending ADB command: {e}")

def press_right_button():
    """Send ADB command to press the Back button."""
    try:
        subprocess.run(["adb", "shell", "input", "keyevent", "22"], check=True)
        print("Back button command sent successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error sending ADB command: {e}")

def press_enter_button():
    """Send ADB command to press the Back button."""
    try:
        subprocess.run(["adb", "shell", "input", "keyevent", "23"], check=True)
        print("Back button command sent successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error sending ADB command: {e}")



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Initial size of the remote image
    remote_width = int(frame_width * 0.3)  # Resize to 30% of the frame width
    remote_height = int(remote_image.shape[0] * (remote_width / remote_image.shape[1]))

    # Ensure the remote image fits within the frame dimensions
    if remote_height > frame_height:
        remote_height = frame_height
        remote_width = int(remote_image.shape[1] * (remote_height / remote_image.shape[0]))
    if remote_width > frame_width:
        remote_width = frame_width
        remote_height = int(remote_image.shape[0] * (remote_width / remote_image.shape[1]))

    # Resize the remote image with the alpha channel
    remote_resized = cv2.resize(remote_image_bgra, (remote_width, remote_height))

    # Define the position for overlaying the remote image in the middle of the frame
    x_offset = (frame_width - remote_width) // 2
    y_offset = (frame_height - remote_height) // 2

    # Overlay the remote image
    if remote_resized.shape[2] == 4:
        b_channel, g_channel, r_channel, alpha_channel = cv2.split(remote_resized)

        # Create a mask using the alpha channel
        overlay_mask = alpha_channel.astype(float) / 255
        overlay_region = frame[y_offset:y_offset + remote_height, x_offset:x_offset + remote_width]

        for c in range(0, 3):
            overlay_region[:, :, c] = (
                (1 - overlay_mask) * overlay_region[:, :, c] +
                overlay_mask * remote_resized[:, :, c]
            )
    else:
        # Ensure the overlay fits within the frame
        if y_offset + remote_height > frame_height:
            remote_height = frame_height - y_offset
            remote_resized = remote_resized[:remote_height, :]
        if x_offset + remote_width > frame_width:
            remote_width = frame_width - x_offset
            remote_resized = remote_resized[:, :remote_width]

        # Overlay without transparency
        frame[y_offset:y_offset + remote_height, x_offset:x_offset + remote_width] = remote_resized

    # Process the frame to detect hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for the index finger (landmarks 5 to 8)
            index_finger_tip = hand_landmarks.landmark[8]

            # Convert landmarks to pixel coordinates
            index_finger_tip_x = int(index_finger_tip.x * frame_width)
            index_finger_tip_y = int(index_finger_tip.y * frame_height)

            # Draw a circle on the index finger tip
            cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 10, (0, 255, 0), -1)

            # Draw a line from the index finger tip to the base
            index_finger_mcp = hand_landmarks.landmark[5]
            index_finger_mcp_x = int(index_finger_mcp.x * frame_width)
            index_finger_mcp_y = int(index_finger_mcp.y * frame_height)
            cv2.line(frame, (index_finger_mcp_x, index_finger_mcp_y), (index_finger_tip_x, index_finger_tip_y), (255, 0, 0), 2)

            # Define the bounding box of the remote image overlay
            remote_box = {
                'x_min': x_offset,
                'y_min': y_offset,
                'x_max': x_offset + remote_width,
                'y_max': y_offset + remote_height
            }

            # Check if the index finger tip is within the remote image overlay
            if (remote_box['x_min'] <= index_finger_tip_x <= remote_box['x_max'] and
                remote_box['y_min'] <= index_finger_tip_y <= remote_box['y_max']):
                print(f"Index Finger Tip Coordinates: ({index_finger_tip_x}, {index_finger_tip_y})")
                
                # Define regions for buttons
                power_button_region = {
                    'x_min': 576,
                    'y_min': 40,
                    'x_max': 596,  # Example values, adjust as needed
                    'y_max': 60
                }
                
                back_button_region = {
                    'x_min': 576,
                    'y_min': 310,
                    'x_max': 584,
                    'y_max': 317
                }

                left_button_region = {
                    'x_min': 576,
                    'y_min': 146,
                    'x_max': 600,
                    'y_max': 246
                }
                right_button_region = {
                    'x_min': 671,
                    'y_min': 146,
                    'x_max': 715,
                    'y_max': 246
                }
                enter_button_region = {
                    'x_min': 607,
                    'y_min': 146,
                    'x_max': 670,
                    'y_max': 246
                }

                up_button_region = {
                    'x_min': 576,
                    'y_min': 150,
                    'x_max': 600,
                    'y_max': 290
                } 
                down_button_region = {
                    'x_min': 576,
                    'y_min': 150,
                    'x_max': 600,
                    'y_max': 290
                } 

                # Check if the index finger tip is within any defined button regions
                if (power_button_region['x_min'] <= index_finger_tip_x <= power_button_region['x_max'] and
                    power_button_region['y_min'] <= index_finger_tip_y <= power_button_region['y_max']):
                    print("Power button pressed")

                if (back_button_region['x_min'] <= index_finger_tip_x <= back_button_region['x_max'] and
                    back_button_region['y_min'] <= index_finger_tip_y <= back_button_region['y_max']):
                    print("Back button pressed")
                    press_back_button()  # Call the function to press the back button

                if (left_button_region['x_min'] <= index_finger_tip_x <= left_button_region['x_max'] and
                    left_button_region['y_min'] <= index_finger_tip_y <= left_button_region['y_max']):
                    print("Left button pressed")
                    press_left_button()  # Call the function to press the back button    

                if (right_button_region['x_min'] <= index_finger_tip_x <= right_button_region['x_max'] and
                    right_button_region['y_min'] <= index_finger_tip_y <= right_button_region['y_max']):
                    print("Right button pressed")
                    press_right_button()  # Call the function to press the back button    

                if (enter_button_region['x_min'] <= index_finger_tip_x <= enter_button_region['x_max'] and
                    enter_button_region['y_min'] <= index_finger_tip_y <= enter_button_region['y_max']):
                    print("Enter button pressed")
                    press_enter_button()  # Call the function to press the back button        

    # Display the resulting frame
    cv2.imshow('Video with Remote Overlay and Finger Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
