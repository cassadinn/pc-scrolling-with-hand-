import cv2
import mediapipe as mp
import pyautogui
import math
import keyboard

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define default scroll amount
default_scroll_amount = 155

def is_fist(hand_landmarks):
    """Check if the hand is making a fist."""
    finger_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    finger_base_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    for tip_id, base_id in zip(finger_tips_ids, finger_base_ids):
        tip = hand_landmarks.landmark[tip_id]
        base = hand_landmarks.landmark[base_id]
        if tip.y < base.y:  # If the tip is higher than the base, the finger is not curled in a fist
            return False
    return True

def is_index_finger_up(hand_landmarks):
    """Check if the index finger is up."""
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    return tip.y < dip.y

def are_index_and_middle_fingers_up(hand_landmarks):
    """Check if the index and middle fingers are up."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    return (index_tip.y < index_dip.y) and (middle_tip.y < middle_dip.y)

def is_hand_open(hand_landmarks):
    """Check if the hand is open (all fingers extended)."""
    finger_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP
    ]
    
    finger_dip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_DIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp_hands.HandLandmark.RING_FINGER_DIP,
        mp_hands.HandLandmark.PINKY_DIP,
        mp_hands.HandLandmark.THUMB_IP
    ]
    
    for tip_id, dip_id in zip(finger_tips_ids, finger_dip_ids):
        tip = hand_landmarks.landmark[tip_id]
        dip = hand_landmarks.landmark[dip_id]
        if tip.y >= dip.y:  # If any tip is not higher than its DIP, the hand is not open
            return False
    return True

def are_thumb_and_index_tips_together(hand_landmarks, threshold=0.05):
    """Check if the thumb and index finger tips are close to each other."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < threshold

# Flag to stop the loop
stop = False

# Function to stop the loop when 'Esc' is pressed
def stop_program():
    global stop
    stop = True

# Register the key press event
keyboard.add_hotkey('esc', stop_program)

while True:
    ret, frame = cap.read()
    if not ret or stop:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)

    closest_hand = None
    closest_distance = float('inf')

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the depth of the hand (distance from the camera)
            z = hand_landmarks.landmark[0].z  # The first landmark (wrist) is used as a reference
            # Update closest hand if the current hand is closer to the camera
            if z < closest_distance:
                closest_hand = hand_landmarks
                closest_distance = z

    if closest_hand:
        # Draw hand landmarks on the frame (for debugging purposes, you can comment this out)
        mp_draw.draw_landmarks(frame, closest_hand, mp_hands.HAND_CONNECTIONS)

        if is_fist(closest_hand):
            continue  # If fist is detected, skip scrolling

        if is_hand_open(closest_hand):
            continue  # If the hand is open, skip scrolling

        if are_thumb_and_index_tips_together(closest_hand):
            continue  # If thumb and index fingertips are together, skip scrolling

        if are_index_and_middle_fingers_up(closest_hand):
            pyautogui.scroll(default_scroll_amount)
        elif is_index_finger_up(closest_hand):
            pyautogui.scroll(-default_scroll_amount)

    # Break the loop if 'Esc' key is pressed
    if stop:
        break

    # Note: We are not using cv2.imshow() to avoid opening the camera window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Remove the hotkey registration
keyboard.remove_hotkey('esc')
