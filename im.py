import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Load images (real color, no filter)
repulsor_img = Image.open("beam.png").convert("RGBA")
arc_reactor_img = Image.open("iron-man-arc.png").convert("RGBA")

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# Webcam setup
cap = cv2.VideoCapture(0)

def overlay_transparent(background_bgr, overlay_rgba, x, y):
    try:
        background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
        background_pil = Image.fromarray(background_rgb)
        background_pil.paste(overlay_rgba, (x, y), overlay_rgba)
        return cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("[ERROR] Overlay failed:", e)
        return background_bgr

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # 🌀 Arc Reactor (Chest)
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        try:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            lx, ly = int(left_shoulder.x * img_w), int(left_shoulder.y * img_h)
            rx, ry = int(right_shoulder.x * img_w), int(right_shoulder.y * img_h)

            # Distance between shoulders (in pixels)
            shoulder_dist_px = np.linalg.norm([lx - rx, ly - ry])

            # Scale arc reactor based on distance
            arc_size = int(np.clip(shoulder_dist_px * 0.45, 50, 100))  # Smaller and clamped

            arc_resized = arc_reactor_img.resize((arc_size, arc_size), Image.LANCZOS)

            # Arc reactor position: between shoulders, slightly below
            chest_x = int((lx + rx) / 2) - arc_size // 2
            chest_y = int((ly + ry) / 2) - arc_size // 2 + 70

            frame = overlay_transparent(frame, arc_resized, chest_x, chest_y)
        except Exception as e:
            print("[ERROR] Arc overlay:", e)

    # ✋ Repulsor (Palm)
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            try:
                # Get pixel coordinates
                index_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_lm = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                x1, y1 = int(index_lm.x * img_w), int(index_lm.y * img_h)
                x2, y2 = int(pinky_lm.x * img_w), int(pinky_lm.y * img_h)

                palm_center_x = int((x1 + x2) / 2)
                palm_center_y = int((y1 + y2) / 2) +30

                hand_pixel_dist = np.linalg.norm([x1 - x2, y1 - y2])

                # Ensure repulsor isn't too small
                repulsor_size = int(np.clip(hand_pixel_dist * 2.0, 60, 140))  # Set min to 60
                print(f"[DEBUG] Repulsor size: {repulsor_size}px at ({palm_center_x}, {palm_center_y})")

                repulsor_resized = repulsor_img.resize((repulsor_size, repulsor_size), Image.LANCZOS)

                repulsor_x = palm_center_x - repulsor_size // 2
                repulsor_y = palm_center_y - repulsor_size // 2 + 10  # Adjusted lower

                # Optional: draw a red circle to show center
                cv2.circle(frame, (palm_center_x, palm_center_y), 5, (0, 0, 255), -1)

                frame = overlay_transparent(frame, repulsor_resized, repulsor_x, repulsor_y)
            except Exception as e:
                print("[ERROR] Repulsor overlay:", e)

    cv2.imshow("Iron Man AR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
