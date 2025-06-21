import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load your image
image = cv2.imread('serve.jpg')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find pose landmarks
results = pose.process(image_rgb)

# List of body joints to consider (excluding face)
body_joints = {
    11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow", 14: "Right Elbow",
    15: "Left Wrist", 16: "Right Wrist",
    23: "Left Hip", 24: "Right Hip",
    25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle"
}

if results.pose_landmarks:
    h, w, _ = image.shape
    joint_positions = {}
    for idx, name in body_joints.items():
        lm = results.pose_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        joint_positions[name] = (x, y)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Example: Calculate angles for arms and legs
    def calc_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Left Arm
    if all(j in joint_positions for j in ["Left Shoulder", "Left Elbow", "Left Wrist"]):
        angle = calc_angle(joint_positions["Left Shoulder"], joint_positions["Left Elbow"], joint_positions["Left Wrist"])
        print(f"Left Arm Angle: {angle:.2f} degrees")

    # Right Arm
    if all(j in joint_positions for j in ["Right Shoulder", "Right Elbow", "Right Wrist"]):
        angle = calc_angle(joint_positions["Right Shoulder"], joint_positions["Right Elbow"], joint_positions["Right Wrist"])
        print(f"Right Arm Angle: {angle:.2f} degrees")

    # Left Leg
    if all(j in joint_positions for j in ["Left Hip", "Left Knee", "Left Ankle"]):
        angle = calc_angle(joint_positions["Left Hip"], joint_positions["Left Knee"], joint_positions["Left Ankle"])
        print(f"Left Leg Angle: {angle:.2f} degrees")

    # Right Leg
    if all(j in joint_positions for j in ["Right Hip", "Right Knee", "Right Ankle"]):
        angle = calc_angle(joint_positions["Right Hip"], joint_positions["Right Knee"], joint_positions["Right Ankle"])
        print(f"Right Leg Angle: {angle:.2f} degrees")

    # Show the image with joints
    cv2.imshow('Pose Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose detected.")
