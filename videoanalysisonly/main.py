import csv

def makeData(filePath):
    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(filePath)

    relevant_joints = [
        ("11", "13", "15"),  # Left Arm (Shoulder-Elbow-Wrist)
        ("12", "14", "16"),  # Right Arm (Shoulder-Elbow-Wrist)
        ("23", "25", "27"),  # Left Leg (Hip-Knee-Ankle)
        ("24", "26", "28"),  # Right Leg (Hip-Knee-Ankle)
        ("13", "11", "23"),  # Left Shoulder (Elbow-Shoulder-Hip)
        ("14", "12", "24"),  # Right Shoulder (Elbow-Shoulder-Hip)
        ("11", "23", "12"),  # Chest (LShoulder-LHip-RShoulder)
        ("12", "24", "11"),  # Chest (RShoulder-RHip-LShoulder)
    ]
    joint_names = {
        ("11", "13", "15"): "Left Arm",
        ("12", "14", "16"): "Right Arm",
        ("23", "25", "27"): "Left Leg",
        ("24", "26", "28"): "Right Leg",
        ("13", "11", "23"): "Left Shoulder",
        ("14", "12", "24"): "Right Shoulder",
        ("11", "23", "12"): "Chest L",
        ("12", "24", "11"): "Chest R",
    }

    def calc_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    angle_data = []
    coords_data = []

    while cap.isOpened():    
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        row = [cap.get(cv2.CAP_PROP_POS_MSEC)]  # Timestamp in milliseconds

            
        flip_state = None  # global or outer scope variable

        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]

            # Decision based on visibility
            if left_shoulder.visibility < right_shoulder.visibility:
                flip_state = True
            elif right_shoulder.visibility < left_shoulder.visibility:
                flip_state = False
            else:
                # Fallback on position
                flip_state = left_shoulder.x > right_shoulder.x
                
            # Get left and right shoulder positions
            left_shoulder_x = results.pose_landmarks.landmark[11].x
            right_shoulder_x = results.pose_landmarks.landmark[12].x

            flipped = left_shoulder_x > right_shoulder_x
            if flipped or flip_state:
                # Swap relevant joint definitions
                relevant_joints = [
                    ("12", "14", "16"),  # Left Arm -> Right Arm
                    ("11", "13", "15"),  # Right Arm -> Left Arm
                    ("24", "26", "28"),  # Left Leg -> Right Leg
                    ("23", "25", "27"),  # Right Leg -> Left Leg
                    ("14", "12", "24"),  # Left Shoulder -> Right Shoulder
                    ("13", "11", "23"),  # Right Shoulder -> Left Shoulder
                    ("12", "24", "11"),  # Chest L -> Chest R
                    ("11", "23", "12"),  # Chest R -> Chest L
                ]
                joint_names = {
                    ("12", "14", "16"): "Left Arm",
                    ("11", "13", "15"): "Right Arm",
                    ("24", "26", "28"): "Left Leg",
                    ("23", "25", "27"): "Right Leg",
                    ("14", "12", "24"): "Left Shoulder",
                    ("13", "11", "23"): "Right Shoulder",
                    ("12", "24", "11"): "Chest L",
                    ("11", "23", "12"): "Chest R",
                }
            else:
                # Reset to original
                relevant_joints = [
                    ("11", "13", "15"),  # Left Arm
                    ("12", "14", "16"),  # Right Arm
                    ("23", "25", "27"),  # Left Leg
                    ("24", "26", "28"),  # Right Leg
                    ("13", "11", "23"),  # Left Shoulder
                    ("14", "12", "24"),  # Right Shoulder
                    ("11", "23", "12"),  # Chest L
                    ("12", "24", "11"),  # Chest R
                ]
                joint_names = {
                    ("11", "13", "15"): "Left Arm",
                    ("12", "14", "16"): "Right Arm",
                    ("23", "25", "27"): "Left Leg",
                    ("24", "26", "28"): "Right Leg",
                    ("13", "11", "23"): "Left Shoulder",
                    ("14", "12", "24"): "Right Shoulder",
                    ("11", "23", "12"): "Chest L",
                    ("12", "24", "11"): "Chest R",
                }
        

        
        coords = {}
        coords_row = {'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1}
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for idx in range(33):
                lm = results.pose_landmarks.landmark[idx]
                coords[str(idx)] = [lm.x, lm.y]
                coords_row[f"{idx}_x"] = lm.x
                coords_row[f"{idx}_y"] = lm.y
            # Draw and label angles
            for idxs in relevant_joints:
                try:
                    a = coords[idxs[0]]
                    b = coords[idxs[1]]
                    c = coords[idxs[2]]
                    angle = calc_angle(a, b, c)
                    # Draw label at joint b
                    px = int(b[0] * w)
                    py = int(b[1] * h)
                    cv2.putText(frame, f"{joint_names[idxs]}: {angle:.1f}", (px, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                except Exception:
                    angle = ""
                row.append(angle)
        else:
            # If no pose detected, fill with empty angles and coords
            row += [""] * len(relevant_joints)
            for idx in range(33):
                coords_row[f"{idx}_x"] = ""
                coords_row[f"{idx}_y"] = ""

        angle_data.append(row)
        coords_data.append(coords_row)

        #cv2.imshow('Pose with Angles', frame)
        #if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        #    break

    cap.release()
    cv2.destroyAllWindows()

    # Write angles to CSV
    with open(filePath.split(".")[0] + "_angles.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ['timestamp_milis'] + [joint_names[j] for j in relevant_joints]
        writer.writerow(header)
        writer.writerows(angle_data)

    # Write coords to CSV
    coords_df = pd.DataFrame(coords_data)
    coords_df.to_csv(filePath.split(".")[0] + "_coords.csv", index=False)

    print(f"Angles saved to {filePath.split('.')[0]}_angles.csv")
    print(f"Coords saved to {filePath.split('.')[0]}_coords.csv")

# Example usage:
for file in [
    "betteranalysis\\test.mp4",
    "betteranalysis\\good.mp4"
]:
    print(f"Processing {file}...")
    makeData(file)
    print(f"Finished processing {file}.\n")
    
