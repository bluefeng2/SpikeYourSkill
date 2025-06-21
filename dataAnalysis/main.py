def main(filePath):
    import cv2
    import mediapipe as mp
    import numpy as np
    import csv

    # Initialize MediaPipe pose and hands
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # List of body joints to consider (excluding face)
    body_joints = {
        11: "Left Shoulder", 12: "Right Shoulder",
        13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist",
        23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee",
        27: "Left Ankle", 28: "Right Ankle"
    }

    data = []

    # Open video file (replace 'your_video.mp4' with your video file path)
    cap = cv2.VideoCapture(filePath)

    def calc_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        row = [cap.get(cv2.CAP_PROP_POS_MSEC)]  # Timestamp in milliseconds

        if results.pose_landmarks:
            h, w, _ = frame.shape
            joint_positions = {}
            for idx, name in body_joints.items():
                lm = results.pose_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                joint_positions[name] = (x, y)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                row.extend([lm.x, lm.y, lm.z])  # Normalized coordinates

            # Calculate and display angles for arms and legs
            if all(j in joint_positions for j in ["Left Shoulder", "Left Elbow", "Left Wrist"]):
                angle = calc_angle(joint_positions["Left Shoulder"], joint_positions["Left Elbow"], joint_positions["Left Wrist"])
                cv2.putText(frame, f"L Arm: {angle:.1f}", joint_positions["Left Elbow"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if all(j in joint_positions for j in ["Right Shoulder", "Right Elbow", "Right Wrist"]):
                angle = calc_angle(joint_positions["Right Shoulder"], joint_positions["Right Elbow"], joint_positions["Right Wrist"])
                cv2.putText(frame, f"R Arm: {angle:.1f}", joint_positions["Right Elbow"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if all(j in joint_positions for j in ["Left Hip", "Left Knee", "Left Ankle"]):
                angle = calc_angle(joint_positions["Left Hip"], joint_positions["Left Knee"], joint_positions["Left Ankle"])
                cv2.putText(frame, f"L Leg: {angle:.1f}", joint_positions["Left Knee"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if all(j in joint_positions for j in ["Right Hip", "Right Knee", "Right Ankle"]):
                angle = calc_angle(joint_positions["Right Hip"], joint_positions["Right Knee"], joint_positions["Right Ankle"])
                cv2.putText(frame, f"R Leg: {angle:.1f}", joint_positions["Right Knee"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Calculate and display angles for hips and shoulders
            # Hip angle: Left Shoulder - Left Hip - Right Hip
            if all(j in joint_positions for j in ["Left Shoulder", "Left Hip", "Right Hip"]):
                angle = calc_angle(joint_positions["Left Shoulder"], joint_positions["Left Hip"], joint_positions["Right Hip"])
                cv2.putText(frame, f"L Hip: {angle:.1f}", joint_positions["Left Hip"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # Hip angle: Right Shoulder - Right Hip - Left Hip
            if all(j in joint_positions for j in ["Right Shoulder", "Right Hip", "Left Hip"]):
                angle = calc_angle(joint_positions["Right Shoulder"], joint_positions["Right Hip"], joint_positions["Left Hip"])
                cv2.putText(frame, f"R Hip: {angle:.1f}", joint_positions["Right Hip"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # Shoulder angle: Left Elbow - Left Shoulder - Right Shoulder
            if all(j in joint_positions for j in ["Left Elbow", "Left Shoulder", "Right Shoulder"]):
                angle = calc_angle(joint_positions["Left Elbow"], joint_positions["Left Shoulder"], joint_positions["Right Shoulder"])
                cv2.putText(frame, f"L Shoulder: {angle:.1f}", joint_positions["Left Shoulder"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # Shoulder angle: Right Elbow - Right Shoulder - Left Shoulder
            if all(j in joint_positions for j in ["Right Elbow", "Right Shoulder", "Left Shoulder"]):
                angle = calc_angle(joint_positions["Right Elbow"], joint_positions["Right Shoulder"], joint_positions["Left Shoulder"])
                cv2.putText(frame, f"R Shoulder: {angle:.1f}", joint_positions["Right Shoulder"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw hand landmarks (fingers)
        if hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"F{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    row.extend([lm.x, lm.y, lm.z])  # Normalized coordinates of hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Append the row to data   
        data.append(row)
        
        # Display the frame with pose and hand landmarks
        #cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    print(data)

    def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=1e-2):
        """Simple 1D Kalman filter for a sequence."""
        n = len(data)
        x = np.zeros(n)
        P = np.zeros(n)
        x[0] = data[0]
        P[0] = 1.0
        Q = process_variance
        R = measurement_variance

        for k in range(1, n):
            # Prediction
            x_pred = x[k-1]
            P_pred = P[k-1] + Q

            # Update
            K = P_pred / (P_pred + R)
            x[k] = x_pred + K * (data[k] - x_pred)
            P[k] = (1 - K) * P_pred

        return x

    def apply_kalman_to_all(data):
        """Apply Kalman filter to all columns except timestamp."""
        data_np = np.array(data)
        # Transpose to iterate over columns
        filtered = [data_np[:, 0]]  # Keep timestamp as is
        for i in range(1, data_np.shape[1]):
            filtered_col = kalman_filter_1d(data_np[:, i])
            filtered.append(filtered_col)
        # Transpose back to rows
        return np.array(filtered).T.tolist()

    # Apply Kalman filter to all values except timestamp
    if len(data) > 1:
        data = apply_kalman_to_all(data)

    with open(filePath.split(".")[0] + "_analysis.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # --- Write header row ---
        header = ['timestamp_milis']
        # Pose landmarks (33 points)
        for i in range(33):
            header.extend([f"{i}_x", f"{i}_y", f"{i}_z"])
        # Hand landmarks (21 points per hand, up to 2 hands)
        for hand in range(2):
            for i in range(21):
                header.extend([f"{33 + hand*21 + i}_x", f"{33 + hand*21 + i}_y", f"{33 + hand*21 + i}_z"])
        writer.writerow(header)
        # --- Write all data rows ---
        writer.writerows(data)

    print(f"Analysis saved to {filePath.split('.')[0]}_analysis.csv")

for file in ["dataAnalysis\\testmedia\\1.mp4", "dataAnalysis\\testmedia\\2.mp4", "dataAnalysis\\testmedia\\3.mp4", "dataAnalysis\\testmedia\\badvideo.mp4"]:  # Replace with your video files
    print(f"Processing {file}...")
    main(file)
    print(f"Finished processing {file}.\n")