import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
import os
import matplotlib.pyplot as plt

def coords_to_array(coords_df):
    joint_indices = [11,12,13,14,15,16,23,24,25,26,27,28]
    arr = []
    for idx, row in coords_df.iterrows():
        frame_data = []
        valid = True
        for joint in joint_indices:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            if pd.isnull(row[x_col]) or pd.isnull(row[y_col]) or row[x_col] == "" or row[y_col] == "":
                valid = False
                break
            frame_data.extend([float(row[x_col]), float(row[y_col])])
        if valid:
            arr.append(frame_data)
    return np.array(arr)

def draw_wireframe(frame, coords, color, width, height):
    POSE_CONNECTIONS = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (11, 23), (12, 24),  # Torso sides
    ]
    for j1, j2 in POSE_CONNECTIONS:
        if j1 in coords and j2 in coords:
            cv2.line(frame, coords[j1], coords[j2], color, 2)
    for (x, y) in coords.values():
        cv2.circle(frame, (x, y), 5, color, -1)

def overlay_with_dtw(test_video, pro_coords_csv, test_coords_csv, output_path):
    # Load coords
    test_coords_df = pd.read_csv(test_coords_csv)
    pro_coords_df = pd.read_csv(pro_coords_csv)

    # DTW mapping: pro to test, using all joints
    pro_arr = coords_to_array(pro_coords_df)
    test_arr = coords_to_array(test_coords_df)
    path, _ = dtw_path(test_arr, pro_arr)
    # path: (test_idx, pro_idx) pairs

    # Build mapping: for each test frame, which pro frame is aligned
    test_to_pro = {}
    for test_idx, pro_idx in path:
        test_to_pro[test_idx] = pro_idx

    cap = cv2.VideoCapture(test_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw test wireframe (blue)
        coords = {}
        if idx < len(test_coords_df):
            row = test_coords_df.iloc[idx]
            for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                if pd.notnull(row[x_col]) and pd.notnull(row[y_col]) and row[x_col] != "" and row[y_col] != "":
                    coords[joint] = (int(float(row[x_col]) * width), int(float(row[y_col]) * height))
            draw_wireframe(frame, coords, (255, 0, 0), width, height)

        # Draw pro wireframe (red), mapped to test frame
        pro_idx = test_to_pro.get(idx, 0)
        coords_pro = {}
        if pro_idx < len(pro_coords_df):
            row_pro = pro_coords_df.iloc[pro_idx]
            for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                if pd.notnull(row_pro[x_col]) and pd.notnull(row_pro[y_col]) and row_pro[x_col] != "" and row_pro[y_col] != "":
                    coords_pro[joint] = (float(row_pro[x_col]), float(row_pro[y_col]))

            # --- Affine transform: keep chest and hips corners close, scale everything else ---
            chest_hips_joints = [11, 12, 23, 24]  # LShoulder, RShoulder, LHip, RHip

            # Only apply if all four points are present in both
            if all(j in coords for j in chest_hips_joints) and all(j in coords_pro for j in chest_hips_joints):
                # Get test and pro chest/hip corners in pixel coordinates
                src = np.float32([
                    [coords_pro[11][0] * width, coords_pro[11][1] * height],
                    [coords_pro[12][0] * width, coords_pro[12][1] * height],
                    [coords_pro[23][0] * width, coords_pro[23][1] * height],
                    [coords_pro[24][0] * width, coords_pro[24][1] * height],
                ])
                dst = np.float32([
                    coords[11],
                    coords[12],
                    coords[23],
                    coords[24],
                ])
                # Find affine transform (least-squares for 4 points)
                # Use cv2.getPerspectiveTransform for 4 points
                M = cv2.getPerspectiveTransform(src, dst)

                coords_pro_trans = {}
                for joint, (x, y) in coords_pro.items():
                    pt = np.array([[x * width, y * height]], dtype=np.float32)
                    pt = np.array([pt])
                    pt_trans = cv2.perspectiveTransform(pt, M)[0][0]
                    coords_pro_trans[joint] = (int(pt_trans[0]), int(pt_trans[1]))

                # Limit the length of each limb to at most 1.5x the test limb length
                LIMB_PAIRS = [
                    (11, 13), (13, 15),  # Left arm
                    (12, 14), (14, 16),  # Right arm
                    (23, 25), (25, 27),  # Left leg
                    (24, 26), (26, 28),  # Right leg
                    (11, 12),            # Shoulders
                    (23, 24),            # Hips
                    (11, 23), (12, 24),  # Torso sides
                ]
                MAX_LIMB_FACTOR = 1.5

                for j1, j2 in LIMB_PAIRS:
                    if j1 in coords and j2 in coords and j1 in coords_pro_trans and j2 in coords_pro_trans:
                        # Test limb length
                        test_len = np.linalg.norm(np.array(coords[j1]) - np.array(coords[j2]))
                        # Pro limb length (transformed)
                        pro_pt1 = np.array(coords_pro_trans[j1])
                        pro_pt2 = np.array(coords_pro_trans[j2])
                        pro_len = np.linalg.norm(pro_pt1 - pro_pt2)
                        max_len = MAX_LIMB_FACTOR * test_len
                        if pro_len > max_len and pro_len > 0 and test_len > 0:
                            # Shorten the pro limb to max_len, keeping the midpoint the same
                            mid = (pro_pt1 + pro_pt2) / 2
                            direction = (pro_pt2 - pro_pt1) / pro_len
                            new_vec = direction * (max_len / 2)
                            coords_pro_trans[j1] = tuple((mid - new_vec).astype(int))
                            coords_pro_trans[j2] = tuple((mid + new_vec).astype(int))

                draw_wireframe(frame, coords_pro_trans, (0, 0, 255), width, height)
            else:
                # fallback: just draw in place
                coords_pro_img = {joint: (int(x * width), int(y * height)) for joint, (x, y) in coords_pro.items()}
                draw_wireframe(frame, coords_pro_img, (0, 0, 255), width, height)

        out.write(frame)

    out.release()
    cap.release()
    print(f"Overlay video saved to {output_path}")

def main(filePath):
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

        cv2.imshow('Pose with Angles', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

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

def plot_joint_trajectories(test_coords_csv, pro_coords_csv, output_dir="videoanalysisonly/outputcsv/joint_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(test_coords_csv)
    pro_df = pd.read_csv(pro_coords_csv)
    joint_indices = [11,12,13,14,15,16,23,24,25,26,27,28]

    for joint in joint_indices:
        test_x = test_df[f"{joint}_x"].astype(float)
        test_y = test_df[f"{joint}_y"].astype(float)
        pro_x = pro_df[f"{joint}_x"].astype(float)
        pro_y = pro_df[f"{joint}_y"].astype(float)

        # Plot X trajectory
        plt.figure(figsize=(10, 4))
        plt.plot(test_x, label="Test X", color="blue")
        plt.plot(pro_x, label="Pro X", color="red")
        plt.title(f"Joint {joint} X trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Normalized X")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"joint_{joint}_x.png"))
        plt.close()

        # Plot Y trajectory
        plt.figure(figsize=(10, 4))
        plt.plot(test_y, label="Test Y", color="blue")
        plt.plot(pro_y, label="Pro Y", color="red")
        plt.title(f"Joint {joint} Y trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Normalized Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"joint_{joint}_y.png"))
        plt.close()

def plot_joint_angle_trajectories(test_angles_csv, pro_angles_csv, output_dir="videoanalysisonly/outputcsv/angle_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(test_angles_csv)
    pro_df = pd.read_csv(pro_angles_csv)

    # Get joint names from columns (skip timestamp)
    joint_names = [col for col in test_df.columns if col != "timestamp_milis"]

    for joint in joint_names:
        test_angle = test_df[joint].astype(float)
        pro_angle = pro_df[joint].astype(float)

        plt.figure(figsize=(10, 4))
        plt.plot(test_angle, label="Test", color="blue")
        plt.plot(pro_angle, label="Pro", color="red")
        plt.title(f"{joint} Angle Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.tight_layout()
        # Save with joint name in filename
        safe_joint = joint.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_joint}_angle.png"))
        plt.close()

# Example usage:
for file in [
    "videoanalysisonly\\test.mp4",
    "videoanalysisonly\\good.mp4"
]:
    print(f"Processing {file}...")
    main(file)
    print(f"Finished processing {file}.\n")

# Overlay: pro mapped to test, both drawn on test video
overlay_with_dtw(
    test_video="videoanalysisonly/test.mp4",
    pro_coords_csv="videoanalysisonly/good_coords.csv",
    test_coords_csv="videoanalysisonly/test_coords.csv",
    output_path="videoanalysisonly/outputcsv/test_overlay_dtw.mp4"
)

# Plot joint trajectories
plot_joint_trajectories(
    test_coords_csv="videoanalysisonly/test_coords.csv",
    pro_coords_csv="videoanalysisonly/good_coords.csv",
    output_dir="videoanalysisonly/outputcsv/joint_graphs"
)

# Plot joint angle trajectories
plot_joint_angle_trajectories(
    test_angles_csv="videoanalysisonly/test_angles.csv",
    pro_angles_csv="videoanalysisonly/good_angles.csv",
    output_dir="videoanalysisonly/outputcsv/angle_graphs"
)

