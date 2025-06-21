import os
import cv2
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw_path
import matplotlib.cm as cm

# ----------- PART 1: Generate angles and coords CSVs for all videos -----------

import mediapipe as mp

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_video(filePath):
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
            for idxs in relevant_joints:
                try:
                    a = coords[idxs[0]]
                    b = coords[idxs[1]]
                    c = coords[idxs[2]]
                    angle = calc_angle(a, b, c)
                except Exception:
                    angle = ""
                row.append(angle)
        else:
            row += [""] * len(relevant_joints)
            for idx in range(33):
                coords_row[f"{idx}_x"] = ""
                coords_row[f"{idx}_y"] = ""

        angle_data.append(row)
        coords_data.append(coords_row)

    cap.release()

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

# Process all videos
video_files = [
    "dataAnalysis\\media\\donny_1.mp4",
    "dataAnalysis\\media\\donny_2.mp4",
    "dataAnalysis\\media\\donny_3.mp4",
    "dataAnalysis\\media\\test.mp4"
]
for file in video_files:
    print(f"Processing {file}...")
    process_video(file)
    print(f"Finished processing {file}.\n")

# ----------- PART 2: DTW, Stage Mapping, and Overlay Video Generation -----------

# Load CSVs
proPaths = [
    "dataAnalysis\\media\\donny_1_angles.csv",
    "dataAnalysis\\media\\donny_2_angles.csv",
    "dataAnalysis\\media\\donny_3_angles.csv"
]
testPath = "dataAnalysis\\media\\test_angles.csv"
proCSVS = [pd.read_csv(proPath) for proPath in proPaths]
testCSV = pd.read_csv(testPath)

# Interpolate missing values
for df in proCSVS:
    df.interpolate().bfill().ffill(inplace=True)
testCSV.interpolate().bfill().ffill(inplace=True)

# Get all joint columns (skip timestamp)
joint_columns = [col for col in proCSVS[0].columns if col != "timestamp_milis"]

# Convert to numpy arrays (frames, joints)
pro_arrays = [df[joint_columns].values for df in proCSVS]
test_array = testCSV[joint_columns].values

# Use the first pro as reference, align others to it
ref = pro_arrays[0]
aligned_pros = [ref]
pro_dtw_mappings = {0: list(range(len(ref)))}

for i, arr in enumerate(pro_arrays[1:], 1):
    path, _ = dtw_path(ref, arr)
    mapping = []
    aligned_arr = []
    last_j = 0
    for j1 in range(len(ref)):
        matches = [j2 for (pj1, j2) in path if pj1 == j1]
        if matches:
            mapping.append(matches[0])
            aligned_arr.append(arr[matches[0]])
            last_j = matches[0]
        else:
            mapping.append(last_j)
            aligned_arr.append(arr[last_j])
    aligned_pros.append(np.array(aligned_arr))
    pro_dtw_mappings[i] = mapping

# Average the aligned pro videos
pro_avg = np.mean(np.stack(aligned_pros), axis=0)
pro_avg_df = pd.DataFrame(pro_avg, columns=joint_columns)

# DTW align test to pro_avg
test_path, _ = dtw_path(pro_avg, test_array)
test_mapping = []
aligned_test = []
last_j = 0
for j1 in range(len(pro_avg)):
    matches = [j2 for (pj1, j2) in test_path if pj1 == j1]
    if matches:
        test_mapping.append(matches[0])
        aligned_test.append(test_array[matches[0]])
        last_j = matches[0]
    else:
        test_mapping.append(last_j)
        aligned_test.append(test_array[last_j])
aligned_test = np.array(aligned_test)
test_aligned_df = pd.DataFrame(aligned_test, columns=joint_columns)

# --- Stage definitions from aligned (DTW) frames (1-based in your notes, so subtract 1 for Python) ---
stages = {
    "preparation": list(range(0, 13)),      # frames 1-13 -> 0-12
    "throw": list(range(13, 26)),           # frames 14-26 -> 13-25
    "precontact": list(range(26, 39)),      # frames 27-39 -> 26-38
    "swing": list(range(39, 44)),           # frames 40-44 -> 39-43
    "contact": list(range(44, 46)),         # frames 45-46 -> 44-45
    "followthrough": list(range(46, 57)),   # frames 47-57 -> 46-56
}

# Map aligned frame indices to original test frame indices for each stage
stage_to_test_frames = {}
for stage, aligned_indices in stages.items():
    test_indices = [test_mapping[i] for i in aligned_indices if i < len(test_mapping)]
    stage_to_test_frames[stage] = test_indices

with open("outputcsv/test_stage_frame_indices.json", "w") as f:
    json.dump(stage_to_test_frames, f, indent=2)

# ----------- PART 3: Overlay Stage Videos -----------

os.makedirs("outputcsv/test_stages_overlay", exist_ok=True)

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (11, 23), (12, 24),  # Torso sides
]

# Load joint coordinates for test and Donny_1
test_df = pd.read_csv("dataAnalysis\\media\\test_coords.csv")
donny_df = pd.read_csv("dataAnalysis\\media\\donny_1_coords.csv")

# Open the source test video
video_path = 'dataAnalysis\\media\\test.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for stage, test_indices in stage_to_test_frames.items():
    # Donny_1 indices for this stage (aligned to original Donny_1, not DTW)
    donny_indices = stages[stage]
    out_path = f"outputcsv/test_stages_overlay/test_{stage}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i, idx in enumerate(test_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw test skeleton (red)
        coords = {}
        if idx < len(test_df):
            row = test_df.iloc[idx]
            for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                if pd.notnull(row[x_col]) and pd.notnull(row[y_col]):
                    coords[joint] = (int(row[x_col] * width), int(row[y_col] * height))
            for j1, j2 in POSE_CONNECTIONS:
                if j1 in coords and j2 in coords:
                    cv2.line(frame, coords[j1], coords[j2], (0, 0, 255), 2)
            for (x, y) in coords.values():
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw Donny_1 skeleton (blue), using the corresponding stage frame, aligned to test
        if i < len(donny_indices):
            donny_idx = donny_indices[i]
            if donny_idx < len(donny_df):
                row_donny = donny_df.iloc[donny_idx]
                coords_donny = {}
                for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                    x_col = f"{joint}_x"
                    y_col = f"{joint}_y"
                    if pd.notnull(row_donny[x_col]) and pd.notnull(row_donny[y_col]):
                        coords_donny[joint] = (float(row_donny[x_col]), float(row_donny[y_col]))

                # Only align if both shoulders are present for both skeletons
                if 11 in coords and 12 in coords and 11 in coords_donny and 12 in coords_donny:
                    # test's chest center and scale
                    g11 = coords[11]
                    g12 = coords[12]
                    g_center = ((g11[0] + g12[0]) / 2, (g11[1] + g12[1]) / 2)
                    g_shoulder_dist = ((g11[0] - g12[0]) ** 2 + (g11[1] - g12[1]) ** 2) ** 0.5

                    # Donny's chest center and scale (normalized, so scale to image size)
                    d11 = (coords_donny[11][0] * width, coords_donny[11][1] * height)
                    d12 = (coords_donny[12][0] * width, coords_donny[12][1] * height)
                    d_center = ((d11[0] + d12[0]) / 2, (d11[1] + d12[1]) / 2)
                    d_shoulder_dist = ((d11[0] - d12[0]) ** 2 + (d11[1] - d12[1]) ** 2) ** 0.5

                    # Compute scale and translation
                    scale = g_shoulder_dist / d_shoulder_dist if d_shoulder_dist > 0 else 1.0
                    dx = g_center[0] - d_center[0] * scale
                    dy = g_center[1] - d_center[1] * scale

                    # Transform and draw Donny's skeleton
                    coords_donny_trans = {}
                    for joint, (x, y) in coords_donny.items():
                        x_img = x * width * scale + dx
                        y_img = y * height * scale + dy
                        coords_donny_trans[joint] = (int(x_img), int(y_img))
                    for j1, j2 in POSE_CONNECTIONS:
                        if j1 in coords_donny_trans and j2 in coords_donny_trans:
                            cv2.line(frame, coords_donny_trans[j1], coords_donny_trans[j2], (255, 0, 0), 2)
                    for (x, y) in coords_donny_trans.values():
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                else:
                    # fallback: draw Donny in original position if alignment not possible
                    coords_donny_img = {}
                    for joint, (x, y) in coords_donny.items():
                        x_img = int(x * width)
                        y_img = int(y * height)
                        coords_donny_img[joint] = (x_img, y_img)
                    for j1, j2 in POSE_CONNECTIONS:
                        if j1 in coords_donny_img and j2 in coords_donny_img:
                            cv2.line(frame, coords_donny_img[j1], coords_donny_img[j2], (255, 0, 0), 2)
                    for (x, y) in coords_donny_img.values():
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        out.write(frame)
    out.release()
    print(f"Saved {out_path}")

cap.release()
print("Overlay stage videos saved to outputcsv/test_stages_overlay/")