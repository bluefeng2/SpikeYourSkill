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

# ----------- PART 4: Generate full wireframe overlay video first, then cut for each stage -----------

# Add hands to the wireframe
ALL_JOINTS = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
POSE_CONNECTIONS = [
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), (21, 15),  # Left arm and hand
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 16),  # Right arm and hand
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (11, 23), (12, 24),  # Torso sides
]

output_path = "outputcsv/test_wireframe_duo_full.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

test_len = len(test_df)
donny_len = len(donny_df)

frame_indices_by_stage = {stage: [] for stage in stages}

for idx in range(min(test_len, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, frame = cap.read()
    if not ret:
        break

    # Draw test skeleton (red)
    coords = {}
    if idx < test_len:
        row = test_df.iloc[idx]
        for joint in ALL_JOINTS:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            if pd.notnull(row[x_col]) and pd.notnull(row[y_col]):
                coords[joint] = (int(row[x_col] * width), int(row[y_col] * height))
        for j1, j2 in POSE_CONNECTIONS:
            if j1 in coords and j2 in coords:
                cv2.line(frame, coords[j1], coords[j2], (0, 0, 255), 2)
        for (x, y) in coords.values():
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Draw Donny_1 skeleton (blue), aligned to test
    if idx < donny_len and idx < test_len:
        row_donny = donny_df.iloc[idx]
        coords_donny = {}
        for joint in ALL_JOINTS:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            if pd.notnull(row_donny[x_col]) and pd.notnull(row_donny[y_col]):
                coords_donny[joint] = (float(row_donny[x_col]), float(row_donny[y_col]))

        if 11 in coords and 12 in coords and 11 in coords_donny and 12 in coords_donny:
            g11 = coords[11]
            g12 = coords[12]
            g_center = ((g11[0] + g12[0]) / 2, (g11[1] + g12[1]) / 2)
            g_shoulder_dist = ((g11[0] - g12[0]) ** 2 + (g11[1] - g12[1]) ** 2) ** 0.5

            d11 = (coords_donny[11][0] * width, coords_donny[11][1] * height)
            d12 = (coords_donny[12][0] * width, coords_donny[12][1] * height)
            d_center = ((d11[0] + d12[0]) / 2, (d11[1] + d12[1]) / 2)
            d_shoulder_dist = ((d11[0] - d12[0]) ** 2 + (d11[1] - d12[1]) ** 2) ** 0.5

            scale = g_shoulder_dist / d_shoulder_dist if d_shoulder_dist > 0 else 1.0
            dx = g_center[0] - d_center[0] * scale
            dy = g_center[1] - d_center[1] * scale

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

    # For each stage, record which frames belong to it
    for stage, indices in stage_to_test_frames.items():
        if idx in indices:
            frame_indices_by_stage[stage].append(idx)

out.release()
cap.release()
print(f"Full wireframe overlay video saved to {output_path}")

# ----------- PART 6: Ball position and distance to chest/wrist -----------

import cv2
import csv
from ultralytics import YOLO
import os
import pandas as pd

def detect_ball_yolov8(video_path, output_csv, weights_path='best.pt', conf_thres=0.25):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'x', 'y'])

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_thres, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

            # If any detections, take the one with highest confidence
            if len(boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                idx = confs.argmax()
                x1, y1, x2, y2 = boxes[idx][:4]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                writer.writerow([frame_num, cx, cy])
                # Draw bounding box and center
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'({cx}, {cy})', (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            '''
            cv2.imshow('YOLOv8 Ball Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''
            
            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    
def interpolate_ball_positions(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)
    # Set frame as the index for interpolation
    df = df.set_index('frame')
    # Reindex to include all frames between min and max
    all_frames = range(df.index.min(), df.index.max() + 1)
    df = df.reindex(all_frames)
    # Interpolate missing x and y values linearly
    df['x'] = df['x'].interpolate()
    df['y'] = df['y'].interpolate()
    # Optionally, fill any remaining NaNs (at start/end) with nearest values
    df['x'] = df['x'].fillna(method='bfill').fillna(method='ffill')
    df['y'] = df['y'].fillna(method='bfill').fillna(method='ffill')
    # Reset index and save
    df = df.reset_index().rename(columns={'index': 'frame'})
    df.to_csv(output_csv, index=False)
    
def analyze_ball_and_body(ball_csv, coords_csv, output_csv, width, height):
    import numpy as np
    import pandas as pd
    ball_df = pd.read_csv(ball_csv).set_index('frame')
    coords_df = pd.read_csv(coords_csv).set_index('frame')

    results = []
    contact_frame = None
    contact_time = None
    contact_threshold = 200  # pixels, adjust as needed

    # Use only frames present in both
    common_frames = ball_df.index.intersection(coords_df.index)
    for frame in common_frames:
        bx, by = ball_df.loc[frame, ['x', 'y']]
        s11_x = coords_df.loc[frame, '11_x']
        s11_y = coords_df.loc[frame, '11_y']
        s12_x = coords_df.loc[frame, '12_x']
        s12_y = coords_df.loc[frame, '12_y']
        rw_x = coords_df.loc[frame, '16_x']
        rw_y = coords_df.loc[frame, '16_y']

        # Convert to pixel coordinates if not nan
        if not np.isnan([s11_x, s11_y, s12_x, s12_y]).any():
            s11_x_pix = float(s11_x) * width
            s11_y_pix = float(s11_y) * height
            s12_x_pix = float(s12_x) * width
            s12_y_pix = float(s12_y) * height
            chest_x = (s11_x_pix + s12_x_pix) / 2
            chest_y = (s11_y_pix + s12_y_pix) / 2
        else:
            chest_x, chest_y = np.nan, np.nan

        if not np.isnan([rw_x, rw_y]).any():
            rw_x_pix = float(rw_x) * width
            rw_y_pix = float(rw_y) * height
        else:
            rw_x_pix, rw_y_pix = np.nan, np.nan

        # Distance to chest
        if not np.isnan([bx, by, chest_x, chest_y]).any():
            dist_ball_chest = np.sqrt((bx - chest_x) ** 2 + (by - chest_y) ** 2)
        else:
            dist_ball_chest = np.nan

        # Distance to right wrist (use your provided logic)
        if np.isnan([bx, by, rw_x, rw_y]).any():
            dist_ball_rw = np.nan
        else:
            dist_ball_rw = np.sqrt((bx - rw_x_pix) ** 2 + (by - rw_y_pix) ** 2)
        print(dist_ball_rw)
        # Check for contact
        contact = 0
        if not np.isnan(dist_ball_rw) and dist_ball_rw < contact_threshold:
            if contact_frame is None:
                contact_frame = frame
                # Try to get timestamp if available
                if 'timestamp_milis' in coords_df.columns:
                    contact_time = coords_df.loc[frame, 'timestamp_milis']
                elif 'timestamp_millis' in coords_df.columns:
                    contact_time = coords_df.loc[frame, 'timestamp_millis']
                else:
                    contact_time = None
            contact = 1

        results.append({
            'frame': frame,
            'ball_x': bx,
            'ball_y': by,
            'chest_x': chest_x,
            'chest_y': chest_y,
            'rw_x': rw_x_pix,
            'rw_y': rw_y_pix,
            'dist_ball_chest': dist_ball_chest,
            'dist_ball_right_wrist': dist_ball_rw,
            'contact': contact,
            'contact_frame': contact_frame,
            'contact_time': contact_time
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved ball-body analysis to {output_csv}")
    if contact_frame is not None:
        print(f"First contact frame: {contact_frame}")
        if contact_time is not None:
            print(f"Contact time (ms): {contact_time}")

video_path = "dataAnalysis\\media\\donny_1.mp4"  # Change to your video file
output_csv = "ball_positions.csv"
weights_path = r"dataAnalysis\data\best.pt"
print("Weights exist?", os.path.exists(weights_path), weights_path)
detect_ball_yolov8(video_path, output_csv, weights_path=weights_path)
interpolate_ball_positions(output_csv, 'ball_positions_interpolated.csv')

# Example usage after all previous processing:
# Make sure you have ball_positions_interpolated.csv and test_coords.csv
ball_csv = "ball_positions_interpolated.csv"  # Should be generated by your ball detector/interpolator
coords_csv = "dataAnalysis\\media\\test_coords.csv"
output_csv = "outputcsv/ball_body_analysis.csv"

analyze_ball_and_body(ball_csv, coords_csv, output_csv, width, height)

def plot_ball_wrist_distance(ball_csv, coords_csv, which_wrist='right'):
    import matplotlib.pyplot as plt
    import numpy as np
    ball_df = pd.read_csv(ball_csv).set_index('frame')
    coords_df = pd.read_csv(coords_csv).set_index('frame')

    # Choose wrist index
    wrist_idx = 16 if which_wrist == 'right' else 15
    wrist_x_col = f"{wrist_idx}_x"
    wrist_y_col = f"{wrist_idx}_y"

    # If coords are normalized, you need width/height for pixel conversion
    # Use the same width/height as the video
    width = int(ball_df['x'].max() * 1.1)
    height = int(ball_df['y'].max() * 1.1)

    # Align frames and compute distance
    common_frames = ball_df.index.intersection(coords_df.index)
    distances = []
    for frame in common_frames:
        bx, by = ball_df.loc[frame, ['x', 'y']]
        wx = coords_df.loc[frame, wrist_x_col]
        wy = coords_df.loc[frame, wrist_y_col]
        if np.isnan([bx, by, wx, wy]).any():
            distances.append(np.nan)
        else:
            wx_pix = float(wx) * width
            wy_pix = float(wy) * height
            dist = np.sqrt((bx - wx_pix) ** 2 + (by - wy_pix) ** 2)
            distances.append(dist)

    plt.figure(figsize=(12, 5))
    plt.plot(list(common_frames), distances, label=f'Ball to {which_wrist} wrist distance (pixels)', color='blue')
    plt.xlabel('Frame')
    plt.ylabel('Distance (pixels)')
    plt.title(f'Distance from Ball to {which_wrist.capitalize()} Wrist')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
plot_ball_wrist_distance("ball_positions_interpolated.csv", "dataAnalysis\\media\\test_coords.csv", which_wrist='right')
