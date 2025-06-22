from flask import Flask, request, send_file
from flask import jsonify
from flask import request
import os
import zipfile
import os
from io import BytesIO
from flask_cors import CORS
def bigasdmaina(locat):
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
        os.path.join("Script", "dataAnalysis", "media", "donny_1.mp4"),
        os.path.join("Script", "dataAnalysis", "media", "donny_2.mp4"),
        os.path.join("Script", "dataAnalysis", "media", "donny_3.mp4"),
        os.path.join("Script", "dataAnalysis", "media", "test.mp4")
    ]
    for file in video_files:
        print(f"Processing {file}...")
        process_video(file)
        print(f"Finished processing {file}.\n")

    # ----------- PART 2: DTW, Stage Mapping, and Overlay Video Generation -----------

    # Load CSVs
    proPaths = [
        os.path.join("Script", "dataAnalysis", "media", "donny_1_angles.csv"),
        os.path.join("Script", "dataAnalysis", "media", "donny_2_angles.csv"),
        os.path.join("Script", "dataAnalysis", "media", "donny_3_angles.csv")
    ]
    testPath = os.path.join("Script", "dataAnalysis", "media", "test_angles.csv")
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
    test_path, _ = dtw_path(test_array, pro_avg)
    test_mapping = []
    aligned_test = []
    last_j = 0
    for j1 in range(min(len(pro_avg), len(test_array))):
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

    with open("Script\\outputcsv/test_stage_frame_indices.json", "w") as f:
        json.dump(stage_to_test_frames, f, indent=2)

    # ----------- PART 3: Overlay Stage Videos -----------

    os.makedirs("Script\\outputcsv\\test_stages_overlay", exist_ok=True)

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
    test_df = pd.read_csv("Script/dataAnalysis\\media\\test_coords.csv")
    donny_df = pd.read_csv("Script/dataAnalysis\\media\\donny_1_coords.csv")

    # Open the source test video
    video_path = "Script/dataAnalysis\\media\\test.mp4"
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

            # Draw Gordon skeleton (red)
        coords = {}
        if idx < len(test_aligned_df):
            row = test_aligned_df.iloc[idx]
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

        # Draw Donny_1 skeleton (blue), using the corresponding stage frame, aligned to Gordon
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
                    # Gordon's chest center and scale
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

        # For each stage, record which frames belong to it
        for stage, indices in stage_to_test_frames.items():
            if idx in indices:
                frame_indices_by_stage[stage].append(idx)

    out.release()
    cap.release()
    print(f"Full wireframe overlay video saved to {output_path}")
    return output_path



import os
def bigasdmaina2(filePath):
    try: 
        os.remove("Script/dataAnalysis\\media\\test.mp4")
    except:
        pass
    import shutil
    shutil.copyfile(filePath, "Script/dataAnalysis\\media\\test.mp4")
    return bigasdmaina("A")

os.makedirs('uploads', exist_ok=True)
os.makedirs('m', exist_ok=True)
def bigmain(filelinnk):
    import cv2
    import csv
    from ultralytics import YOLO
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import mediapipe as mp

    def detect_ball_yolov8(video_path, output_csv, weights_path='best.pt', conf_thres=0.3):
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

                
                #cv2.imshow('YOLOv8 Ball Detection', frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                
                
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
        df['x'] = df['x'].fillna(method='ffill')
        df['y'] = df['y'].fillna(method='ffill')

        # Extrapolate at the end if there are still NaNs
        for col in ['x', 'y']:
            if df[col].isnull().any():
                # Fit a line to the last 5 valid points and extrapolate
                valid = df[col].last_valid_index()
                if valid is not None:
                    last_vals = df[col].dropna()[-5:]
                    if len(last_vals) >= 2:
                        idx = last_vals.index.values
                        vals = last_vals.values
                        # Linear fit
                        coef = np.polyfit(idx, vals, 1)
                        nan_idx = df[col][valid+1:].index
                        df.loc[nan_idx, col] = coef[0] * nan_idx + coef[1]

        # Reset index and save
        df = df.reset_index().rename(columns={'index': 'frame'})
        df.to_csv(output_csv, index=False)

    def plot_ball_trajectory(ball_csv):
        df = pd.read_csv(ball_csv)
        plt.figure(figsize=(8, 6))
        plt.plot(df['x'], df['y'], marker='o', linestyle='-', color='blue')
        plt.title('Ball Trajectory')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.gca().invert_yaxis()  # Invert y-axis for image coordinates
        plt.grid(True)
        plt.show()

    def overlay_ball_on_video(video_path, ball_csv, output_video):
        df = pd.read_csv(ball_csv).set_index('frame')
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num in df.index:
                x, y = int(df.loc[frame_num, 'x']), int(df.loc[frame_num, 'y'])
                cv2.circle(frame, (x, y), 12, (0, 255, 255), 3)
                cv2.putText(frame, f"Ball", (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()
        print(f"Overlay video saved to {output_video}")

    def extract_wrist_positions(video_path, output_csv):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(video_path)

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y'])

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                h, w, _ = frame.shape
                left_wrist = right_wrist = (None, None)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Left side
                    lw = landmarks[15]
                    le = landmarks[13]
                    # Right side
                    rw = landmarks[16]
                    re = landmarks[14]

                    # Helper to check if a landmark is visible
                    def is_visible(lm):
                        return lm.visibility > 0.5 and (lm.x != 0 or lm.y != 0)

                    # Left wrist
                    if is_visible(lw):
                        left_wrist = (int(lw.x * w), int(lw.y * h))
                    else:
                        # Interpolate between elbow and hand tip (if available)
                        # Pose model doesn't have hand tip, so just use elbow as fallback
                        left_wrist = (int(le.x * w), int(le.y * h))

                    # Right wrist
                    if is_visible(rw):
                        right_wrist = (int(rw.x * w), int(rw.y * h))
                    else:
                        right_wrist = (int(re.x * w), int(re.y * h))

                    # Draw for visualization
                    cv2.circle(frame, left_wrist, 8, (255, 0, 0), -1)
                    cv2.circle(frame, right_wrist, 8, (0, 255, 0), -1)

                writer.writerow([frame_num, left_wrist[0], left_wrist[1], right_wrist[0], right_wrist[1]])

                #cv2.imshow('Wrist Detection', frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

                frame_num += 1

        cap.release()
        cv2.destroyAllWindows()

    def plot_ball_wrist_distance(ball_csv, wrist_csv, which_wrist='right'):
        ball_df = pd.read_csv(ball_csv).set_index('frame')
        wrist_df = pd.read_csv(wrist_csv).set_index('frame')

        # Choose which wrist to use
        if which_wrist == 'right':
            wrist_x = wrist_df['right_wrist_x']
            wrist_y = wrist_df['right_wrist_y']
        else:
            wrist_x = wrist_df['left_wrist_x']
            wrist_y = wrist_df['left_wrist_y']

        # Align frames and compute distance
        common_frames = ball_df.index.intersection(wrist_df.index)
        distances = []
        for frame in common_frames:
            bx, by = ball_df.loc[frame, ['x', 'y']]
            wx, wy = wrist_x.loc[frame], wrist_y.loc[frame]
            if np.isnan([bx, by, wx, wy]).any():
                distances.append(np.nan)
            else:
                dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
                distances.append(dist)

        plt.figure(figsize=(10, 5))
        plt.plot(common_frames, distances, label=f'Ball to {which_wrist.capitalize()} Wrist')
        plt.xlabel('Frame')
        plt.ylabel('Distance (pixels)')
        plt.title(f'Distance Between Ball Center and {which_wrist.capitalize()} Wrist')
        plt.grid(True)
        plt.legend()
        
        plt.show()

    video_files = [
        ("Yourself", filelinnk),
        ("Pro Player", 'betteranalysis\good.mp4')
    ]
    weights_path = r"Script\dataAnalysis\data\best.pt"
    if os.path.exists(weights_path):
        print(f"Using YOLOv8 weights from {weights_path}")

    all_distances = {}
    for label, video_path in video_files:
        print(f"Processing {label}...")
        ball_csv = f"{label}_ball_positions.csv"
        interp_csv = f"{label}_ball_positions_interpolated.csv"
        wrist_csv = f"{label}_wrist_positions.csv"

        # Ball detection
        detect_ball_yolov8(video_path, ball_csv, weights_path=weights_path)
        interpolate_ball_positions(ball_csv, interp_csv)
        # Wrist detection
        extract_wrist_positions(video_path, wrist_csv)

        # Compute distances
        ball_df = pd.read_csv(interp_csv).set_index('frame')
        wrist_df = pd.read_csv(wrist_csv).set_index('frame')
        wrist_x = wrist_df['right_wrist_x']
        wrist_y = wrist_df['right_wrist_y']
        common_frames = ball_df.index.intersection(wrist_df.index)
        distances = []
        for frame in common_frames:
            bx, by = ball_df.loc[frame, ['x', 'y']]
            wx, wy = wrist_x.loc[frame], wrist_y.loc[frame]
            if np.isnan([bx, by, wx, wy]).any():
                distances.append(np.nan)
            else:
                dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
                distances.append(dist)
        all_distances[label] = (common_frames, distances)
    # Plot all on one graph
    plt.figure(figsize=(12, 6))
    for label, (frames, distances) in all_distances.items():
        plt.plot(frames, distances, label=label)
    plt.xlabel('Frame')
    plt.ylabel('Distance (pixels)')
    plt.title('Distance Between Ball Center and Right Wrist (All Videos)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("m", "timing.png"))
    return os.path.join("m", "timing.png")

'''


               __                       ______                                           
              /  |                     /      \                                          
 __   __   __ $$/   ______    ______  /$$$$$$  |______   ______   _____  ____    ______  
/  | /  | /  |/  | /      \  /      \ $$ |_ $$//      \ /      \ /     \/    \  /      \ 
$$ | $$ | $$ |$$ |/$$$$$$  |/$$$$$$  |$$   |  /$$$$$$  |$$$$$$  |$$$$$$ $$$$  |/$$$$$$  |
$$ | $$ | $$ |$$ |$$ |  $$/ $$    $$ |$$$$/   $$ |  $$/ /    $$ |$$ | $$ | $$ |$$    $$ |
$$ \_$$ \_$$ |$$ |$$ |      $$$$$$$$/ $$ |    $$ |     /$$$$$$$ |$$ | $$ | $$ |$$$$$$$$/ 
$$   $$   $$/ $$ |$$ |      $$       |$$ |    $$ |     $$    $$ |$$ | $$ | $$ |$$       |
 $$$$$/$$$$/  $$/ $$/        $$$$$$$/ $$/     $$/       $$$$$$$/ $$/  $$/  $$/  $$$$$$$/ 
                                                                                         
                                                                                         
                                                                                         

'''
def bigaasasdmainasa(videopath):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import mediapipe as mp
    from ultralytics import YOLO
    # --- Helper: In-memory CSV-like storage ---
    class InMemoryCSV:
        def __init__(self):
            self.data = []
            self.columns = None
        def writeheader(self, columns):
            self.columns = columns
        def writerow(self, row):
            self.data.append(row)
        def to_df(self):
            return pd.DataFrame(self.data, columns=self.columns)

    # --- Pose extraction and angle calculation ---
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
        cv2.destroyAllWindows()

        # In-memory DataFrames
        angles_df = pd.DataFrame(angle_data, columns=['timestamp_milis'] + [joint_names[j] for j in relevant_joints])
        coords_df = pd.DataFrame(coords_data)
        return angles_df, coords_df
    
    # --- Generate wireframe for the input video ---
    

    # --- Ball detection and interpolation (in-memory) ---
    def detect_ball_yolov8(video_path, weights_path='best.pt', conf_thres=0.25):
        model = YOLO(weights_path)
        cap = cv2.VideoCapture(video_path)
        rows = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf_thres, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            if len(boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                idx = confs.argmax()
                x1, y1, x2, y2 = boxes[idx][:4]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                rows.append({'frame': frame_num, 'x': cx, 'y': cy})
            frame_num += 1
        cap.release()
        return pd.DataFrame(rows)

    def interpolate_ball_positions_df(df):
        df = df.set_index('frame')
        all_frames = range(df.index.min(), df.index.max() + 1)
        df = df.reindex(all_frames)
        df['x'] = df['x'].interpolate()
        df['y'] = df['y'].interpolate()
        df['x'] = df['x'].fillna(method='bfill').fillna(method='ffill')
        df['y'] = df['y'].fillna(method='bfill').fillna(method='ffill')
        df = df.reset_index()
        return df

    # --- Ball/body analysis (in-memory) ---
    def analyze_ball_and_body_with_time(ball_df, coords_df, width, height):
        ball_df = ball_df.set_index('frame')
        coords_df = coords_df.set_index('frame')
        results = []
        contact_frame = None
        contact_time = None
        contact_threshold = 200  # pixels
        # Try to get timing column name
        time_col = None
        for col in coords_df.columns:
            if "timestamp" in col:
                time_col = col
                break
        common_frames = ball_df.index.intersection(coords_df.index)
        for frame in common_frames:
            bx, by = ball_df.loc[frame, ['x', 'y']]
            s11_x = coords_df.loc[frame, '11_x']
            s11_y = coords_df.loc[frame, '11_y']
            s12_x = coords_df.loc[frame, '12_x']
            s12_y = coords_df.loc[frame, '12_y']
            rw_x = coords_df.loc[frame, '16_x']
            rw_y = coords_df.loc[frame, '16_y']
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
            if not np.isnan([bx, by, chest_x, chest_y]).any():
                dist_ball_chest = np.sqrt((bx - chest_x) ** 2 + (by - chest_y) ** 2)
            else:
                dist_ball_chest = np.nan
            if np.isnan([bx, by, rw_x, rw_y]).any():
                dist_ball_rw = np.nan
            else:
                dist_ball_rw = np.sqrt((bx - rw_x_pix) ** 2 + (by - rw_y_pix) ** 2)
            contact = 0
            frame_time = coords_df.loc[frame, time_col] if time_col and time_col in coords_df.columns else None
            if not np.isnan(dist_ball_rw) and dist_ball_rw < contact_threshold:
                if contact_frame is None:
                    contact_frame = frame
                    contact_time = frame_time
                contact = 1
            results.append({
                'frame': frame,
                'time': frame_time,
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
        return pd.DataFrame(results)

    # --- Process both videos: pro and test ---
    video_files = [
        ("betteranalysis/good.mp4", "good"),
        (videopath, "test"),
    ]
    all_data = {}
    for video_path, label in video_files:
        print(f"Processing {label} video: {video_path}")
        angles_df, coords_df = main(video_path)
        all_data[f"{label}_angles"] = angles_df
        all_data[f"{label}_coords"] = coords_df
        # Ball detection/interpolation
        weights_path = r"betteranalysis\best.pt"
        ball_df = detect_ball_yolov8(video_path, weights_path=weights_path)
        interp_ball_df = interpolate_ball_positions_df(ball_df)
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        ball_body_df = analyze_ball_and_body_with_time(interp_ball_df, coords_df, width, height)
        all_data[f"{label}_ball_body"] = ball_body_df

    # --- Plot ball-hand distance for both videos ---
    def plot_ball_hand_distance(test_df, pro_df, label_test="Test", label_pro="Pro"):
        test_frames = test_df[~test_df['dist_ball_right_wrist'].isna()]['frame']
        test_dist = test_df[~test_df['dist_ball_right_wrist'].isna()]['dist_ball_right_wrist']
        pro_frames = pro_df[~pro_df['dist_ball_right_wrist'].isna()]['frame']
        pro_dist = pro_df[~pro_df['dist_ball_right_wrist'].isna()]['dist_ball_right_wrist']
        plt.figure(figsize=(12, 6))
        plt.plot(test_frames, test_dist, label=label_test, color='red')
        plt.plot(pro_frames, pro_dist, label=label_pro, color='blue')
        plt.xlabel("Frame")
        plt.ylabel("Distance (pixels)")
        plt.title("Distance Between Ball and Right Hand (Wrist) Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    #plot_ball_hand_distance(all_data["test_ball_body"],all_data["good_ball_body"])

    # --- Prepare AI prompt data ---
    def get_ball_contact_summary(df):
        contact_rows = df[df['contact'] == 1]
        if not contact_rows.empty:
            first_contact = contact_rows.iloc[0]
            return f"First contact at frame {int(first_contact['frame'])}, time {first_contact['time']} ms, dist_ball_right_wrist={first_contact['dist_ball_right_wrist']:.1f}"
        else:
            return "No contact detected."

    test_contact_info = get_ball_contact_summary(all_data["test_ball_body"])
    good_contact_info = get_ball_contact_summary(all_data["good_ball_body"])

    # --- Angle comparison and reporting (in-memory) ---
    def compare_angles_and_report(donny_angles_df, test_angles_df, threshold_degrees=30):
        min_len = min(len(donny_angles_df), len(test_angles_df))
        donny_df = donny_angles_df.iloc[:min_len]
        test_df = test_angles_df.iloc[:min_len]
        joint_columns = [col for col in donny_df.columns if col != "timestamp_milis"]
        differences = []
        for i in range(min_len):
            for joint in joint_columns:
                try:
                    d_angle = float(donny_df.iloc[i][joint])
                    t_angle = float(test_df.iloc[i][joint])
                except Exception:
                    continue
                if np.isnan(d_angle) or np.isnan(t_angle):
                    continue
                diff = abs(d_angle - t_angle)
                if diff > threshold_degrees:
                    differences.append({
                        "frame": i,
                        "joint": joint,
                        "donny_angle": d_angle,
                        "test_angle": t_angle,
                        "difference": diff,
                        "donny_time": donny_df.iloc[i].get("timestamp_milis", None),
                        "test_time": test_df.iloc[i].get("timestamp_milis", None)
                    })
        timing_mismatches = []
        if "timestamp_milis" in donny_df.columns and "timestamp_milis" in test_df.columns:
            for i in range(min_len):
                d_time = donny_df.iloc[i]["timestamp_milis"]
                t_time = test_df.iloc[i]["timestamp_milis"]
                if not pd.isna(d_time) and not pd.isna(t_time):
                    if abs(float(d_time) - float(t_time)) > 100:
                        timing_mismatches.append({
                            "frame": i,
                            "donny_time": d_time,
                            "test_time": t_time,
                            "time_difference_ms": abs(float(d_time) - float(t_time))
                        })
        values = []
        for diff in differences:
            values.append(f"Frame {diff['frame']} | Joint: {diff['joint']} | Donny: {diff['donny_angle']:.2f} | Test: {diff['test_angle']:.2f} | Diff: {diff['difference']:.2f} deg | Donny time: {diff['donny_time']} | Test time: {diff['test_time']}")
        for tm in timing_mismatches:
            values.append(f"Frame {tm['frame']} | Donny time: {tm['donny_time']} | Test time: {tm['test_time']} | Diff: {tm['time_difference_ms']} ms")
        return values

    outputtext = compare_angles_and_report(all_data["good_angles"], all_data["test_angles"], threshold_degrees=30)
    outputtext.append("\n--- Ball Contact Info (Test Video) ---\n" + test_contact_info)
    outputtext.append("\n--- Ball Contact Info (Pro Video) ---\n" + good_contact_info)

    # --- AI prompt (unchanged) ---
    from google import genai
    client = genai.Client(api_key="AIzaSyC3M6h-GS102ruxezI6dGcLYYLdVM7aXZ0")
    outputtext = "\n".join(outputtext)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f'''"Analyze the following stream of biomechanical data. Each line represents a single frame, indicating joint angles for a reference individual ('A Professional') and a 'Test' subject, along with the difference and respective timestamps.

The data is from a volleyball serve.
Your task is to summarize this data concisely and informatively. Focus on identifying significant deviations, general trends, and key observations for each joint group (arms, legs, shoulders). Present the summary in a highly readable format, using bullet points for clarity.
You are summarizing this to the test subject, so keep the language clear and direct, avoiding technical jargon where possible. Refer to them as "you" rather than "the tets subject." Remain professional and helpful.
Do NOT NOT NOT use the frame number to point out the time of an action, but rather which part of the serving motion it is part of.
Offer a couple tips or things to improve on.

Data to Analyze:

{outputtext}

Summary Requirements:

Provide an overall introductory observation.
Group findings by joint (e.g., Left Arm, Right Arm, Left Shoulder, Right Shoulder, Left Leg, Right Leg).
For each joint, note:
The general magnitude of differences.
Whether the 'Test' subject's angle is consistently higher or lower than 'the professional's'.
Any notable trends over the frames (e.g., increasing/decreasing difference, specific frames with extreme deviations).
Conclude with a high-level summary of the most pronounced differences.
Keep the language direct and avoid conversational filler."'''
    )
    return response.text
    



#        #############       ####
#        #############       ####
#        #############       ####
#        ####                ####
#        ####                ####                          #### 
#        ####                ####                          #### 
#        ####                ####                          #### 
#        #######             ####                          #### 
#        #######             ####                          #### 
#        #######             ####                          ####
#        ####                ####                         ####
#        ####                ####                         #### 
#        ####                ####                         ####
#        ####                ####                        ####
#        ####                ####################        ####
#        ####                ####################        ####





app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"

@app.route('/main', methods=['POST'])
def casc():

    video_file = request.files["video"]

    # Save the file
    import os
    import shutil

    for root, dirs, files in os.walk('uploads'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
    save_path = os.path.join("uploads", video_file.filename)
    video_file.save(save_path)
    
    image_path= bigmain(save_path)

    geminiResponse = bigaasasdmainasa(save_path)
    videolin = bigasdmaina2(save_path)  
    # video
    # video
    angle_graphs_path = "Script/outputcsv/angle_graphs"
    angle_graphs = []
    for root, dirs, files in os.walk(angle_graphs_path):
        for file in files:
            if file.endswith('.png'):
                angle_graphs.append(os.path.join(root, file))

    # Just append the returned paths directly
    angle_graphs.append(image_path)  
    angle_graphs.append(videolin)
    
    print(angle_graphs)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for filepath in angle_graphs:
            arcname = os.path.basename(filepath)
            zipf.write(filepath, arcname=arcname)

        zipf.writestr("results.txt", geminiResponse)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name="results.zip",
        mimetype="application/zip"
    )



app.run(debug=False)

