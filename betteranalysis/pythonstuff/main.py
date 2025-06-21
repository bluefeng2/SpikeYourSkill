import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw_path
import matplotlib.cm as cm
from ultralytics import YOLO
import mediapipe as mp

def bigMain(videopath):
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
    print(response.text)
    
    # ----------- Overlay for the entire video, Donny_1 (blue) and test (red) wireframes, both synced to video frames -----------

    # This assumes you have already extracted the coordinates for both videos into all_data["test_coords"] and all_data["good_coords"]
    # and that both have the same number of frames or you want to align by frame index.

    test_coords_df = all_data["test_coords"]
    pro_coords_df = all_data["good_coords"]

    video_path = videopath
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    POSE_CONNECTIONS = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (11, 23), (12, 24),  # Torso sides
    ]

    out_path = "betteranalysis/outputcsv/test_complete_overlay_fullvideo_synced.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Draw the red outline (test subject, matches the video frame) ---
        coords = {}
        if idx < len(test_coords_df):
            row = test_coords_df.iloc[idx]
            for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                if pd.notnull(row[x_col]) and pd.notnull(row[y_col]):
                    coords[joint] = (int(float(row[x_col]) * width), int(float(row[y_col]) * height))
            for j1, j2 in POSE_CONNECTIONS:
                if j1 in coords and j2 in coords:
                    cv2.line(frame, coords[j1], coords[j2], (0, 0, 255), 2)
            for (x, y) in coords.values():
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # --- Draw the blue outline (pro, also matches the video frame index, so no lag) ---
        coords_pro = {}
        if idx < len(pro_coords_df):
            row_pro = pro_coords_df.iloc[idx]
            for joint in [11,12,13,14,15,16,23,24,25,26,27,28]:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                if pd.notnull(row_pro[x_col]) and pd.notnull(row_pro[y_col]):
                    coords_pro[joint] = (int(float(row_pro[x_col]) * width), int(float(row_pro[y_col]) * height))
            for j1, j2 in POSE_CONNECTIONS:
                if j1 in coords_pro and j2 in coords_pro:
                    cv2.line(frame, coords_pro[j1], coords_pro[j2], (255, 0, 0), 2)
            for (x, y) in coords_pro.values():
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        out.write(frame)

    out.release()
    cap.release()
    print(f"Full overlay video with both wireframes synced to video saved to {out_path}")

bigMain("betteranalysis/test.mp4")