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

            cv2.imshow('Wrist Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    plt.savefig(os.path.join("media/", "timing.png"))
    plt.show()

if __name__ == "__main__":
    video_files = [
        ("donny_1", "dataAnalysis/media/donny_1.mp4"),
        ("donny_2", "dataAnalysis/media/donny_2.mp4"),
        ("donny_3", "dataAnalysis/media/donny_3.mp4"),
    ]
    weights_path = r"dataAnalysis\data\best.pt"

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
    plt.show()
