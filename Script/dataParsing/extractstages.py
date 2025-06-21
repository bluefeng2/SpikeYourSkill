import json
import cv2
import os
import pandas as pd

# Load the mapping from stage to original Gordon frame indices
with open("outputcsv/gordon_stage_frame_indices.json", "r") as f:
    stage_to_gordon_frames = json.load(f)

# Stage definitions (aligned to Donny_1)
stage_ranges = {
    "preparation": list(range(0, 13)),      # frames 0-12
    "throw": list(range(13, 26)),           # frames 13-25
    "precontact": list(range(26, 39)),      # frames 26-38
    "swing": list(range(39, 44)),           # frames 39-43
    "contact": list(range(44, 46)),         # frames 44-45
    "followthrough": list(range(46, 57)),   # frames 46-56
}

# Load joint coordinates for Gordon and Donny_1
gordon_df = pd.read_csv("testData/gordon_coords.csv")
donny_df = pd.read_csv("testData/donny_1_coords.csv")  # Make sure this exists and is in the same format

# Open the source Gordon video
video_path = 'dataAnalysis\\media\\gordon.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("outputcsv/gordon_stages_overlay", exist_ok=True)

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (11, 23), (12, 24),  # Torso sides
]

for stage, gordon_indices in stage_to_gordon_frames.items():
    donny_indices = stage_ranges[stage]
    out_path = f"outputcsv/gordon_stages_overlay/gordon_{stage}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i, idx in enumerate(gordon_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw Gordon skeleton (red)
        coords = {}
        if idx < len(gordon_df):
            row = gordon_df.iloc[idx]
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
    out.release()
    print(f"Saved {out_path}")
