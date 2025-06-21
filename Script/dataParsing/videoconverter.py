import json
import cv2

# Load frame indices from JSON (just a list)
with open('outputcsv\\test_multivariate_dtw_frame_mapping.json', 'r') as f:
    frame_indices = json.load(f)

# Open the source video
cap = cv2.VideoCapture('dataAnalysis\\media\\gordon.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('gordon_selected_frames.mp4', fourcc, fps, (width, height))

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        out.write(frame)

cap.release()
out.release()