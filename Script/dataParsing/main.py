import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw_path
import matplotlib.cm as cm
import json
import cv2
import os

# Load CSVs
proPaths = [
    "testData\\donny_1_angles.csv",
    "testData\\donny_2_angles.csv",
    "testData\\donny_3_angles.csv"
]
testPath = "testData\\gordon_angles.csv"
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
    # For each ref frame, find the corresponding frame in arr
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

# Save mappings
with open("outputcsv/pro_multivariate_dtw_frame_mappings.json", "w") as f:
    json.dump(pro_dtw_mappings, f, indent=2)
with open("outputcsv/test_multivariate_dtw_frame_mapping.json", "w") as f:
    json.dump(test_mapping, f, indent=2)

# Save aligned data
pro_avg_df.insert(0, "frame", range(len(pro_avg_df)))
pro_avg_df.to_csv("outputcsv/donny_pro_multivariate_dtw_aligned_avg.csv", index=False)
test_aligned_df.insert(0, "frame", range(len(test_aligned_df)))
test_aligned_df.to_csv("outputcsv/gordon_multivariate_dtw_aligned.csv", index=False)

# Plot all joints, same color for pro and test, dashed for test
cmap = cm.get_cmap('tab10', len(joint_columns))
joint_colors = {joint: cmap(i) for i, joint in enumerate(joint_columns)}

plt.figure(figsize=(16, 8))
for joint in joint_columns:
    plt.plot(pro_avg_df[joint], color=joint_colors[joint], alpha=0.7, label=f'Pro: {joint}')
    plt.plot(test_aligned_df[joint], color=joint_colors[joint], alpha=0.7, linestyle='dashed', label=f'Test: {joint}')
plt.title("All Joint Angles (Multivariate DTW aligned): Pro (solid) vs Test (dashed), colored by joint")
plt.xlabel("Frame (DTW aligned)")
plt.ylabel("Angle (degrees)")
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict()
for l, h in zip(labels, handles):
    if l not in unique:
        unique[l] = h
plt.legend(unique.values(), unique.keys(), loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()



# --- Stage definitions from aligned (DTW) frames (1-based in your notes, so subtract 1 for Python) ---
stages = {
    "preparation": list(range(0, 13)),      # frames 1-13 -> 0-12
    "throw": list(range(13, 26)),           # frames 14-26 -> 13-25
    "precontact": list(range(26, 39)),      # frames 27-39 -> 26-38
    "swing": list(range(39, 44)),           # frames 40-44 -> 39-43
    "contact": list(range(44, 46)),         # frames 45-46 -> 44-45
    "followthrough": list(range(46, 57)),   # frames 47-57 -> 46-56
}

# Map aligned frame indices to original Gordon frame indices for each stage
stage_to_gordon_frames = {}
for stage, aligned_indices in stages.items():
    gordon_indices = [test_mapping[i] for i in aligned_indices if i < len(test_mapping)]
    stage_to_gordon_frames[stage] = gordon_indices

# Save the mapping for later use (optional)
with open("outputcsv/gordon_stage_frame_indices.json", "w") as f:
    json.dump(stage_to_gordon_frames, f, indent=2)

print("Stage to original Gordon frame mapping:")
for stage, frames in stage_to_gordon_frames.items():
    print(f"{stage}: {frames}")

# --- Ensure all frames are assigned to a stage, in order, with no gaps or overlaps ---

# Load the original video to get total frame count
video_path = 'dataAnalysis\\media\\gordon.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Build a list of (frame_idx, stage) pairs from your mapping, preserving order and no duplicates
frame_stage_pairs = []
for stage, indices in stage_to_gordon_frames.items():
    for idx in indices:
        frame_stage_pairs.append((idx, stage))

# Sort by frame index
frame_stage_pairs.sort(key=lambda x: x[0])

# Remove duplicates (keep first occurrence)
seen = set()
ordered_pairs = []
for idx, stage in frame_stage_pairs:
    if idx not in seen:
        ordered_pairs.append((idx, stage))
        seen.add(idx)

# Fill in any missing frames (assign to previous stage or "other")
full_pairs = []
last_stage = None
j = 0
for i in range(total_frames):
    if j < len(ordered_pairs) and ordered_pairs[j][0] == i:
        last_stage = ordered_pairs[j][1]
        full_pairs.append((i, last_stage))
        j += 1
    else:
        # Assign missing frames to previous stage or "other"
        full_pairs.append((i, last_stage if last_stage else "other"))

# Save the new mapping for each stage
stage_to_full_gordon_frames = {}
for idx, stage in full_pairs:
    stage_to_full_gordon_frames.setdefault(stage, []).append(idx)

with open("outputcsv/gordon_stage_frame_indices_full.json", "w") as f:
    json.dump(stage_to_full_gordon_frames, f, indent=2)

print("Full stage-to-frame mapping (no gaps) saved to outputcsv/gordon_stage_frame_indices_full.json")

# --- Optional: Extract each stage as a video clip ---

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs("outputcsv/gordon_stages_full", exist_ok=True)

writers = {}
for stage in stage_to_full_gordon_frames:
    out_path = f"outputcsv/gordon_stages_full/gordon_{stage}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers[stage] = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    stage = full_pairs[i][1]
    writers[stage].write(frame)

for writer in writers.values():
    writer.release()
cap.release()

print("Saved stage videos. All clips together cover the full video with no gaps or overlaps.")