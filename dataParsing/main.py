import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

joint_name_map = {
    "0": "Nose",
    "1": "Left Eye Inner",
    "2": "Left Eye",
    "3": "Left Eye Outer",
    "4": "Right Eye Inner",
    "5": "Right Eye",
    "6": "Right Eye Outer",
    "7": "Left Ear",
    "8": "Right Ear",
    "9": "Mouth Left",
    "10": "Mouth Right",
    "11": "Left Shoulder",
    "12": "Right Shoulder",
    "13": "Left Elbow",
    "14": "Right Elbow",
    "15": "Left Wrist",
    "16": "Right Wrist",
    "17": "Left Pinky",
    "18": "Right Pinky",
    "19": "Left Index",
    "20": "Right Index",
    "21": "Left Thumb",
    "22": "Right Thumb",
    "23": "Left Hip",
    "24": "Right Hip",
    "25": "Left Knee",
    "26": "Right Knee",
    "27": "Left Ankle",
    "28": "Right Ankle",
    "29": "Left Heel",
    "30": "Right Heel",
    "31": "Left Foot Index",
    "32": "Right Foot Index",
    "33": "Spine Base",
    "34": "Spine Mid",
    "35": "Spine Top",
    "36": "Neck",
    "37": "Head Top",
    "38": "Left Big Toe",
    "39": "Right Big Toe",
    "40": "Left Small Toe",
    "41": "Right Small Toe",
    "42": "Left Heel",
    "43": "Right Heel",
    "44": "Pelvis",
    "45": "Thorax",
    "46": "Neck Base",
    "47": "Head",
    "48": "Left Clavicle",
    "49": "Right Clavicle",
    "50": "Left Scapula",
    "51": "Right Scapula",
    "52": "Left Upper Arm",
    "53": "Right Upper Arm",
    "54": "Left Forearm",
    "55": "Right Forearm",
    "56": "Left Hand",
    "57": "Right Hand",
    "58": "Left Thigh",
    "59": "Right Thigh",
    "60": "Left Calf",
    "61": "Right Calf",
    "62": "Left Foot",
    "63": "Right Foot",
    "64": "Left Toe",
    "65": "Right Toe",
    "66": "Left Eyebrow",
    "67": "Right Eyebrow",
    "68": "Chin",
    "69": "Forehead",
    "70": "Left Cheek",
    "71": "Right Cheek",
    "72": "Nose Tip",
    "73": "Upper Lip",
    "74": "Lower Lip"
}

def load_and_average_csvs(csv_paths):
    dfs = [pd.read_csv(p) for p in csv_paths]
    min_len = min(len(df) for df in dfs)
    dfs = [df.iloc[:min_len] for df in dfs]
    avg_df = sum(dfs) / len(dfs)
    return avg_df

def load_and_align_csv(csv_path, columns, length):
    df = pd.read_csv(csv_path)
    if len(df) > length:
        df = df.iloc[:length]
    elif len(df) < length:
        df = df.reindex(range(length)).interpolate()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df[columns]

def compare_with_dtw(pro_ref, pro_columns, user_df):
    dtw_distances = {}
    for col in pro_columns:
        ref_seq = pro_ref[col].values
        user_seq = user_df[col].values
        # Skip columns with NaN or empty sequences
        if np.any(pd.isna(ref_seq)) or np.any(pd.isna(user_seq)):
            print(f"Skipping {col} due to NaN values.")
            continue
        if len(ref_seq) == 0 or len(user_seq) == 0:
            print(f"Skipping {col} due to empty sequence.")
            continue
        distance, _ = fastdtw(ref_seq, user_seq, dist=lambda x, y: abs(x - y))
        dtw_distances[col] = distance
    return dtw_distances

def get_joint_label(col):
    # col is like "3_x" or "7_z"
    parts = col.split("_")
    idx = parts[0]
    axis = parts[1]
    joint = joint_name_map.get(idx, f"Joint {idx}")
    axis_label = {"x": "X", "y": "Y", "z": "Z"}.get(axis, axis)
    return f"{joint} ({axis_label})"

def plot_differences(pro_ref, user_df, columns):
    for col in columns:
        plt.figure(figsize=(10, 4))
        plt.plot(pro_ref[col].values, label='Professional (avg)')
        plt.plot(user_df[col].values, label='You')
        plt.title(f'Joint: {get_joint_label(col)}')
        plt.xlabel('Frame')
        plt.ylabel('Angle/Position')
        plt.legend()
        plt.tight_layout()
        plt.show()

# --- Usage ---

pro_csvs = [
    "goodData\\1_analysis.csv",
    "goodData\\2_analysis.csv",
    "goodData\\3_analysis.csv"
]
user_csv = "testData\\badvideo_analysis.csv"

pro_avg = load_and_average_csvs(pro_csvs)

# List of joint indices to exclude (face features)
face_joint_indices = {
    "0",  # Nose
    "1", "2", "3", "4", "5", "6",  # Eyes
    "7", "8",  # Ears
    "9", "10",  # Mouth corners
    "66", "67",  # Eyebrows
    "68", "69", "70", "71", "72", "73", "74"  # Chin, forehead, cheeks, nose tip, lips
}

# Filter out face features from columns
columns = [
    col for col in pro_avg.columns
    if col != "timestamp_milis" and col.split("_")[0] not in face_joint_indices
]

user_df = load_and_align_csv(user_csv, columns, len(pro_avg))

dtw_results = compare_with_dtw(pro_avg, columns, user_df)

sorted_joints = sorted(dtw_results.items(), key=lambda x: -x[1])
print("Joints with largest DTW differences:")
for joint, dist in sorted_joints[:10]:
    print(f"{joint}: {dist:.2f}")

# Plot all differences
plot_differences(pro_avg, user_df, columns)