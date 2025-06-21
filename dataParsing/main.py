import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import matplotlib.cm as cm

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

# Align all pro videos to the first one using DTW
aligned_pros = []
ref = proCSVS[0]
for i, df in enumerate(proCSVS):
    aligned = pd.DataFrame()
    for joint in joint_columns:
        # DTW warping path
        path = dtw.warping_path(ref[joint].values, df[joint].values)
        # Align df[joint] to ref[joint]
        aligned_joint = [df[joint].values[j2] for (j1, j2) in path if j1 < len(ref[joint])]
        # Pad or trim to match reference length
        if len(aligned_joint) < len(ref[joint]):
            aligned_joint += [aligned_joint[-1]] * (len(ref[joint]) - len(aligned_joint))
        aligned[joint] = aligned_joint[:len(ref[joint])]
    aligned_pros.append(aligned)

# Now average the aligned pro videos
pro_avg = sum([df[joint_columns].values for df in aligned_pros]) / len(aligned_pros)
pro_avg_df = pd.DataFrame(pro_avg, columns=joint_columns)

# Optionally, align test to reference as well (not required for DTW distance)
dtw_results = {}
for joint in joint_columns:
    d = dtw.distance(testCSV[joint].values, pro_avg_df[joint].values)
    dtw_results[joint] = d

    plt.figure(figsize=(12, 6))
    plt.plot(testCSV[joint].values, label="User (Test)")
    plt.plot(pro_avg_df[joint].values, label="Pro Average (DTW aligned)", linewidth=3, alpha=0.7)
    plt.title(f"Joint Angle Comparison: {joint}\nDTW: {d:.2f}")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("DTW distances for each joint (User vs Pro Average):")
for joint, d in dtw_results.items():
    print(f"{joint}: {d:.2f}")

# Save the DTW-aligned and averaged pro data to a CSV
pro_avg_df.insert(0, "frame", range(len(pro_avg_df)))  # Optionally add a frame index
pro_avg_df.to_csv("outputcsv\\donny_pro_dtw_aligned_avg.csv", index=False)
print("Saved DTW-aligned and averaged pro data to goodData\\donny_pro_dtw_aligned_avg.csv")

# Optionally align testCSV to the same length as pro_avg_df for direct comparison
test_aligned = testCSV[joint_columns].iloc[:len(pro_avg_df)].reset_index(drop=True)
test_aligned.insert(0, "frame", range(len(test_aligned)))
test_aligned.to_csv("outputcsv\\gordon_aligned_to_pro.csv", index=False)
print("Saved test data aligned to pro length to testData\\gordon_aligned_to_pro.csv")

# Align test (Gordon) to pro_avg_df for each joint using DTW warping path
test_dtw_aligned = pd.DataFrame()
for joint in joint_columns:
    # Get DTW warping path between pro and test
    path = dtw.warping_path(pro_avg_df[joint].values, testCSV[joint].values)
    # For each pro frame, find the corresponding test frame via DTW
    aligned_test_joint = [testCSV[joint].values[j2] for (j1, j2) in path if j1 < len(pro_avg_df[joint])]
    # Pad or trim to match pro length
    if len(aligned_test_joint) < len(pro_avg_df[joint]):
        aligned_test_joint += [aligned_test_joint[-1]] * (len(pro_avg_df[joint]) - len(aligned_test_joint))
    test_dtw_aligned[joint] = aligned_test_joint[:len(pro_avg_df[joint])]

# Assign a unique color to each joint
cmap = cm.get_cmap('tab10', len(joint_columns))
joint_colors = {joint: cmap(i) for i, joint in enumerate(joint_columns)}

plt.figure(figsize=(16, 8))
for joint in joint_columns:
    plt.plot(pro_avg_df[joint], color=joint_colors[joint], alpha=0.7, label=f'Pro: {joint}')
    plt.plot(test_dtw_aligned[joint], color=joint_colors[joint], alpha=0.7, linestyle='dashed', label=f'Test: {joint}')

plt.title("All Joint Angles (DTW aligned): Pro (solid) vs Test (dashed), colored by joint")
plt.xlabel("Frame (DTW aligned)")
plt.ylabel("Angle (degrees)")
# Only show each label once in the legend
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict()
for h, l in zip(labels, handles):
    if l not in unique:
        unique[l] = h
plt.legend(unique.values(), unique.keys(), loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()