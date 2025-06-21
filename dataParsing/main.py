import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

# Load CSVs
proPaths = [
    "goodData\donny_1_angles.csv",
    "goodData\donny_2_angles.csv",
    "goodData\donny_3_angles.csv"
]
testPath = "testData\\gordon_angles.csv"
proCSVS = [pd.read_csv(proPath) for proPath in proPaths]
testCSV = pd.read_csv(testPath)

# Interpolate missing values
for df in proCSVS:
    df.interpolate().fillna(method='bfill').fillna(method='ffill', inplace=True)
testCSV.interpolate().fillna(method='bfill').fillna(method='ffill', inplace=True)

# Find the minimum number of frames
min_len = min([len(proCSV) for proCSV in proCSVS] + [len(testCSV)])

# Trim all to the same length
proTrims = [x.iloc[:min_len].reset_index(drop=True) for x in proCSVS]
testCSV_trim = testCSV.iloc[:min_len].reset_index(drop=True)

# Get all joint columns (skip timestamp)
joint_columns = [col for col in proCSVS[0].columns if col != "timestamp_milis"]

# Average the three Donny videos for each joint
pro_avg = sum([proTrim[joint_columns].values for proTrim in proTrims]) / 3
pro_avg_df = pd.DataFrame(pro_avg, columns=joint_columns)

dtw_results = {}

for joint in joint_columns:
    d = dtw.distance(testCSV_trim[joint].values, pro_avg_df[joint].values)
    dtw_results[joint] = d

    plt.figure(figsize=(12, 6))
    plt.plot(testCSV_trim[joint].values, label="User (Test)")
    plt.plot(pro_avg_df[joint].values, label="Professional Average (Reference)", linewidth=3, alpha=0.7)
    plt.title(f"Joint Angle Comparison: {joint}\nDTW: {d:.2f}")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Print summary of DTW distances for all joints
print("DTW distances for each joint (User vs Professional Average):")
for joint, d in dtw_results.items():
    print(f"{joint}: {d:.2f}")