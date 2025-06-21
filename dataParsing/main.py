import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

# Load CSVs
donny1 = pd.read_csv("dataAnalysis\\media\\donny_1_angles.csv")
donny2 = pd.read_csv("dataAnalysis\\media\\donny_2_angles.csv")
donny3 = pd.read_csv("dataAnalysis\\media\\donny_3_angles.csv")
gordon = pd.read_csv("dataAnalysis\\media\\gordon_angles.csv")

# Interpolate missing values
donny1 = donny1.interpolate().fillna(method='bfill').fillna(method='ffill')
donny2 = donny2.interpolate().fillna(method='bfill').fillna(method='ffill')
donny3 = donny3.interpolate().fillna(method='bfill').fillna(method='ffill')
gordon = gordon.interpolate().fillna(method='bfill').fillna(method='ffill')

# Find the minimum number of frames
min_len = min(len(donny1), len(donny2), len(donny3))

# Trim all to the same length
donny1_trim = donny1.iloc[:min_len].reset_index(drop=True)
donny2_trim = donny2.iloc[:min_len].reset_index(drop=True)
donny3_trim = donny3.iloc[:min_len].reset_index(drop=True)
gordon_trim = gordon.iloc[:min_len].reset_index(drop=True)

# Get all joint columns (skip timestamp)
joint_columns = [col for col in donny1.columns if col != "timestamp_milis"]

# Average the three Donny videos for each joint
donny_avg = (donny1_trim[joint_columns].values + donny2_trim[joint_columns].values + donny3_trim[joint_columns].values) / 3
donny_avg_df = pd.DataFrame(donny_avg, columns=joint_columns)

dtw_results = {}

for joint in joint_columns:
    d = dtw.distance(gordon_trim[joint].values, donny_avg_df[joint].values)
    dtw_results[joint] = d

    plt.figure(figsize=(12, 6))
    plt.plot(gordon_trim[joint].values, label="Gordon (Test)")
    plt.plot(donny_avg_df[joint].values, label="Donny Average (Reference)", linewidth=3, alpha=0.7)
    plt.title(f"Joint Angle Comparison: {joint}\nDTW: {d:.2f}")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Print summary of DTW distances for all joints
print("DTW distances for each joint (Gordon vs Donny Average):")
for joint, d in dtw_results.items():
    print(f"{joint}: {d:.2f}")