import pandas as pd
import matplotlib.pyplot as plt
from main import get_joint_angles, joint_names

# Load the CSV
df = pd.read_csv("testData\\gordon_analysis.csv")

# Calculate angles (interpolates and fills NaNs internally)
angles_df = get_joint_angles(df)

# Only keep arms, shoulders, chest joints
relevant = [
    ("11", "13", "15"),  # Left Arm
    ("12", "14", "16"),  # Right Arm
    ("13", "11", "23"),  # Left Shoulder
    ("14", "12", "24"),  # Right Shoulder
    ("11", "23", "12"),  # Chest (LShoulder-LHip-RShoulder)
    ("12", "24", "11"),  # Chest (RShoulder-RHip-LShoulder)
]

for idxs in relevant:
    col = f"{idxs[0]}-{idxs[1]}-{idxs[2]}"
    label = joint_names.get(idxs, col)
    plt.figure(figsize=(10, 4))
    plt.plot(angles_df[col])
    plt.title(f"Angle over Time: {label}")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.tight_layout()
    plt.show()