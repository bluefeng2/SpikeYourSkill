import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw # pip install fastdtw
from scipy.spatial.distance import euclidean

# --- 1. Define Landmark Mapping for Easier Access ---
# This dictionary maps the descriptive names to their numerical IDs for column access.
# Assumes columns are named f'{ID}_x', f'{ID}_y', f'{ID}_z'
LANDMARK_ID_MAP = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
    'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
    'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
    'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
    # Add hand landmarks if they follow ID 32 and you want to use them for angles
}

# --- 2. Function to Calculate a 3D Angle ---
def calculate_3d_angle(df_row, p1_name, p2_name, p3_name, landmark_map):
    """
    Calculates the 3D angle (in degrees) between three points for a single row of a DataFrame.
    p2 is the vertex of the angle.
    Input names should be keys from landmark_map (e.g., 'right_shoulder').
    """
    try:
        p1_coords = np.array([df_row[f'{landmark_map[p1_name]}_x'],
                              df_row[f'{landmark_map[p1_name]}_y'],
                              df_row[f'{landmark_map[p1_name]}_z']])
        p2_coords = np.array([df_row[f'{landmark_map[p2_name]}_x'],
                              df_row[f'{landmark_map[p2_name]}_y'],
                              df_row[f'{landmark_map[p2_name]}_z']])
        p3_coords = np.array([df_row[f'{landmark_map[p3_name]}_x'],
                              df_row[f'{landmark_map[p3_name]}_y'],
                              df_row[f'{landmark_map[p3_name]}_z']])
    except KeyError as e:
        # print(f"Warning: Missing landmark coordinate {e} in row. Returning NaN.")
        return np.nan

    v1 = p1_coords - p2_coords
    v2 = p3_coords - p2_coords

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return np.nan # Avoid division by zero if points are identical/collapsed

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Clamp to avoid floating point errors

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

# --- 3. Function to Process a Single CSV File (Load & Calculate Angles) ---
def process_serve_data(file_path, landmark_map=LANDMARK_ID_MAP):
    """
    Loads a CSV, calculates specified joint angles, and returns the processed DataFrame.
    """
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Rename timestamp column for consistency (if it's not already 'timestamp_milis')
    if 'timestamp in milis' in df.columns: # Adjust if your actual column name is different
        df = df.rename(columns={'timestamp in milis': 'timestamp_milis'})
    
    # Sort by timestamp to ensure correct temporal order
    df = df.sort_values(by='timestamp_milis').reset_index(drop=True)

    # Define the joint angles to calculate
    # Format: (new_column_name, (P1_name, P2_name, P3_name))
    joint_definitions = {
        # Right Arm
        'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'right_shoulder_angle': ('right_hip', 'right_shoulder', 'right_elbow'), # Simplified for general flexion/extension
        'right_wrist_angle': ('right_elbow', 'right_wrist', 'right_pinky'), # Using pinky as a distal reference

        # Left Arm
        'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'left_shoulder_angle': ('left_hip', 'left_shoulder', 'left_elbow'),
        'left_wrist_angle': ('left_elbow', 'left_wrist', 'left_pinky'),

        # Right Leg
        'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
        'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'), # Simplified for general flexion/extension
        'right_ankle_angle': ('right_knee', 'right_ankle', 'right_foot_index'),

        # Left Leg
        'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
        'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
        'left_ankle_angle': ('left_knee', 'left_ankle', 'left_foot_index'),
        
        # Torso/Spine angles can be more complex and usually require a defined global plane
        # A simplified torso angle (e.g., between hips and shoulders) could be added:
        # 'torso_angle': ('left_hip', 'right_hip', 'right_shoulder') # Example, needs careful definition
    }

    # Calculate all specified angles
    for angle_name, (p1, p2, p3) in joint_definitions.items():
        df[angle_name] = df.apply(
            lambda row: calculate_3d_angle(row, p1, p2, p3, landmark_map),
            axis=1
        )

    # Drop rows with NaN angles (e.g., if a landmark was not detected in some frames)
    # This is important before DTW.
    initial_rows = len(df)
    df = df.dropna(subset=[col for col in df.columns if '_angle' in col])
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows due to missing angle calculations.")

    # Convert timestamps to seconds (relative to start of serve)
    if not df.empty:
        df['time_seconds'] = (df['timestamp_milis'] - df['timestamp_milis'].iloc[0]) / 1000
    else:
        df['time_seconds'] = pd.Series([]) # Empty Series if df is empty

    print(f"Processed {len(df)} frames. Calculated angles: {', '.join([col for col in df.columns if '_angle' in col])}")
    return df

# --- 4. DTW Functions for Alignment ---

def get_average_aligned_professional_angles(professional_dfs, angle_name):
    """
    Aligns multiple professional serves for a given angle using DTW to the first serve as reference,
    then calculates the average aligned angle curve and its standard deviation.
    """
    if not professional_dfs:
        print(f"No professional data provided for angle: {angle_name}")
        return pd.Series(), pd.Series(), pd.Series() # Return empty series

    # Ensure angle_name column exists in all DFs before proceeding
    for df in professional_dfs:
        if angle_name not in df.columns or df[angle_name].isnull().all():
            print(f"Warning: '{angle_name}' missing or all NaN in one of the professional DataFrames. Skipping average.")
            return pd.Series(), pd.Series(), pd.Series()

    # Use the first professional's angle as the reference template
    reference_angles = professional_dfs[0][angle_name].values
    
    # Store aligned angles, initialized with the reference itself
    aligned_pro_angles_list = [reference_angles] 

    for i in range(1, len(professional_dfs)):
        current_pro_angles = professional_dfs[i][angle_name].values
        
        # Handle cases where current_pro_angles might be empty or all NaNs
        if current_pro_angles.size == 0 or np.all(np.isnan(current_pro_angles)):
            print(f"Skipping professional DF {i} for {angle_name} due to empty/NaN data.")
            continue

        # Calculate DTW path
        distance, path = fastdtw(current_pro_angles, reference_angles, dist=euclidean)
        
        # Use the path to create an aligned version of the current pro's angles
        aligned_current_pro_angles = np.full(len(reference_angles), np.nan) # Initialize with NaN
        counts = np.zeros(len(reference_angles), dtype=int)

        for current_idx, ref_idx in path:
            if ref_idx < len(aligned_current_pro_angles): # Ensure index is within bounds
                aligned_current_pro_angles[ref_idx] = current_pro_angles[current_idx]
                counts[ref_idx] += 1
        
        aligned_pro_angles_list.append(aligned_current_pro_angles)

    # Convert list of aligned arrays to a DataFrame for easier averaging
    # Pad shorter arrays with NaN to ensure all have the same length as the longest reference
    max_len = max(len(arr) for arr in aligned_pro_angles_list)
    padded_aligned_list = [np.pad(arr.astype(float), (0, max_len - len(arr)), 'constant', constant_values=np.nan) for arr in aligned_pro_angles_list]

    aligned_pro_df_temp = pd.DataFrame(padded_aligned_list).T 
    
    # Calculate the average and standard deviation
    avg_angles = aligned_pro_df_temp.mean(axis=1)
    std_angles = aligned_pro_df_temp.std(axis=1)
    
    # Interpolate any remaining NaNs (e.g., from padding or unmapped points)
    avg_angles = avg_angles.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    std_angles = std_angles.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')

    # Create a consistent time array for the averaged data
    avg_time_seconds = professional_dfs[0]['time_seconds'].values[:len(avg_angles)]

    return avg_angles, std_angles, avg_time_seconds

def align_your_serve_to_pro_average(your_df, pro_avg_angles, your_angle_name):
    """
    Aligns your single serve's angle data to the professional average angle data using DTW.
    Returns your angle data resampled to match the length of pro_avg_angles.
    """
    if your_angle_name not in your_df.columns or your_df[your_angle_name].isnull().all():
        print(f"Warning: '{your_angle_name}' missing or all NaN in your DataFrame. Cannot align.")
        return np.full(len(pro_avg_angles), np.nan)

    your_angles = your_df[your_angle_name].values
    
    # Handle cases where pro_avg_angles might be empty or all NaNs
    if pro_avg_angles.empty or np.all(np.isnan(pro_avg_angles.values)):
        print(f"Warning: Professional average for {your_angle_name} is empty or all NaN. Cannot align.")
        return np.full(len(your_angles), np.nan) # Return unaligned series if pro_avg is invalid

    distance, path = fastdtw(your_angles, pro_avg_angles.values, dist=euclidean)
    
    # Create an aligned version of your angles, matching the length of pro_avg_angles
    aligned_your_angles = np.full(len(pro_avg_angles), np.nan) # Initialize with NaN
    counts = np.zeros(len(pro_avg_angles), dtype=int)

    for your_idx, pro_idx in path:
        if pro_idx < len(aligned_your_angles): # Ensure index is within bounds
            aligned_your_angles[pro_idx] = your_angles[your_idx]
            counts[pro_idx] += 1
    
    # Fill any gaps created by DTW resampling with interpolation
    aligned_your_angles_series = pd.Series(aligned_your_angles)
    aligned_your_angles_series = aligned_your_angles_series.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')

    print(f"DTW Distance for {your_angle_name} (Your Serve vs Pro Avg): {distance:.2f}")
    return aligned_your_angles_series.values

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Configuration ---
    YOUR_SERVE_FILE = 'testData/serve.csv' # Path to your serve CSV
    # List of professional serve CSVs (for a single professional's multiple serves)
    PROFESSIONAL_SERVE_FILES = ['goodData/serve.csv', 'goodData/serve_2.csv', 'goodData/serve_3.csv'] 

    # --- 1. Load and Process Your Serve Data ---
    your_serve_df = process_serve_data(YOUR_SERVE_FILE)

    # --- 2. Load and Process Professional's Multiple Serve Data ---
    professional_dfs = [process_serve_data(f) for f in PROFESSIONAL_SERVE_FILES]

    # --- 3. Define Angles to Compare ---
    # These should match the column names generated in process_serve_data
    angles_to_compare = [
        'right_elbow_angle', 'right_shoulder_angle', 'right_wrist_angle',
        'left_elbow_angle', 'left_shoulder_angle', 'left_wrist_angle',
        'right_knee_angle', 'right_hip_angle', 'right_ankle_angle',
        'left_knee_angle', 'left_hip_angle', 'left_ankle_angle',
    ]

    # --- 4. Perform DTW Alignment and Comparison for Each Angle ---
    aligned_results = {}
    pro_averages = {}

    for angle_name in angles_to_compare:
        print(f"\n--- Analyzing {angle_name} ---")
        
        # Get average and std dev for professional(s) (aligned internally)
        pro_avg, pro_std, pro_time = get_average_aligned_professional_angles(professional_dfs, angle_name)
        
        if pro_avg.empty: # Skip if no valid pro average was generated
            print(f"Skipping {angle_name} due to insufficient professional data.")
            continue
        
        pro_averages[angle_name] = {'avg': pro_avg, 'std': pro_std, 'time': pro_time}

        # Align your serve to the professional average
        aligned_your_angle = align_your_serve_to_pro_average(your_serve_df, pro_avg, angle_name)
        
        aligned_results[angle_name] = aligned_your_angle

        # --- 5. Plotting and Quantifying Differences for current angle ---
        if not your_serve_df.empty and len(aligned_your_angle) > 0 and not pro_avg.empty:
            plt.figure(figsize=(12, 6))
            
            plt.plot(pro_time, pro_avg, label='Professional Average', color='red', linewidth=2)
            plt.fill_between(pro_time, pro_avg - pro_std, pro_avg + pro_std,
                             color='red', alpha=0.2, label='Pro Std Dev Range')
            plt.plot(pro_time, aligned_your_angle, label='Your Serve (Aligned)', color='blue', linestyle='--')
            
            plt.xlabel('Aligned Time (s)')
            plt.ylabel('Angle (degrees)')
            plt.title(f'Comparison of {angle_name.replace("_", " ").title()} During Volleyball Serve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{angle_name}_comparison.png')
            plt.close() # Close plot to save memory
            print(f"Saved plot: {angle_name}_comparison.png")

            # Quantify Differences
            if np.all(np.isnan(aligned_your_angle)) or np.all(np.isnan(pro_avg.values)):
                print(f"Cannot quantify differences for {angle_name} due to NaN values after alignment.")
            else:
                # Only compare non-NaN parts
                valid_indices = ~np.isnan(aligned_your_angle) & ~np.isnan(pro_avg.values)
                if np.any(valid_indices):
                    mean_abs_diff = np.nanmean(np.abs(aligned_your_angle[valid_indices] - pro_avg.values[valid_indices]))
                    rmse = np.sqrt(np.nanmean((aligned_your_angle[valid_indices] - pro_avg.values[valid_indices])**2))
                    print(f"Mean Absolute Difference ({angle_name}): {mean_abs_diff:.2f} degrees")
                    print(f"RMSE ({angle_name}): {rmse:.2f} degrees")
                else:
                    print(f"No common valid data points to quantify differences for {angle_name}.")
        else:
            print(f"Skipping plotting for {angle_name} due to empty or invalid data.")

    print("\n--- Analysis Complete ---")
    print("Check the generated PNG files for visual comparisons.")