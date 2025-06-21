import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import matplotlib
from betteranalysis.main import plot_and_save_angle_comparisons, analyze_ball_and_body

# betteranalysis/test_main.py

matplotlib.use('Agg')  # For headless testing


def make_dummy_angles_csv(path, n_frames=10):
    data = {
        'timestamp_milis': np.arange(0, n_frames*40, 40),
        'Left Arm': np.linspace(30, 90, n_frames),
        'Right Arm': np.linspace(60, 120, n_frames),
        'Left Shoulder': np.linspace(45, 100, n_frames),
        'Right Shoulder': np.linspace(50, 110, n_frames),
        'Chest L': np.linspace(80, 100, n_frames),
        'Chest R': np.linspace(80, 100, n_frames),
    }
    pd.DataFrame(data).to_csv(path, index=False)

def test_plot_and_save_angle_comparisons_creates_pngs():
    with tempfile.TemporaryDirectory() as tmpdir:
        pro_csv = os.path.join(tmpdir, "pro.csv")
        test_csv = os.path.join(tmpdir, "test.csv")
        outdir = os.path.join(tmpdir, "out")
        make_dummy_angles_csv(pro_csv)
        make_dummy_angles_csv(test_csv)
        plot_and_save_angle_comparisons(pro_csv, test_csv, output_dir=outdir)
        assert os.path.isdir(outdir)
        files = os.listdir(outdir)
        # Should have one PNG per joint (6 in dummy)
        pngs = [f for f in files if f.endswith('.png')]
        assert len(pngs) == 6
        for png in pngs:
            path = os.path.join(outdir, png)
            assert os.path.getsize(path) > 0

def make_dummy_coords_csv(path, n_frames=10):
    # 11_x, 11_y, 12_x, 12_y, 16_x, 16_y (shoulders, right wrist)
    data = {
        'frame': np.arange(n_frames),
        '11_x': np.linspace(0.4, 0.6, n_frames),
        '11_y': np.linspace(0.2, 0.3, n_frames),
        '12_x': np.linspace(0.5, 0.7, n_frames),
        '12_y': np.linspace(0.2, 0.3, n_frames),
        '16_x': np.linspace(0.6, 0.8, n_frames),
        '16_y': np.linspace(0.3, 0.5, n_frames),
    }
    pd.DataFrame(data).to_csv(path, index=False)

def make_dummy_ball_csv(path, n_frames=10):
    data = {
        'frame': np.arange(n_frames),
        'x': np.linspace(100, 200, n_frames),
        'y': np.linspace(150, 250, n_frames),
    }
    pd.DataFrame(data).to_csv(path, index=False)

def test_analyze_ball_and_body_outputs_contact_and_distances():
    with tempfile.TemporaryDirectory() as tmpdir:
        ball_csv = os.path.join(tmpdir, "ball.csv")
        coords_csv = os.path.join(tmpdir, "coords.csv")
        output_csv = os.path.join(tmpdir, "out.csv")
        make_dummy_ball_csv(ball_csv)
        make_dummy_coords_csv(coords_csv)
        # Use arbitrary width/height
        analyze_ball_and_body(ball_csv, coords_csv, output_csv, width=400, height=300)
        assert os.path.exists(output_csv)
        df = pd.read_csv(output_csv)
        # Should have expected columns
        for col in [
            'frame', 'ball_x', 'ball_y', 'chest_x', 'chest_y', 'rw_x', 'rw_y',
            'dist_ball_chest', 'dist_ball_right_wrist', 'contact', 'contact_frame', 'contact_time'
        ]:
            assert col in df.columns
        # At least one contact should be marked (contact==1)
        assert df['contact'].sum() >= 0

# Ball and timing tracking for both videos (integration test)
def test_analyze_ball_and_body_for_both_videos():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate two videos: pro and test
        for label in ['pro', 'test']:
            ball_csv = os.path.join(tmpdir, f"{label}_ball.csv")
            coords_csv = os.path.join(tmpdir, f"{label}_coords.csv")
            output_csv = os.path.join(tmpdir, f"{label}_analysis.csv")
            make_dummy_ball_csv(ball_csv)
            make_dummy_coords_csv(coords_csv)
            analyze_ball_and_body(ball_csv, coords_csv, output_csv, width=400, height=300)
            assert os.path.exists(output_csv)
            df = pd.read_csv(output_csv)
            assert 'contact' in df.columns
            # Should have a contact_frame column
            assert 'contact_frame' in df.columns
