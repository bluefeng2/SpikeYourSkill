import cv2
import csv
from ultralytics import YOLO
import os
import pandas as pd

def detect_ball_yolov8(video_path, output_csv, weights_path='best.pt', conf_thres=0.25):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'x', 'y'])

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_thres, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

            # If any detections, take the one with highest confidence
            if len(boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                idx = confs.argmax()
                x1, y1, x2, y2 = boxes[idx][:4]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                writer.writerow([frame_num, cx, cy])
                # Draw bounding box and center
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'({cx}, {cy})', (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Ball Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

def interpolate_ball_positions(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)
    # Set frame as the index for interpolation
    df = df.set_index('frame')
    # Reindex to include all frames between min and max
    all_frames = range(df.index.min(), df.index.max() + 1)
    df = df.reindex(all_frames)
    # Interpolate missing x and y values linearly
    df['x'] = df['x'].interpolate()
    df['y'] = df['y'].interpolate()
    # Optionally, fill any remaining NaNs (at start/end) with nearest values
    df['x'] = df['x'].fillna(method='bfill').fillna(method='ffill')
    df['y'] = df['y'].fillna(method='bfill').fillna(method='ffill')
    # Reset index and save
    df = df.reset_index().rename(columns={'index': 'frame'})
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    video_path = "dataAnalysis\\media\\donny_1.mp4"  # Change to your video file
    output_csv = "ball_positions.csv"
    weights_path = r"C:\Users\Gordon Li\VbForm\VolleyballForm\dataAnalysis\data\best.pt"
    print("Weights exist?", os.path.exists(weights_path), weights_path)
    detect_ball_yolov8(video_path, output_csv, weights_path=weights_path)
    interpolate_ball_positions(output_csv, 'ball_positions_interpolated.csv')
