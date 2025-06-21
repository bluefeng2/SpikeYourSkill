from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import os
import shutil

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_mp4_duration_opencv(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0:
            duration = frame_count / fps
            cap.release()
            return duration
        else:
            print("Error: FPS is zero or invalid.")
            cap.release()
            return None
    except Exception as e:
        print(f"Error getting duration with OpenCV: {e}")
        return None

@app.post("/runcode")
async def run_code(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get duration
    duration = get_mp4_duration_opencv(temp_path)

    # Clean up the temp file
    os.remove(temp_path)

    if duration is not None:
        return JSONResponse(content={"duration": duration})
    else:
        return JSONResponse(content={"error": "Could not process video."}, status_code=400)