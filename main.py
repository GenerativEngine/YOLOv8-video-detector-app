from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
from ultralytics import YOLO
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO('yolov8n.pt')

ALLOWED_CLASSES = {
    "person",
    "cat",
    "dog",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files allowed.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit.")

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_input.name, "wb") as f:
        f.write(contents)

    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        os.unlink(temp_input.name)
        raise HTTPException(status_code=400, detail="Error opening video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))

    total_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()

        for cls_id in class_ids:
            class_name = model.names[cls_id]
            if class_name in ALLOWED_CLASSES:
                total_counts[class_name] = total_counts.get(class_name, 0) + 1

        for box, cls_id, score in zip(boxes, class_ids, scores):
            class_name = model.names[cls_id]
            if class_name in ALLOWED_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    os.unlink(temp_input.name)

    return JSONResponse({
        "detected_objects": total_counts,
        "annotated_video_url": f"/download-video/{os.path.basename(temp_output_path)}"
    })

@app.get("/download-video/{video_name}")
def download_video(video_name: str):
    video_path = os.path.join(tempfile.gettempdir(), video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(video_path, media_type='video/mp4', filename=video_name)
