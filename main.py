
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile,os,io,cv2
import numpy as np
import easyocr
from PIL import Image
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
from uuid import uuid4
import math
from routes import car 
from database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()
model = YOLO("models/truckAndCar.pt")
reader = easyocr.Reader(['vi', 'en'])

#car router
app.include_router(car.router)

#AI
@app.post("/detect_license_plate")
async def detect_license_plate(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    results = reader.readtext(image_np)

    plates = []
    for (bbox, text, confidence) in results:
        if 6 <= len(text) <= 10: 
            plates.append({"plate": text, "confidence": round(confidence, 2)})

    return {"plates": plates}


@app.post("/predict_car")
async def predict_car(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    results = model.predict(source=tmp_path)

    image = cv2.imread(tmp_path)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{int(cls)}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    os.remove(tmp_path)

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


def is_close(cx, cy, counted_centers, threshold=20):
    for (x, y) in counted_centers:
        dist = math.hypot(cx - x, cy - y)
        if dist < threshold:
            return True
    return False

# @app.post("/predict_from_video")
# async def predict_from_video(file: UploadFile = File(...)):
#     suffix = os.path.splitext(file.filename)[1]
#     if suffix not in ['.mp4', '.avi', '.mov']:
#         return {"error": "Unsupported video format"}

#     with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
#         tmp.write(await file.read())
#         tmp_input_path = tmp.name

#     tmp_output_path = f"{uuid4()}.mp4"

#     cap = cv2.VideoCapture(tmp_input_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     line_y = int(height * 0.85)
#     counted_centers = []
#     vehicle_count = 0

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.predict(source=frame, verbose=False)

#         any_vehicle_crossed = False

#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()
#             scores = r.boxes.conf.cpu().numpy()
#             classes = r.boxes.cls.cpu().numpy()

#             for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
#                 x1, y1, x2, y2 = map(int, box)
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)

#                 class_names = ['Car', 'Truck']
#                 label = f"{class_names[int(cls)]}: {score:.2f}"

#                 counted = False
#                 if (cy > line_y - 10) and (cy < line_y + 10):
#                     if not is_close(cx, cy, counted_centers):
#                         counted_centers.append((cx, cy))
#                         vehicle_count += 1
#                         counted = True
#                         any_vehicle_crossed = True
#                     else:
#                         counted = True
#                 else:
#                     counted = False

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 if counted:
#                     label += " -counted"
#                     cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         line_color = (0, 0, 255) if any_vehicle_crossed else (0, 255, 0)
#         cv2.line(frame, (0, line_y), (width, line_y), line_color, 2)

#         cv2.putText(frame, f"Count: {vehicle_count}", (20, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         out.write(frame)

#     cap.release()
#     out.release()
#     os.remove(tmp_input_path)
#     return FileResponse(tmp_output_path, media_type="video/mp4", filename="output.mp4")

@app.post("/predict_from_video")
async def predict_from_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    if suffix not in ['.mp4', '.avi', '.mov']:
        return {"error": "Unsupported video format"}

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_input_path = tmp.name

    os.makedirs("videos", exist_ok=True)
    # tmp_output_path = f"{uuid4()}.mp4"
    tmp_output_path = os.path.join("videos", f"{uuid4()}.mp4")

    cap = cv2.VideoCapture(tmp_input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    line_y = int(height * 0.65)
    tracked_ids = set()
    vehicle_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, verbose=False)

        any_vehicle_crossed = False

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            try:
                track_ids = r.boxes.id.cpu().numpy()
            except:
                track_ids = [None] * len(boxes)

            for i, (box, score, cls, track_id) in enumerate(zip(boxes, scores, classes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                class_names = ['Car', 'Truck']
                label = f"{class_names[int(cls)]}: {score:.2f}"

                counted = False

                if (cy > line_y - 30) and (cy < line_y + 30):
                    if track_id is not None and track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        vehicle_count += 1
                        counted = True
                        any_vehicle_crossed = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if counted:
                    label += " -counted"
                    cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        line_color = (0, 0, 255) if any_vehicle_crossed else (0, 255, 0)
        cv2.line(frame, (0, line_y), (width, line_y), line_color, 2)

        cv2.putText(frame, f"Count: {vehicle_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(tmp_input_path)

    return FileResponse(tmp_output_path, media_type="video/mp4", filename="output.mp4")

# video feed
def gen_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{int(cls)}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


    