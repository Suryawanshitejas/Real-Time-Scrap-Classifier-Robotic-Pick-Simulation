import cv2, time, csv, os
from ultralytics import YOLO
from datetime import datetime

weights = r"runs/detect/train2/weights/best.pt"

source = 0  # webcam
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

model = YOLO(weights)
cap = cv2.VideoCapture(source)
csv_file = open(os.path.join(out_dir, "picks.csv"), "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp","frame","class","conf","x","y"])

frame_id = 0
print("Press 'q' to stop.")
while True:
    ret, frame = cap.read()
    if not ret: break
    t0 = time.time()
    results = model(frame, imgsz=640)[0]
    latency = (time.time() - t0) * 1000

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.drawMarker(frame, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 10, 2)
        writer.writerow([datetime.utcnow().isoformat(), frame_id, name, conf, cx, cy])

    cv2.putText(frame, f"Latency: {latency:.1f} ms", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Scrap Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    frame_id += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("✅ Done! Check outputs/picks.csv")
