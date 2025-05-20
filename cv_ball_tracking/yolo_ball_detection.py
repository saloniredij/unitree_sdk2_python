import cv2
import torch
from nanotrack import NanoTrack

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Confidence threshold
model.iou = 0.45  # IoU threshold

# Initialize NanoTrack
tracker = NanoTrack()

# Start video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes with scores and classes

    # Prepare detections for NanoTrack
    formatted_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        formatted_detections.append([x1, y1, x2, y2, conf, cls])

    # Update tracker with current detections
    tracks = tracker.update(formatted_detections)

    # Draw tracking results
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv5 + NanoTrack', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()