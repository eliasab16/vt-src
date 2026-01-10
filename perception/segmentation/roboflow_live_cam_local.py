# To run this script with mps acceleration:
#   cd inference
#   uvicorn cpu_http:app --port 9001 --host 0.0.0.0
#
# Make sure that the server is running on localhost:9001

import cv2
import os
import time
import numpy as np
from collections import deque
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

# The model runs locally on a docker container
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=ROBOFLOW_API_KEY
)

config = InferenceConfiguration(
    confidence_threshold=0.8,
    iou_threshold=0.5
)
client.configure(config)
client.select_model(MODEL_ID)

cap = cv2.VideoCapture(0)
frame_times = deque(maxlen=30)

print("Starting segmentation, press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = client.infer(frame)
    
    # fps calculations
    curr_time = time.time()
    frame_times.append(curr_time)
    if len(frame_times) > 1:
        fps = len(frame_times) / (frame_times[-1] - frame_times[0])
    else:
        fps = 0
    
    # Add segmentation masks
    display_frame = frame.copy()
    if result and 'predictions' in result:
        for pred in result['predictions']:
            if 'points' in pred:
                points = np.array([[p['x'], p['y']] for p in pred['points']], dtype=np.int32)
                
                # Polygon mask: green highlight/overlay
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [points], (0, 255, 0))  # Green mask
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                
                # Polygon outline: border around the mask
                cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)
            
    # Add fps display to the frame
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Local Segmentation", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
