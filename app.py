import cv2
import base64
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import FileResponse
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# --- ENDPOINT FOR BRANDING ASSETS ---
@app.get("/logo.png")
async def get_logo():
    """Returns the Brecaresia brand logo file."""
    return FileResponse("logo.png")

# --- AI MODEL INITIALIZATION ---
# Automatically select CUDA (NVIDIA GPU) if available, otherwise fallback to CPU
device = 0 if torch.cuda.is_available() else "cpu"
# Using YOLOv11 Pose model to extract human joints (keypoints)
model = YOLO("yolo11n-pose.pt") 

# --- WEB PAGE ROUTING ---
@app.get("/")
async def get_index(): 
    """Serves the main dashboard user interface."""
    return FileResponse("index.html")

@app.get("/manifest.json")
async def get_manifest(): 
    """Serves the PWA manifest for iPad installation."""
    return FileResponse("manifest.json")

# --- MATHEMATICAL HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three joints using trigonometry (arctan2).
    a, b, c are [x, y] coordinates. 'b' is the vertex (e.g., the elbow).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def process_frame(frame, exercise_name, wrist_path, state):
    """
    Performs AI Pose Detection and calculates specific clinical metrics.
    Requires state variables (wrist_path, state) to track movement across frames.
    Returns: (annotated_frame, metrics_dictionary)
    """
    try:
        # Run inference using YOLOv11-Pose
        results = model(frame, verbose=False, device=device, half=(device==0))
        annotated_frame = results[0].plot()
        
        # Default fallback metric
        metrics = {"val": 0.0, "type": "Awaiting Detection..."}
        
        # Ensure keypoints are detected (Need at least up to index 9 for the wrist)
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            if results[0].keypoints.xy.shape[1] > 9:
                # Extract keypoint coordinates from GPU to CPU
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                
                # Indexes: 5=Shoulder, 7=Elbow, 9=Wrist (Left side of body)
                s = kpts[5]
                e = kpts[7]
                w = kpts[9]
                
                # Ignore calculation if confidence is too low (coordinates are exactly 0)
                if w[0] != 0 and w[1] != 0:
                    
                    # 1. CHEST EXPANSION: Angle between Shoulder, Elbow, and Wrist
                    if exercise_name == "Chest expansion":
                        metrics = {"val": calculate_angle(s, e, w), "type": "Angle (Degrees)"}
                    
                    # 2. HAND GRIPPING: Angle of wrist relative to a straight downward line
                    elif exercise_name == "Hand gripping":
                        # Create an imaginary point straight down from the wrist
                        down_point = [w[0], w[1] + 100]
                        metrics = {"val": calculate_angle(e, w, down_point), "type": "Wrist Angle (Degrees)"}
                    
                    # 3. CIRCUMDUCTION: Area of the convex hull formed by the wrist's path
                    elif exercise_name in ["Shoulder circumduction", "Upper limb circumduction"]:
                        wrist_path.append([int(w[0]), int(w[1])]) # Save current wrist position
                        
                        # Need at least 5 points to draw a meaningful shape
                        if len(wrist_path) > 5:
                            hull = cv2.convexHull(np.array(wrist_path).astype(np.int32))
                            metrics = {"val": cv2.contourArea(hull), "type": "Area (Pixels^2)"}
                            
                            # Draw the tracked circular path on the frame
                            pts = np.array(wrist_path)
                            center = np.mean(pts, axis=0).astype(int)
                            radius = int(np.max(np.linalg.norm(pts - center, axis=1)))
                            cv2.circle(annotated_frame, tuple(center), radius, (0, 255, 0), 2)
                    
                    # 4. WALL WALKING: Vertical distance traveled by the wrist
                    elif exercise_name == "Wall walking":
                        if state['start_y'] is None:
                            state['start_y'] = w[1] # Set initial resting position
                        
                        # Calculate distance moved from the start point
                        metrics = {"val": abs(state['start_y'] - w[1]), "type": "Y-Distance (Pixels)"}

        return annotated_frame, metrics
        
    except Exception as e:
        print(f"[ERROR] AI processing failed: {e}")
        return frame, {"val": 0.0, "type": "Error"}

# --- STATIC FILE INFERENCE ---
@app.post("/infer-file")
async def infer_file(
    file: UploadFile = File(...), 
    exercise: str = Form("Chest expansion") # Default to Chest Expansion if not provided
):
    """
    Endpoint for static file inference.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Provide empty state arrays for a single static image
            img, metrics = process_frame(frame, exercise, wrist_path=[], state={"start_y": None})
            _, buffer = cv2.imencode('.jpg', img)
            
            return {
                "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}", 
                "area": f"{metrics['val']:.2f}",
                "metric_type": metrics['type']
            }
        else:
            return {"error": "Invalid file format", "area": "0.00"}
            
    except Exception as e:
        print(f"[ERROR] File inference failed: {e}")
        return {"error": str(e), "area": "0.00"}

# --- LIVE VIDEO STREAMING ---
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint. Maintains memory (state) of the patient's movement.
    """
    await websocket.accept()
    print("[INFO] iPad Live Connection: Established")
    
    # Initialize memory states for the current live session
    session_wrist_path = []
    session_state = {"start_y": None}
    
    # NOTE: You can change this hardcoded value based on a dropdown in your iPad HTML
    current_exercise = "Chest expansion" 
    
    try:
        frame_count = 0
        while True:
            # Receive base64 frame from iPad
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            
            # AI Inference on the frame using the tracked memory
            img, metrics = process_frame(frame, current_exercise, session_wrist_path, session_state)
            
            # Compress processed frame
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 45])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Send JSON response back to iPad
            response_data = {
                "annotated_image": f"data:image/jpeg;base64,{jpg_as_text}",
                "area": f"{metrics['val']:.2f}",
                "metric_type": metrics['type'],
                "device": "CUDA (RTX Enabled)" if device == 0 else "CPU (Fallback)"
            }
            
            await websocket.send_json(response_data)
            
            frame_count += 1
                
    except Exception as e:
        print(f"[ERROR] WebSocket connection lost: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    # --- STARTUP LOGS ---
    print("=" * 60)
    print("BRECARESIA MEDICAL AI - BIOMECHANICAL SERVER")
    print(f"HARDWARE ACCELERATION: {'ACTIVE (CUDA)' if device == 0 else 'INACTIVE (CPU)'}")
    print(f"CORE MODEL: YOLOv11n-pose (Joint Tracking Mode)")
    print("STATUS: ONLINE - Listening on http://0.0.0.0:8080")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)