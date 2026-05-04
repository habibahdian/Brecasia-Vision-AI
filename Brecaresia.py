import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import threading
import torch
import ctypes
import datetime 
import uvicorn
import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image, ImageTk
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- FastAPI Setup for PWA Hosting ---
app = FastAPI()
app_gui = None

@app.get("/manifest.json")
async def get_manifest():
    return FileResponse("manifest.json")

@app.get("/sw.js")
async def get_sw():
    return FileResponse("sw.js")

@app.get("/logo.png")
async def get_logo():
    return FileResponse("logo.png")

@app.get("/", response_class=HTMLResponse)
async def get():
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error</h1><p>index.html not found.</p>"

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    wrist_path, state = [], {"start_y": None}
    try:
        while True:
            data = await websocket.receive_text()
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None and app_gui is not None:
                ann, m, corr = app_gui.analyze_and_annotate(frame, "Chest expansion", wrist_path, state)
                _, buffer = cv2.imencode('.jpg', ann)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    "annotated_image": f"data:image/jpeg;base64,{jpg_as_text}",
                    "area": f"{m['val']:.2f}"
                })
    except Exception as e:
        print(f"WebSocket error: {e}")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

try:
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
except:
    pass

class ExerciseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Exercise Analysis System - Live Tracking")
        self.root.geometry("1450x850")
        
        threading.Thread(target=run_server, daemon=True).start()
        
        self.bg_color = "#f8fafc"      
        self.pink_accent = "#e91e63"   
        self.light_pink = "#1e293b"    
        self.text_white = "#e91e63"

        self.kmeans_model = None
        self.cluster_to_group_map = {} 
        self.exercise_columns = ["Chest expansion exercise", "Hand gripping exercise", "Shoulder circumduction exercise", "Upper limb circumduction exercise", "Wall walking exercise"]

        self.style = ttk.Style()
        self.style.theme_use('clam') 
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_white, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5, background=self.pink_accent)
        
        self.root.configure(bg=self.bg_color)
        self.root.bind('<space>', self.toggle_pause)

        # FIX 1: Gunakan model NANO dan Path Lokal
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo11n-pose.pt") 
        self.model.to(self.device)
        
        self.display_width, self.display_height = 600, 400
        self.video_paths = {}
        self.results_data = [] 
        self.is_webcam_running = False
        self.is_recording = False
        self.video_writer = None
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.inference_stop_event = threading.Event()
        self.active_session_id = 0 
        self.reminder_time = None 
        
        header_frame = tk.Frame(root, bg=self.bg_color, pady=10)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header_frame, text="Rehabilitation Dashboard", font=("Segoe UI", 24, "bold"), bg=self.bg_color, fg=self.pink_accent).pack(side=tk.LEFT, padx=20)
        
        # FIX 2: Path Logo Lokal
        try:
            logo_img = Image.open("logo.png").resize((80, 80)) 
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(header_frame, image=self.logo_photo, bg=self.bg_color).pack(side=tk.RIGHT, padx=30)
        except Exception as e:
            print(f"Logo 'logo.png' tidak ditemukan di folder.")

        control_panel = ttk.Frame(root, padding=10)
        control_panel.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = ttk.Frame(control_panel)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(control_panel)
        row2_frame.pack(fill=tk.X, pady=2)
        
        file_frame = ttk.LabelFrame(row1_frame, text=" File Inference ", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.exercises = ["Chest expansion", "Hand gripping", "Shoulder circumduction", "Upper limb circumduction", "Wall walking"]
        for ex in self.exercises:
            ttk.Button(file_frame, text=ex, command=lambda e=ex: self.load_video(e)).pack(side=tk.LEFT, padx=2)

        webcam_frame = ttk.LabelFrame(row1_frame, text=" Live Camera ", padding=5)
        webcam_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.mode_select = ttk.Combobox(webcam_frame, values=self.exercises, state="readonly", width=20)
        self.mode_select.pack(side=tk.LEFT, padx=5)
        ttk.Button(webcam_frame, text="Start", command=self.start_webcam).pack(side=tk.LEFT, padx=2)
        self.record_btn = ttk.Button(webcam_frame, text="Record", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(webcam_frame, text="Stop", command=self.stop_webcam).pack(side=tk.LEFT, padx=2)

        action_frame = ttk.LabelFrame(row2_frame, text=" Data Actions ", padding=5)
        action_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(action_frame, text="Run Files", command=self.start_inference).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="|| Pause", command=self.pause_inference).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="▶ Play", command=self.resume_inference).pack(side=tk.LEFT, padx=2)

        record_frame = ttk.LabelFrame(row2_frame, text=" Reporting ", padding=5)
        record_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(record_frame, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)

        ml_frame = ttk.LabelFrame(row2_frame, text=" ML Model ", padding=5)
        ml_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(ml_frame, text="Train", command=self.train_ml_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(ml_frame, text="Predict", command=self.predict_stage).pack(side=tk.LEFT, padx=5)

        display_frame = tk.Frame(root, bg="#0d1a0d", bd=2, relief="groove") 
        display_frame.pack(pady=10, padx=20, expand=True, fill=tk.BOTH)
        self.label_proc = tk.Label(display_frame, text="System Ready", bg="#0d1a0d", fg=self.pink_accent)
        self.label_proc.pack(expand=True)
        
        res_frame = tk.Frame(root, bg="#112211", pady=10) 
        res_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.feedback_label = tk.Label(res_frame, text="Awaiting Movement...", font=("Segoe UI", 18, "bold"), bg="#112211", fg=self.pink_accent)
        self.feedback_label.pack()
        self.results_display = tk.Label(res_frame, text="Metrics: -", font=("Segoe UI", 14), bg="#112211", fg="#ffffff")
        self.results_display.pack()
        
        self.status_label = tk.Label(root, text=f"Device: {self.device.upper()}", bg=self.bg_color, fg=self.light_pink, font=("Segoe UI", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Bagian Logika (Sama Seperti Sebelumnya Namun Disederhanakan) ---
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    def analyze_and_annotate(self, frame, name, wrist_path, state):
        results = self.model(frame, verbose=False, device=self.device)
        annotated = results[0].plot()
        metrics = {"val": 0, "type": ""}
        is_correct = False
        if results[0].keypoints.xy.shape[1] > 9:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            s, e, w = kpts[5], kpts[7], kpts[9]
            if name == "Chest expansion":
                metrics = {"val": self.calculate_angle(s, e, w), "type": "Angle"}
                is_correct = metrics["val"] > 160
            elif name == "Hand gripping":
                metrics = {"val": self.calculate_angle(e, w, [w[0], w[1]+100]), "type": "Wrist Angle"}
                is_correct = metrics["val"] > 30
            elif "circumduction" in name:
                wrist_path.append(w)
                if len(wrist_path) > 5:
                    hull = cv2.convexHull(np.array(wrist_path).astype(np.int32))
                    metrics = {"val": cv2.contourArea(hull), "type": "Area"}
                    is_correct = metrics["val"] > 5000
            elif name == "Wall walking":
                if state['start_y'] is None: state['start_y'] = w[1]
                metrics = {"val": abs(state['start_y'] - w[1]), "type": "Distance"}
                is_correct = metrics["val"] > 50
        return annotated, metrics, is_correct

    def update_ui_text(self, metrics, is_correct, session_id):
        if session_id == self.active_session_id:
            self.results_display.config(text=f"{metrics['type']}: {metrics['val']:.2f}")
            self.feedback_label.config(text="CORRECT ✅" if is_correct else "ADJUST ⚠️", fg="#2e7d32" if is_correct else "#d32f2f")

    def update_gui_image(self, label, frame, session_id):
        if session_id == self.active_session_id:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            label.config(image=img)
            label.image = img

    def start_webcam(self):
        mode = self.mode_select.get()
        if not mode: return
        self.is_webcam_running = True
        threading.Thread(target=self.run_webcam_loop, args=(mode,), daemon=True).start()

    def run_webcam_loop(self, name):
        cap = cv2.VideoCapture(0)
        wrist_path, state = [], {"start_y": None}
        while self.is_webcam_running and cap.isOpened():
            self.pause_event.wait()
            ret, frame = cap.read()
            if not ret: break
            ann, m, corr = self.analyze_and_annotate(frame, name, wrist_path, state)
            self.update_gui_image(self.label_proc, cv2.resize(ann, (self.display_width, self.display_height)), 0)
            self.root.after(1, lambda: self.update_ui_text(m, corr, 0))
        cap.release()

    def stop_webcam(self): self.is_webcam_running = False

    def load_video(self, exercise):
        path = filedialog.askopenfilename()
        if path: self.video_paths[exercise] = path

    def pause_inference(self): self.pause_event.clear()
    def resume_inference(self): self.pause_event.set()
    def toggle_pause(self, e=None): self.resume_inference() if not self.pause_event.is_set() else self.pause_inference()

    def start_inference(self):
        self.active_session_id += 1 
        threading.Thread(target=self.run_inference_process, args=(self.active_session_id,), daemon=True).start()

    def run_inference_process(self, session_id):
        for name, path in list(self.video_paths.items()):
            cap = cv2.VideoCapture(path)
            wrist_path, state, max_val = [], {"start_y": None}, 0
            while cap.isOpened():
                self.pause_event.wait()
                ret, frame = cap.read()
                if not ret: break
                ann, m, corr = self.analyze_and_annotate(frame, name, wrist_path, state)
                max_val = max(max_val, m["val"])
                self.update_gui_image(self.label_proc, cv2.resize(ann, (600, 400)), session_id)
                self.root.after(1, lambda m=m, c=corr, s=session_id: self.update_ui_text(m, c, s))
            cap.release()
            self.results_data.append({"Exercise": name, "Measurement": round(max_val, 2)})

    def train_ml_model(self):
        try:
            df = pd.read_csv("training.csv")
            self.kmeans_model = KMeans(n_clusters=4, n_init=10).fit(df[self.exercise_columns])
            messagebox.showinfo("Success", "Model Trained")
        except: messagebox.showerror("Error", "Check training.csv")

    def predict_stage(self): messagebox.showinfo("Info", "Prediction logic ready")

    def export_excel(self):
        if self.results_data:
            pd.DataFrame(self.results_data).to_excel(f"Report_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx")
            messagebox.showinfo("Success", "Saved")

    def toggle_recording(self): pass # Sederhanakan untuk kestabilan

if __name__ == "__main__":
    root = tk.Tk()
    app_gui = ExerciseGUI(root)
    root.mainloop()