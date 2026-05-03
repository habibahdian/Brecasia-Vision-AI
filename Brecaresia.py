import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import threading
import torch
import ctypes
import datetime 
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- Terminal Minimization ---
try:
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
except:
    pass

class ExerciseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Exercise Analysis System - Live Tracking")
        self.root.geometry("1350x850")
        
        # --- Modern Style Setup ---
        self.style = ttk.Style()
        self.style.theme_use('clam') 
        self.style.configure("TFrame", background="#2d2d2d")
        self.style.configure("TLabel", background="#2d2d2d", foreground="white", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.root.configure(bg="#2d2d2d")

        # --- Keyboard Binding ---
        self.root.bind('<space>', self.toggle_pause)
        self.root.focus_force() 

        # --- GPU Configuration ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = YOLO("C:/Users/Soumyajit/Pictures/Applying Universities/NCKU/Courses/Sem 2/BIODESIGN(2) - IMPLEMENTATION OF INNOVATIVE MEDICAL DEVICES/GUI/yolo11m-pose.pt")
        self.model.to(self.device)
        
        self.display_width, self.display_height = 600, 400
        self.video_paths = {}
        self.results_data = [] 
        self.is_webcam_running = False
        
        # --- Recording State ---
        self.is_recording = False
        self.video_writer = None
        
        # --- Control Events ---
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.inference_stop_event = threading.Event()
        self.active_session_id = 0 
        
        # --- Scheduler State ---
        self.reminder_time = None
        
        # --- UI Layout: Header with Logo ---
        header_frame = tk.Frame(root, bg="#2d2d2d", pady=10)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(header_frame, text="Medical Exercise Analysis Dashboard", font=("Segoe UI", 20, "bold"), 
                 bg="#2d2d2d", fg="#4da6ff").pack(side=tk.LEFT, padx=20)
        
        try:
            logo_path = "C:/Users/Soumyajit/Pictures/Applying Universities/NCKU/Courses/Sem 2/BIODESIGN(2) - IMPLEMENTATION OF INNOVATIVE MEDICAL DEVICES/GUI/Brecaresia 1.png"
            logo_img = Image.open(logo_path).resize((80, 80)) 
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(header_frame, image=self.logo_photo, bg="#2d2d2d")
            logo_label.pack(side=tk.RIGHT, padx=30)
        except Exception as e:
            print(f"Logo not found: {e}")

        # --- Control Panel Layout (Three Rows) ---
        control_panel = ttk.Frame(root, padding=10)
        control_panel.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = ttk.Frame(control_panel)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(control_panel)
        row2_frame.pack(fill=tk.X, pady=2)
        row3_frame = ttk.Frame(control_panel) # New row for Scheduling
        row3_frame.pack(fill=tk.X, pady=2)
        
        # Group 1: Files
        file_frame = ttk.LabelFrame(row1_frame, text=" File Inference ", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.exercises = ["Chest expansion", "Hand gripping", "Shoulder circumduction", 
                          "Upper limb circumduction", "Wall walking"]
        for ex in self.exercises:
            ttk.Button(file_frame, text=f"Load {ex}", command=lambda e=ex: self.load_video(e)).pack(side=tk.LEFT, padx=2)

        # Group 2: Webcam
        webcam_frame = ttk.LabelFrame(row1_frame, text=" Live Camera ", padding=5)
        webcam_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.mode_select = ttk.Combobox(webcam_frame, values=self.exercises, state="readonly", width=20)
        self.mode_select.pack(side=tk.LEFT, padx=5)
        ttk.Button(webcam_frame, text="Start", command=self.start_webcam).pack(side=tk.LEFT, padx=2)
        self.record_btn = ttk.Button(webcam_frame, text="Record", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(webcam_frame, text="Stop", command=self.stop_webcam).pack(side=tk.LEFT, padx=2)

        # Group 3: Actions
        action_frame = ttk.LabelFrame(row2_frame, text=" Data Actions ", padding=5)
        action_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(action_frame, text="Run Files", command=self.start_inference).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="|| Pause", command=self.pause_inference).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="▶ Play", command=self.resume_inference).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)

        # Group 4: Scheduling
        sched_frame = ttk.LabelFrame(row3_frame, text=" Exercise Reminder ", padding=5)
        sched_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        tk.Label(sched_frame, text="Set Time (HH:MM):", bg="#2d2d2d", fg="white").pack(side=tk.LEFT, padx=5)
        self.time_entry = ttk.Entry(sched_frame, width=10)
        self.time_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(sched_frame, text="Set Reminder", command=self.set_reminder).pack(side=tk.LEFT, padx=5)
        
        # Start the background checker
        self.check_reminders()

        # --- Main Display ---
        display_frame = tk.Frame(root, bg="#1e1e1e", bd=2, relief="groove")
        display_frame.pack(pady=10, padx=20, expand=True, fill=tk.BOTH)
        self.label_proc = tk.Label(display_frame, text="System Ready", bg="#1e1e1e", fg="#888")
        self.label_proc.pack(expand=True)
        
        # Results Dashboard
        res_frame = tk.Frame(root, bg="#333", pady=10)
        res_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.results_display = tk.Label(res_frame, text="Real-time Metrics: Awaiting Data...", 
                                        font=("Segoe UI", 16), bg="#333", fg="#00ff00")
        self.results_display.pack()
        
        self.status_label = tk.Label(root, text=f"Device: {self.device.upper()}", bg="#222", fg="#aaa", font=("Segoe UI", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Scheduling Methods ---
    def set_reminder(self):
        self.reminder_time = self.time_entry.get()
        messagebox.showinfo("Reminder Set", f"Reminder set for {self.reminder_time}")

    def check_reminders(self):
        if self.reminder_time:
            current_time = datetime.datetime.now().strftime("%H:%M")
            if current_time == self.reminder_time:
                messagebox.showinfo("Time to Exercise!", "It's time for your medical exercises!")
                self.reminder_time = None # Reset after triggering
        
        # Check every 60 seconds
        self.root.after(60000, self.check_reminders)

    # --- Toggle Function for Spacebar ---
    def toggle_pause(self, event=None):
        if self.pause_event.is_set():
            self.pause_inference()
        else:
            self.resume_inference()

    # --- Recording Logic ---
    def toggle_recording(self):
        if not self.is_webcam_running:
            messagebox.showwarning("Warning", "Start the webcam first!")
            return

        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            self.is_recording = True
            self.record_btn.config(text="Stop Rec")
            self.status_label.config(text=f"Status: Recording to {filename}")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Record")
            self.status_label.config(text="Status: Recording Stopped")

    # --- Pause/Play Controls ---
    def pause_inference(self):
        self.pause_event.clear()
        self.status_label.config(text="Status: Paused")

    def resume_inference(self):
        self.pause_event.set()
        self.status_label.config(text=f"Status: Playing | Device: {self.device.upper()}")

    # --- Math & Processing ---
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    def analyze_and_annotate(self, frame, name, wrist_path, state):
        results = self.model(frame, verbose=False, device=self.device, half=(self.device=='cuda'))
        annotated = results[0].plot()
        metrics = {"val": 0, "type": ""}
        
        if results[0].keypoints.xy.shape[1] > 9:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            s, e, w = kpts[5], kpts[7], kpts[9]
            
            if name == "Chest expansion":
                metrics = {"val": self.calculate_angle(s, e, w), "type": "Angle (Degrees)"}
            elif name == "Hand gripping":
                metrics = {"val": self.calculate_angle(e, w, [w[0], w[1]+100]), "type": "Wrist Angle (Degrees)"}
            elif name in ["Shoulder circumduction", "Upper limb circumduction"]:
                wrist_path.append(w)
                if len(wrist_path) > 5:
                    hull = cv2.convexHull(np.array(wrist_path).astype(np.int32))
                    metrics = {"val": cv2.contourArea(hull), "type": "Area (Pixels^2)"}
                    pts = np.array(wrist_path)
                    center = np.mean(pts, axis=0).astype(int)
                    radius = int(np.max(np.linalg.norm(pts - center, axis=1)))
                    cv2.circle(annotated, tuple(center), radius, (0, 255, 0), 2)
            elif name == "Wall walking":
                if state['start_y'] is None:
                    state['start_y'] = w[1]
                metrics = {"val": abs(state['start_y'] - w[1]), "type": "Y-Distance (Pixels)"}
        return annotated, metrics

    def update_ui_text(self, metrics, session_id):
        if session_id == self.active_session_id:
            self.results_display.config(text=f"{metrics['type']}: {metrics['val']:.2f}")

    def update_gui_image(self, label, frame, session_id):
        if session_id == self.active_session_id:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            label.config(image=img, text="")
            label.image = img

    def start_webcam(self):
        mode = self.mode_select.get()
        if not mode: return
        self.is_webcam_running = True
        threading.Thread(target=self.run_webcam_loop, args=(mode,), daemon=True).start()

    def stop_webcam(self): 
        self.is_webcam_running = False
        if self.is_recording:
            self.toggle_recording()

    def run_webcam_loop(self, name):
        cap = cv2.VideoCapture(0)
        wrist_path, state = [], {"start_y": None}
        while self.is_webcam_running and cap.isOpened():
            self.pause_event.wait()
            ret, frame = cap.read()
            if not ret: break
            
            ann, m = self.analyze_and_annotate(frame, name, wrist_path, state)
            if self.is_recording and self.video_writer:
                self.video_writer.write(ann)
            
            self.update_gui_image(self.label_proc, cv2.resize(ann, (self.display_width, self.display_height)), session_id=0)
            self.root.after(1, lambda: self.update_ui_text(m, session_id=0))
        cap.release()

    def load_video(self, exercise):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.MOV")])
        if path: self.video_paths[exercise] = path

    def start_inference(self):
        self.inference_stop_event.set()
        self.active_session_id += 1 
        threading.Thread(target=self.run_inference_process, args=(self.active_session_id,), daemon=True).start()

    def run_inference_process(self, session_id):
        self.inference_stop_event.clear()
        
        video_items_snapshot = list(self.video_paths.items())
        
        for name, path in video_items_snapshot:
            if self.inference_stop_event.is_set():
                return
            
            cap = cv2.VideoCapture(path)
            wrist_path, state, max_val = [], {"start_y": None}, 0
            while cap.isOpened():
                if self.inference_stop_event.is_set():
                    cap.release()
                    return
                
                self.pause_event.wait()
                
                ret, frame = cap.read()
                if not ret: break
                ann, m = self.analyze_and_annotate(frame, name, wrist_path, state)
                max_val = max(max_val, m["val"])
                
                self.update_gui_image(self.label_proc, cv2.resize(ann, (600, 400)), session_id)
                self.root.after(1, lambda m=m, s=session_id: self.update_ui_text(m, s))
            
            cap.release()
            if session_id == self.active_session_id:
                self.results_data.append({"Exercise": name, "Measurement": round(max_val, 2)})

    def export_excel(self):
        if self.results_data:
            pd.DataFrame(self.results_data).to_excel("exercise_results.xlsx", index=False)
            messagebox.showinfo("Success", "Saved to exercise_results.xlsx")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseGUI(root)
    root.mainloop()