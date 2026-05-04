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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Terminal Minimization ---
try:
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
except:
    pass

class ExerciseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Exercise Analysis System - Live Tracking")
        self.root.geometry("1450x850")
        
        # --- Breast Cancer Awareness Theme Colors ---
        self.bg_color = "#f8fafc"      
        self.pink_accent = "#e91e63"   
        self.light_pink = "#1e293b"    
        self.text_white = "#e91e63"

        # --- ML State ---
        self.kmeans_model = None
        self.cluster_to_group_map = {} 
        self.exercise_columns = [
            "Chest expansion exercise", 
            "Hand gripping exercise", 
            "Shoulder circumduction exercise", 
            "Upper limb circumduction exercise", 
            "Wall walking exercise"
        ]

        # --- Modern Style Setup ---
        self.style = ttk.Style()
        self.style.theme_use('clam') 
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_white, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5, background=self.pink_accent)
        self.style.map("TButton", background=[('active', self.light_pink)])
        
        self.style.configure("TLabelframe", background=self.bg_color, foreground=self.pink_accent)
        self.style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.pink_accent, font=("Segoe UI", 10, "bold"))
        
        self.root.configure(bg=self.bg_color)

        # --- Keyboard Binding ---
        self.root.bind('<space>', self.toggle_pause)
        self.root.focus_force() 

        # --- GPU Configuration ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("C:/Users/Soumyajit/Pictures/Applying Universities/NCKU/Courses/Sem 2/BIODESIGN(2) - IMPLEMENTATION OF INNOVATIVE MEDICAL DEVICES/GUI/yolo11m-pose.pt")
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
        
        # --- UI Layout: Header ---
        header_frame = tk.Frame(root, bg=self.bg_color, pady=10)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header_frame, text="Breast Cancer Rehabilitation Dashboard", font=("Segoe UI", 24, "bold"), 
                 bg=self.bg_color, fg=self.pink_accent).pack(side=tk.LEFT, padx=20)
        
        try:
            logo_path = "C:/Users/Soumyajit/Pictures/Applying Universities/NCKU/Courses/Sem 2/BIODESIGN(2) - IMPLEMENTATION OF INNOVATIVE MEDICAL DEVICES/GUI/Brecaresia 1.png"
            logo_img = Image.open(logo_path).resize((80, 80)) 
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(header_frame, image=self.logo_photo, bg=self.bg_color)
            logo_label.pack(side=tk.RIGHT, padx=30)
        except Exception as e:
            print(f"Logo not found: {e}")

        # --- Control Panel Layout ---
        control_panel = ttk.Frame(root, padding=10)
        control_panel.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = ttk.Frame(control_panel)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(control_panel)
        row2_frame.pack(fill=tk.X, pady=2)
        
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

        # Group 4: Scheduling
        sched_frame = ttk.LabelFrame(row2_frame, text=" Exercise Reminder ", padding=5)
        sched_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        tk.Label(sched_frame, text="Time (HH:MM):", bg=self.bg_color, fg=self.text_white).pack(side=tk.LEFT, padx=5)
        self.time_entry = ttk.Entry(sched_frame, width=8)
        self.time_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(sched_frame, text="Set", command=self.set_reminder).pack(side=tk.LEFT, padx=2)

        # Group 5: Final Data Recording
        record_frame = ttk.LabelFrame(row2_frame, text=" Daily Reporting ", padding=5)
        record_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(record_frame, text="Export Final Values", command=self.export_excel).pack(side=tk.LEFT, padx=5)

        # Group 6: Machine Learning Prediction
        ml_frame = ttk.LabelFrame(row2_frame, text=" Machine Learning Prediction ", padding=5)
        ml_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(ml_frame, text="Train the ML model", command=self.train_ml_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(ml_frame, text="Predict the Stage", command=self.predict_stage).pack(side=tk.LEFT, padx=5)

        self.check_schedule()

        # --- Main Display ---
        display_frame = tk.Frame(root, bg="#0d1a0d", bd=2, relief="groove") 
        display_frame.pack(pady=10, padx=20, expand=True, fill=tk.BOTH)
        self.label_proc = tk.Label(display_frame, text="System Ready", bg="#0d1a0d", fg=self.pink_accent)
        self.label_proc.pack(expand=True)
        
        # --- Results Dashboard ---
        res_frame = tk.Frame(root, bg="#112211", pady=10) 
        res_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.feedback_label = tk.Label(res_frame, text="Awaiting Movement...", 
                                      font=("Segoe UI", 18, "bold"), bg="#112211", fg=self.pink_accent)
        self.feedback_label.pack()

        self.results_display = tk.Label(res_frame, text="Real-time Metrics: Awaiting Data...", 
                                        font=("Segoe UI", 14), bg="#112211", fg="#ffffff")
        self.results_display.pack()
        
        self.status_label = tk.Label(root, text=f"Device: {self.device.upper()}", bg=self.bg_color, fg=self.light_pink, font=("Segoe UI", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- ML Prediction Logic ---
    def train_ml_model(self):
        try:
            df = pd.read_csv("training.csv")
            X = df[self.exercise_columns]
            
            self.kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = self.kmeans_model.fit_predict(X)
            
            df['Cluster'] = clusters
            self.cluster_to_group_map = {}
            for i in range(4):
                mode_group = df[df['Cluster'] == i]['Group'].mode()
                self.cluster_to_group_map[i] = mode_group[0] if not mode_group.empty else f"Cluster {i+1}"

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k')
            plt.title("K-means Clustering of Exercise Data (PCA Reduced)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.colorbar(scatter, label='Cluster ID')
            
            graph_path = "clustering_analysis.png"
            plt.savefig(graph_path)
            plt.close()
            
            messagebox.showinfo("ML Success", f"Model trained successfully!\nPCA graph saved as '{graph_path}'.")
        except Exception as e:
            messagebox.showerror("ML Error", f"Could not train model: {str(e)}")

    def predict_stage(self):
        if self.kmeans_model is None:
            messagebox.showwarning("Predict Error", "Please train the ML model first.")
            return
            
        if not self.results_data:
            messagebox.showwarning("Predict Error", "No inference data found. Please run exercises first.")
            return

        current_data = {ex: 0.0 for ex in self.exercises}
        for item in self.results_data:
            if item['Exercise'] in current_data:
                current_data[item['Exercise']] = item['Measurement']
        
        input_vals = [
            current_data["Chest expansion"],
            current_data["Hand gripping"],
            current_data["Shoulder circumduction"],
            current_data["Upper limb circumduction"],
            current_data["Wall walking"]
        ]
        input_vector = np.array([input_vals])

        try:
            prediction_idx = self.kmeans_model.predict(input_vector)[0]
            group_name = self.cluster_to_group_map.get(prediction_idx, "Unknown Group")
            
            # --- Generate Explanation Graph ---
            cluster_centers = self.kmeans_model.cluster_centers_[prediction_idx]
            
            labels = ["Chest", "Grip", "Shoulder", "U.Limb", "Wall"]
            x = np.arange(len(labels))
            width = 0.35

            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, input_vals, width, label='Your Performance', color='#e91e63')
            plt.bar(x + width/2, cluster_centers, width, label=f'Group Avg ({group_name})', color='#1a3320')

            plt.ylabel('Measurement Values')
            plt.title(f'Comparison: Why you belong to {group_name}')
            plt.xticks(x, labels)
            plt.legend()
            
            explanation_path = "prediction_explanation.png"
            plt.savefig(explanation_path)
            plt.show() # This opens the graph window immediately
            
            messagebox.showinfo("Stage Prediction", 
                                f"Predicted Group: {group_name}\n\n"
                                f"An explanation graph has been generated and displayed.")
                                
        except Exception as e:
            messagebox.showerror("Predict Error", f"Prediction failed: {str(e)}")

    # --- Logic ---
    def set_reminder(self):
        user_time = self.time_entry.get().strip()
        try:
            datetime.datetime.strptime(user_time, "%H:%M")
            self.reminder_time = user_time
            messagebox.showinfo("Reminder", f"Exercise reminder set for {user_time}")
        except ValueError:
            messagebox.showerror("Error", "Invalid Format. Please use HH:MM (e.g., 14:30)")

    def check_schedule(self):
        if self.reminder_time:
            now = datetime.datetime.now().strftime("%H:%M")
            if now == self.reminder_time:
                self.reminder_time = None 
                threading.Thread(target=lambda: messagebox.showwarning("Exercise Reminder", "It is time for your rehabilitation exercises!")).start()
        self.root.after(10000, self.check_schedule)

    def export_excel(self):
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            now = datetime.datetime.now()
            df['Date'] = now.strftime("%Y-%m-%d")
            df['Time_of_Measurement'] = now.strftime("%H:%M:%S")
            filename = f"Daily_Report_{now.strftime('%Y%m%d')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Daily measurement recorded in {filename}")
        else:
            messagebox.showwarning("No Data", "Run inference first.")

    def toggle_pause(self, event=None):
        if self.pause_event.is_set(): self.pause_inference()
        else: self.resume_inference()

    def toggle_recording(self):
        if not self.is_webcam_running: return
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            self.is_recording = True
            self.record_btn.config(text="Stop Rec")
        else:
            self.is_recording = False
            if self.video_writer: self.video_writer.release(); self.video_writer = None
            self.record_btn.config(text="Record")

    def pause_inference(self): 
        self.pause_event.clear()
        self.status_label.config(text="Status: Paused")
        
    def resume_inference(self): 
        self.pause_event.set()
        self.status_label.config(text=f"Status: Playing | Device: {self.device.upper()}")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    def analyze_and_annotate(self, frame, name, wrist_path, state):
        results = self.model(frame, verbose=False, device=self.device, half=(self.device=='cuda'))
        annotated = results[0].plot()
        metrics = {"val": 0, "type": ""}
        is_correct = False
        
        if results[0].keypoints.xy.shape[1] > 9:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            s, e, w = kpts[5], kpts[7], kpts[9]
            
            if name == "Chest expansion":
                metrics = {"val": self.calculate_angle(s, e, w), "type": "Angle (Degrees)"}
                if metrics["val"] > 160: is_correct = True
            elif name == "Hand gripping":
                metrics = {"val": self.calculate_angle(e, w, [w[0], w[1]+100]), "type": "Wrist Angle (Degrees)"}
                if metrics["val"] > 30: is_correct = True
            elif name in ["Shoulder circumduction", "Upper limb circumduction"]:
                wrist_path.append(w)
                if len(wrist_path) > 5:
                    hull = cv2.convexHull(np.array(wrist_path).astype(np.int32))
                    metrics = {"val": cv2.contourArea(hull), "type": "Area (Pixels^2)"}
                    if metrics["val"] > 5000: is_correct = True
            elif name == "Wall walking":
                if state['start_y'] is None: state['start_y'] = w[1]
                metrics = {"val": abs(state['start_y'] - w[1]), "type": "Y-Distance (Pixels)"}
                if metrics["val"] > 50: is_correct = True

        return annotated, metrics, is_correct

    def update_ui_text(self, metrics, is_correct, session_id):
        if session_id == self.active_session_id:
            self.results_display.config(text=f"{metrics['type']}: {metrics['val']:.2f}")
            if is_correct:
                self.feedback_label.config(text="EXERCISE CORRECT ✅", fg="#2e7d32")
            else:
                self.feedback_label.config(text="ALERT: ADJUST MOVEMENT ⚠️", fg="#d32f2f")

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
        if self.is_recording: self.toggle_recording()

    def run_webcam_loop(self, name):
        cap = cv2.VideoCapture(0)
        wrist_path, state = [], {"start_y": None}
        while self.is_webcam_running and cap.isOpened():
            self.pause_event.wait()
            ret, frame = cap.read()
            if not ret: break
            ann, m, corr = self.analyze_and_annotate(frame, name, wrist_path, state)
            if self.is_recording and self.video_writer: self.video_writer.write(ann)
            self.update_gui_image(self.label_proc, cv2.resize(ann, (self.display_width, self.display_height)), session_id=0)
            self.root.after(1, lambda: self.update_ui_text(m, corr, session_id=0))
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
            if self.inference_stop_event.is_set(): return
            cap = cv2.VideoCapture(path)
            wrist_path, state, max_val = [], {"start_y": None}, 0
            while cap.isOpened():
                if self.inference_stop_event.is_set(): cap.release(); return
                self.pause_event.wait()
                ret, frame = cap.read()
                if not ret: break
                ann, m, corr = self.analyze_and_annotate(frame, name, wrist_path, state)
                max_val = max(max_val, m["val"])
                self.update_gui_image(self.label_proc, cv2.resize(ann, (600, 400)), session_id)
                self.root.after(1, lambda m=m, c=corr, s=session_id: self.update_ui_text(m, c, s))
            cap.release()
            if session_id == self.active_session_id:
                self.results_data.append({"Exercise": name, "Measurement": round(max_val, 2)})

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseGUI(root)
    root.mainloop()