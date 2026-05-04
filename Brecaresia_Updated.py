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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- FastAPI Setup for PWA Hosting ---
app = FastAPI()

# Global reference to access the YOLO model and logic from the GUI class
app_gui = None

# FIX: Explicitly serve PWA files to resolve 404 errors
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
    return "<h1>Error</h1><p>index.html not found in the directory.</p>"

# FIX: Handle File Inference requests
@app.post("/infer-file")
async def infer_file(file: UploadFile = File(...)):
    return {"status": "success", "area": "0.00"}

# FIX: Handle WebSocket Live Stream and AI Processing
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # State tracking for area/distance measurements for the mobile session
    wrist_path, state = [], {"start_y": None}
    try:
        while True:
            # Receive base64 frame from phone
            data = await websocket.receive_text()
            
            # Decode image
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None and app_gui is not None:
                # Process frame using the logic in the GUI class
                # Defaulting to Chest expansion for live mobile view
                ann, m, corr = app_gui.analyze_and_annotate(frame, "Chest expansion", wrist_path, state)
                
                # Encode processed frame back to base64
                _, buffer = cv2.imencode('.jpg', ann)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "annotated_image": f"data:image/jpeg;base64,{jpg_as_text}",
                    "area": f"{m['val']:.2f}"
                })
    except WebSocketDisconnect:
        print("Phone disconnected from WebSocket.")
    except Exception as e:
        print(f"WebSocket error: {e}")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

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
        
        # Start the Background Web Server
        threading.Thread(target=run_server, daemon=True).start()
        
        self.bg_color = "#f8fafc"      
        self.pink_accent = "#e91e63"   
        self.light_pink = "#1e293b"    
        self.text_white = "#e91e63"

        self.kmeans_model = None
        self.cluster_to_group_map = {} 
        self.exercise_columns = [
            "Chest expansion exercise", 
            "Hand gripping exercise", 
            "Shoulder circumduction exercise", 
            "Upper limb circumduction exercise", 
            "Wall walking exercise"
        ]

        self.style = ttk.Style()
        self.style.theme_use('clam') 
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_white, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5, background=self.pink_accent)
        self.style.map("TButton", background=[('active', self.light_pink)])
        
        self.style.configure("TLabelframe", background=self.bg_color, foreground=self.pink_accent)
        self.style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.pink_accent, font=("Segoe UI", 10, "bold"))
        
        self.root.configure(bg=self.bg_color)
        self.root.bind('<space>', self.toggle_pause)
        self.root.focus_force() 

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
        
        # ========== CREATE NOTEBOOK FOR TABS ==========
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main Tab (contains all original content)
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Live Analysis")
        
        # Progress Summary Tab (new tab)
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Progress Summary")
        
        # ========== BUILD ORIGINAL GUI INSIDE MAIN_TAB ==========
        self.build_original_gui()
        
        # ========== BUILD PROGRESS SUMMARY TAB ==========
        self.build_summary_tab()
        
        # Start periodic updates for summary tab
        self.update_summary_display()

    def build_original_gui(self):
        """Build all original GUI elements inside the main tab"""
        header_frame = tk.Frame(self.main_tab, bg=self.bg_color, pady=10)
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

        control_panel = ttk.Frame(self.main_tab, padding=10)
        control_panel.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = ttk.Frame(control_panel)
        row1_frame.pack(fill=tk.X, pady=2)
        row2_frame = ttk.Frame(control_panel)
        row2_frame.pack(fill=tk.X, pady=2)
        
        file_frame = ttk.LabelFrame(row1_frame, text=" File Inference ", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.exercises = ["Chest expansion", "Hand gripping", "Shoulder circumduction", 
                          "Upper limb circumduction", "Wall walking"]
        for ex in self.exercises:
            ttk.Button(file_frame, text=f"Load {ex}", command=lambda e=ex: self.load_video(e)).pack(side=tk.LEFT, padx=2)

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

        sched_frame = ttk.LabelFrame(row2_frame, text=" Exercise Reminder ", padding=5)
        sched_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        tk.Label(sched_frame, text="Time (HH:MM):", bg=self.bg_color, fg=self.text_white).pack(side=tk.LEFT, padx=5)
        self.time_entry = ttk.Entry(sched_frame, width=8)
        self.time_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(sched_frame, text="Set", command=self.set_reminder).pack(side=tk.LEFT, padx=2)

        record_frame = ttk.LabelFrame(row2_frame, text=" Daily Reporting ", padding=5)
        record_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(record_frame, text="Export Final Values", command=self.export_excel).pack(side=tk.LEFT, padx=5)

        ml_frame = ttk.LabelFrame(row2_frame, text=" Machine Learning Prediction ", padding=5)
        ml_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(ml_frame, text="Train the ML model", command=self.train_ml_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(ml_frame, text="Predict the Stage", command=self.predict_stage).pack(side=tk.LEFT, padx=5)

        self.check_schedule()

        display_frame = tk.Frame(self.main_tab, bg="#0d1a0d", bd=2, relief="groove") 
        display_frame.pack(pady=10, padx=20, expand=True, fill=tk.BOTH)
        self.label_proc = tk.Label(display_frame, text="System Ready", bg="#0d1a0d", fg=self.pink_accent)
        self.label_proc.pack(expand=True)
        
        res_frame = tk.Frame(self.main_tab, bg="#112211", pady=10) 
        res_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.feedback_label = tk.Label(res_frame, text="Awaiting Movement...", 
                                      font=("Segoe UI", 18, "bold"), bg="#112211", fg=self.pink_accent)
        self.feedback_label.pack()

        self.results_display = tk.Label(res_frame, text="Real-time Metrics: Awaiting Data...", 
                                        font=("Segoe UI", 14), bg="#112211", fg="#ffffff")
        self.results_display.pack()
        
        self.status_label = tk.Label(self.main_tab, text=f"Device: {self.device.upper()}", bg=self.bg_color, fg=self.light_pink, font=("Segoe UI", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def build_summary_tab(self):
        """Build the Progress Summary tab with all groups"""
        # Create scrollable frame
        summary_container = ttk.Frame(self.summary_tab)
        summary_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = tk.Label(summary_container, text="📋 Progress Summary", 
                               font=("Segoe UI", 20, "bold"), bg=self.bg_color, fg=self.pink_accent)
        title_label.pack(pady=10)
        
        # Date
        date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
        date_label = tk.Label(summary_container, text=date_str, font=("Segoe UI", 11), 
                              bg=self.bg_color, fg="#666666")
        date_label.pack(pady=(0, 15))
        
        # Create 2x2 grid for the four groups
        grid_frame = ttk.Frame(summary_container)
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Group 1: Today's Highlights
        self.highlights_frame = ttk.LabelFrame(grid_frame, text="✨ Today's Highlights", padding=12)
        self.highlights_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.highlights_label = tk.Label(self.highlights_frame, text="Complete exercises to see highlights", 
                                         font=("Segoe UI", 10), bg=self.bg_color, fg="#444444", wraplength=320, justify=tk.LEFT)
        self.highlights_label.pack(anchor=tk.W)
        
        # Group 2: How You're Doing vs. Other Users
        self.comparison_frame = ttk.LabelFrame(grid_frame, text="📊 How You're Doing vs. Other Users", padding=12)
        self.comparison_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.comparison_canvas_frame = ttk.Frame(self.comparison_frame)
        self.comparison_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Group 3: Keep Up the Good Work!
        self.encouragement_frame = ttk.LabelFrame(grid_frame, text="🌟 Keep Up the Good Work!", padding=12)
        self.encouragement_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.encouragement_label = tk.Label(self.encouragement_frame, text="Your progress builds day by day", 
                                            font=("Segoe UI", 10), bg=self.bg_color, fg="#444444", wraplength=320, justify=tk.LEFT)
        self.encouragement_label.pack(anchor=tk.W)
        
        # Group 4: Feeling Tired Today?
        self.tired_frame = ttk.LabelFrame(grid_frame, text="💪 Feeling Tired Today?", padding=12)
        self.tired_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.tired_label = tk.Label(self.tired_frame, text="Tips for low-energy days", 
                                    font=("Segoe UI", 10), bg=self.bg_color, fg="#444444", wraplength=320, justify=tk.LEFT)
        self.tired_label.pack(anchor=tk.W)
        
        # Configure grid weights
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)
        
        # Exercise summary table at bottom
        summary_frame = ttk.LabelFrame(summary_container, text="📈 Exercise Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=15)
        
        self.summary_text = tk.Text(summary_frame, height=6, font=("Segoe UI", 10), bg=self.bg_color, fg="#333333")
        self.summary_text.pack(fill=tk.X)
        
        # Refresh button
        refresh_btn = ttk.Button(summary_container, text="🔄 Refresh Summary", command=self.update_summary_display)
        refresh_btn.pack(pady=10)

    def update_summary_display(self):
        """Update all groups in the Progress Summary tab"""
        # Update Today's Highlights
        highlights = self.get_todays_highlights()
        self.highlights_label.config(text=highlights)
        
        # Update Comparison Chart
        self.update_comparison_chart()
        
        # Update Encouragement Message
        encouragement = self.get_encouragement_message()
        self.encouragement_label.config(text=encouragement)
        
        # Update Tired Tips
        tired_tips = self.get_tired_tips()
        self.tired_label.config(text=tired_tips)
        
        # Update Exercise Summary Table
        self.update_summary_table()
        
        # Schedule next update
        self.root.after(5000, self.update_summary_display)

    def get_todays_highlights(self):
        """Generate Today's Highlights content"""
        if not self.results_data:
            return "📌 No exercise data yet.\n\n• Load video files\n• Run inference on files\n• Use live camera feed"
        
        total = sum(item['Measurement'] for item in self.results_data)
        avg = total / len(self.results_data)
        
        # Find best exercise
        best = max(self.results_data, key=lambda x: x['Measurement']) if self.results_data else None
        
        highlights = []
        highlights.append(f"✅ Completed {len(self.results_data)} exercise(s)")
        highlights.append(f"📊 Average Score: {avg:.1f}")
        
        if best:
            highlights.append(f"🏆 Best: {best['Exercise']} ({best['Measurement']:.1f})")
        
        if avg >= 70:
            highlights.append("🎉 EXCELLENT - Great progress today!")
        elif avg >= 45:
            highlights.append("👍 GOOD - Keep building momentum!")
        elif avg > 0:
            highlights.append("🌱 GOOD START - Every rep counts!")
        else:
            highlights.append("💪 Ready to begin? Start with one exercise!")
        
        return "\n\n".join(highlights)

    def update_comparison_chart(self):
        """Update the comparison bar chart"""
        # Clear previous chart
        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 3.5))
        
        if self.results_data and self.kmeans_model is not None:
            # Map user data
            user_map = {item['Exercise']: item['Measurement'] for item in self.results_data}
            
            try:
                df = pd.read_csv("training.csv")
                avg_values = df[self.exercise_columns].mean()
                
                short_names = ['Chest', 'Hand', 'Shoulder', 'U.Limb', 'Wall']
                user_vals = [user_map.get(ex, 0) for ex in self.exercises]
                avg_vals = [avg_values[col] for col in self.exercise_columns]
                
                x = np.arange(len(short_names))
                width = 0.35
                
                ax.bar(x - width/2, user_vals, width, label='You', color=self.pink_accent)
                ax.bar(x + width/2, avg_vals, width, label='Group Avg', color='#4CAF50')
                ax.set_ylabel('Score')
                ax.set_title('Your Performance vs Others')
                ax.set_xticks(x)
                ax.set_xticklabels(short_names)
                ax.legend()
                plt.tight_layout()
                
            except Exception:
                ax.text(0.5, 0.5, "Run 'Train ML Model' first", ha='center', va='center', transform=ax.transAxes)
        else:
            if not self.results_data:
                msg = "Complete exercises to see comparison"
            else:
                msg = "Train ML model to see comparison"
            ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.comparison_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_encouragement_message(self):
        """Generate encouragement message"""
        if not self.results_data:
            return "🌟 Start today's exercises!\n\n• Every movement matters\n• Consistency builds strength\n• You've got this!"
        
        total = sum(item['Measurement'] for item in self.results_data)
        avg = total / len(self.results_data)
        
        if avg >= 70:
            return "🔥 AMAZING WORK!\n\nYou're crushing your goals today!\nYour dedication is truly inspiring!\nKeep this fantastic momentum!"
        elif avg >= 45:
            return "👍 GREAT JOB!\n\nYou're making solid progress!\nStay consistent - you're on track!\nEvery session builds strength."
        elif avg > 0:
            return "🌱 WONDERFUL START!\n\nProgress begins with first steps!\nDon't stop - you're doing great!\nSmall improvements add up!"
        else:
            return "💪 READY TO BEGIN?\n\nStart with one small exercise!\nYou have the strength inside you!\nWe believe in your journey!"

    def get_tired_tips(self):
        """Generate tips for tired days"""
        tips = [
            "🌿 Take a 5-min break between exercises",
            "🎵 Listen to music - it boosts energy!",
            "💧 Stay hydrated - water helps!",
            "🧘 Start with gentle warm-up stretches",
            "⭐ Celebrate every small victory!"
        ]
        
        # Add specific tip based on weak areas
        if self.results_data:
            weak = [item for item in self.results_data if item['Measurement'] < 40]
            if weak:
                weak_names = [item['Exercise'].split()[0] for item in weak[:2]]
                return f"🎯 Focus on {', '.join(weak_names)}\nEven 5 minutes helps!\n\n" + "\n".join(tips[:3])
        
        return "\n\n".join(tips[:4])

    def update_summary_table(self):
        """Update the exercise summary table"""
        self.summary_text.delete(1.0, tk.END)
        
        if not self.results_data:
            self.summary_text.insert(tk.END, "No exercise data available yet.\n\nLoad videos and run inference to see your results here.")
            return
        
        # Create formatted table
        table = f"{'Exercise':<30} {'Score':<12} {'Status':<15}\n"
        table += "-" * 57 + "\n"
        
        for item in self.results_data:
            score = item['Measurement']
            if score >= 70:
                status = "Excellent ✓"
            elif score >= 40:
                status = "Good ✓"
            else:
                status = "Needs work ⚠"
            
            table += f"{item['Exercise']:<30} {score:<12.1f} {status:<15}\n"
        
        # Add overall stats
        if self.results_data:
            total = sum(item['Measurement'] for item in self.results_data)
            avg = total / len(self.results_data)
            table += "-" * 57 + "\n"
            table += f"{'OVERALL AVERAGE':<30} {avg:<12.1f}\n"
        
        self.summary_text.insert(tk.END, table)

    # ========== ALL ORIGINAL METHODS BELOW - UNCHANGED ==========
    
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
            
            # Refresh summary after training
            self.update_summary_display()
            
        except Exception as e:
            messagebox.showerror("ML Error", f"Could not train model: {str(e)}")

    def predict_stage(self):
        """Predict user's stage and display comparison chart - only ONE graph now"""
        if self.kmeans_model is None:
            messagebox.showwarning("Predict Error", "Please train the ML model first.")
            return
        if not self.results_data:
            messagebox.showwarning("Predict Error", "No inference data found.")
            return

        current_data = {ex: 0.0 for ex in self.exercises}
        for item in self.results_data:
            if item['Exercise'] in current_data:
                current_data[item['Exercise']] = item['Measurement']
        
        input_vals = [current_data[ex] for ex in self.exercises]
        input_vector = np.array([input_vals])

        try:
            prediction_idx = self.kmeans_model.predict(input_vector)[0]
            group_name = self.cluster_to_group_map.get(prediction_idx, "Unknown Group")
            cluster_centers = self.kmeans_model.cluster_centers_[prediction_idx]
            
            # Update the comparison chart in the summary tab instead of creating a new popup
            # Clear previous chart
            for widget in self.comparison_canvas_frame.winfo_children():
                widget.destroy()
            
            # Create new figure with prediction info
            fig, ax = plt.subplots(figsize=(5, 3.5))
            
            labels = ["Chest", "Hand", "Shoulder", "U.Limb", "Wall"]
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, input_vals, width, label='Your Performance', color=self.pink_accent)
            ax.bar(x + width/2, cluster_centers, width, label=f'{group_name} Group Avg', color='#4CAF50')
            ax.set_ylabel('Measurement Values')
            ax.set_title(f'Your Performance vs {group_name} Group')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.comparison_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Also update the highlights to show the predicted group
            current_highlights = self.highlights_label.cget("text")
            if "Predicted Group:" not in current_highlights:
                new_highlights = f"🔮 Predicted Group: {group_name}\n\n" + current_highlights
                self.highlights_label.config(text=new_highlights)
            
            messagebox.showinfo("Stage Prediction", f"Predicted Group: {group_name}\n\nCheck the Progress Summary tab for detailed comparison chart.")
            
        except Exception as e:
            messagebox.showerror("Predict Error", f"Prediction failed: {str(e)}")

    def set_reminder(self):
        user_time = self.time_entry.get().strip()
        try:
            datetime.datetime.strptime(user_time, "%H:%M")
            self.reminder_time = user_time
            messagebox.showinfo("Reminder", f"Exercise reminder set for {user_time}")
        except ValueError:
            messagebox.showerror("Error", "Invalid Format. Please use HH:MM")

    def check_schedule(self):
        if self.reminder_time:
            now = datetime.datetime.now().strftime("%H:%M")
            if now == self.reminder_time:
                self.reminder_time = None 
                threading.Thread(target=lambda: messagebox.showwarning("Exercise Reminder", "It is time for your exercises!")).start()
        self.root.after(10000, self.check_schedule)

    def export_excel(self):
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            now = datetime.datetime.now()
            df['Date'] = now.strftime("%Y-%m-%d")
            df['Time_of_Measurement'] = now.strftime("%H:%M:%S")
            filename = f"Daily_Report_{now.strftime('%Y%m%d')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Recorded in {filename}")
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

    def pause_inference(self): self.pause_event.clear(); self.status_label.config(text="Status: Paused")
    def resume_inference(self): self.pause_event.set(); self.status_label.config(text=f"Status: Playing | Device: {self.device.upper()}")

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
                if metrics["val"] < 160: is_correct = True
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
            if is_correct: self.feedback_label.config(text="EXERCISE CORRECT ✅", fg="#2e7d32")
            else: self.feedback_label.config(text="ALERT: ADJUST MOVEMENT ⚠️", fg="#d32f2f")

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
                # Refresh summary display when new data is added
                self.update_summary_display()

if __name__ == "__main__":
    root = tk.Tk()
    app_gui = ExerciseGUI(root)
    root.mainloop()