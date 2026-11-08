"""
Enhanced Modern Two-Lane Vehicle Counter
Combines modern UI from first code with advanced tracking from second code
Scrollable left panel added
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from pathlib import Path
import time
import cv2
from PIL import Image, ImageTk
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict, deque

from detection.yolo_detection import YOLODetector
from tracking.byte_tracker import ByteTracker

# Class mapping
CLASS_MAP = {
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Bus',
    'motorcycle': 'Motorcycle',
    'motorbike': 'Motorcycle',
    'bicycle': 'Motorcycle',
    'scooter': 'Motorcycle',
    'van': 'Car'  # Map van to Car
}

# Per-class confidence & minimum area thresholds (from second code)
PER_CLASS_CONF = {
    'car': 0.45,
    'motorcycle': 0.55,
    'bus': 0.5,
    'truck': 0.55,
    'van': 0.45
}

PER_CLASS_AREA = {
    'car': 500,
    'motorcycle': 120,
    'bus': 2000,
    'truck': 2000,
    'van': 800
}

# Colors for different vehicle types (BGR format)
COLORS = {
    'Car': (0, 255, 0),        # Green
    'Motorcycle': (0, 165, 255), # Orange
    'Bus': (255, 0, 0),        # Blue
    'Truck': (0, 0, 255),      # Red
    'default': (255, 255, 255)  # White
}

class EnhancedTracker:
    """Advanced tracker with occlusion handling (from second code)"""
    
    def __init__(self, max_disappeared=30):
        self.tracker = ByteTracker(max_disappeared=max_disappeared)
        
        # For occlusion handling
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_classes = defaultdict(lambda: deque(maxlen=10))
        self.track_last_seen = {}
        self.track_velocities = {}
        self.max_disappeared = max_disappeared
        
    def update(self, detections, frame_num):
        """Update tracker with new detections."""
        tracks = self.tracker.update(detections)
        
        # Update track history
        current_tracks = set()
        for track in tracks:
            tid = track.get('track_id')
            if tid is None:
                continue
                
            # Calculate centroid if not present
            centroid = track.get('centroid')
            if centroid is None and 'bbox' in track:
                x1, y1, x2, y2 = track['bbox']
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                track['centroid'] = centroid
            
            if centroid:
                # Update history
                self.track_history[tid].append(centroid)
                self.track_last_seen[tid] = frame_num
                current_tracks.add(tid)
                
                # Track class history
                cls = track.get('class_name', '')
                if cls:
                    self.track_classes[tid].append(cls)
                
                # Calculate velocity
                if len(self.track_history[tid]) >= 2:
                    prev = self.track_history[tid][-2]
                    curr = centroid
                    self.track_velocities[tid] = (curr[0] - prev[0], curr[1] - prev[1])
        
        # Handle occluded tracks (predict positions)
        occluded_tracks = []
        for tid, last_seen in self.track_last_seen.items():
            if tid not in current_tracks:
                frames_missing = frame_num - last_seen
                if frames_missing <= self.max_disappeared:
                    # Predict position based on velocity
                    if tid in self.track_velocities and len(self.track_history[tid]) > 0:
                        vx, vy = self.track_velocities[tid]
                        last_pos = self.track_history[tid][-1]
                        predicted_pos = (last_pos[0] + vx * frames_missing, 
                                       last_pos[1] + vy * frames_missing)
                        
                        # Get most common class
                        if tid in self.track_classes and self.track_classes[tid]:
                            class_counts = defaultdict(int)
                            for cls in self.track_classes[tid]:
                                class_counts[cls] += 1
                            pred_class = max(class_counts.items(), key=lambda x: x[1])[0]
                        else:
                            pred_class = 'unknown'
                        
                        occluded_tracks.append({
                            'track_id': tid,
                            'centroid': predicted_pos,
                            'class_name': pred_class,
                            'occluded': True,
                            'bbox': [predicted_pos[0]-15, predicted_pos[1]-15, 
                                    predicted_pos[0]+15, predicted_pos[1]+15]
                        })
        
        return tracks + occluded_tracks

class LoadingBarFrame(ttk.Frame):
    """Custom frame with loading bar for vehicle counts"""
    
    def __init__(self, parent, vehicle_type, icon, color, max_value=100):
        super().__init__(parent)
        self.vehicle_type = vehicle_type
        self.color = color
        self.max_value = max_value
        self.current_value = 0
        
        # Configure grid
        self.columnconfigure(1, weight=1)
        
        # Icon and label
        self.icon_label = ttk.Label(self, text=icon, font=('Segoe UI', 12))
        self.icon_label.grid(row=0, column=0, padx=(0, 5))
        
        self.type_label = ttk.Label(self, text=vehicle_type, font=('Segoe UI', 10))
        self.type_label.grid(row=0, column=1, sticky='w')
        
        # Count label
        self.count_var = tk.IntVar(value=0)
        self.count_label = ttk.Label(self, textvariable=self.count_var, 
                                     font=('Segoe UI', 10, 'bold'))
        self.count_label.grid(row=0, column=2, padx=(5, 0))
        
        # Loading bar (progress bar)
        self.progress = ttk.Progressbar(self, orient='horizontal', 
                                        length=150, mode='determinate',
                                        style=f"{color}.Striped.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, columnspan=3, pady=(2, 5), sticky='ew')
        
        # Configure style for colored progress bar
        style = ttk.Style()
        style.configure(f"{color}.Striped.Horizontal.TProgressbar", 
                       background=color, troughcolor='#333333')
        
    def update_count(self, value):
        """Update the count and progress bar"""
        self.current_value = value
        self.count_var.set(value)
        
        # Update progress bar (percentage of max)
        if self.max_value > 0:
            percentage = min(100, (value / self.max_value) * 100)
            self.progress['value'] = percentage
        
    def set_max_value(self, max_value):
        """Set maximum value for progress bar scaling"""
        self.max_value = max_value
        self.update_count(self.current_value)


class ModernEnhancedVehicleCounter:
    def __init__(self):
        # Create main window with ttkbootstrap
        self.root = tb.Window(themename="darkly")
        self.root.title('Enhanced Modern Two-Lane Vehicle Counter')
        self.root.geometry('1400x800')
        
        # Video state
        self.video_path = ""
        self.cap = None
        self.detector = None
        self.enhanced_tracker = None
        
        self.is_playing = False
        self.play_thread = None
        
        # Frame info
        self.frame_count = 0
        self.fps = 30
        self.frame_w = 640
        self.frame_h = 480
        self.current_frame_num = 0
        
        # Two counting lines
        self.top_line_y = 0.4
        self.bottom_line_y = 0.7
        
        # Track state
        self.track_last_y = {}
        self.counted_tracks_top = set()
        self.counted_tracks_bottom = set()
        
        # Advanced tracking features
        self.track_paths = defaultdict(list)
        self.track_colors = {}
        self.var_show_paths = tk.BooleanVar(value=True)
        self.var_motorcycle_boost = tk.BooleanVar(value=True)
        self.var_motorcycle_boost_conf = tk.DoubleVar(value=0.35)
        self.var_motorcycle_boost_area = tk.IntVar(value=60)
        
        # Counts
        self.counts = {
            'up_lane': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0},
            'down_lane': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
        }
        
        # Store max counts for progress bar scaling
        self.max_counts = defaultdict(int)
        
        # Performance
        self.process_every_n_frames = 2
        self.frame_skip_counter = 0
        self.last_detections = []
        
        # Detection settings
        self.mc_sensitivity = 0.25
        self.USE_TILING = False  # Keep simple for performance
        self.DET_IMG_SIZE = 800
        
        # Store loading bar references
        self.up_lane_bars = {}
        self.down_lane_bars = {}
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (with scrollbar)
        left_container = ttk.Frame(main_container, width=400)
        left_container.pack(side=LEFT, fill=Y, padx=(0, 10))
        left_container.pack_propagate(False)
        
        # Create canvas with scrollbar for left panel
        canvas = tk.Canvas(left_container, highlightthickness=0, bg='#222222')
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        
        # Create scrollable frame
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Right panel - Video
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Setup control panel inside scrollable frame
        self.setup_control_panel(self.scrollable_frame)
        self.setup_video_panel(right_panel)
        self.setup_status_bar()
        
    def setup_control_panel(self, parent):
        # Header with icon
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="🚦 ENHANCED TWO-LANE VEHICLE COUNTER", 
                                font=('Segoe UI', 10, 'bold'))
        title_label.pack()
        
        # Video Controls Card
        video_card = ttk.LabelFrame(parent, text=" Video Controls ", padding=15)
        video_card.pack(fill=X, pady=(0, 10))
        
        # Browse button with icon
        browse_btn = ttk.Button(video_card, text="📁 Browse Video", 
                                command=self.browse_video, 
                                style="primary.TButton")
        browse_btn.pack(fill=X, pady=(0, 10))
        
        # Playback controls in a grid
        playback_frame = ttk.Frame(video_card)
        playback_frame.pack(fill=X)
        
        self.play_btn = ttk.Button(playback_frame, text="▶ Play", 
                                   command=self.play, state=DISABLED,
                                   style="success.TButton")
        self.play_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        
        self.pause_btn = ttk.Button(playback_frame, text="⏸ Pause", 
                                    command=self.pause, state=DISABLED,
                                    style="warning.TButton")
        self.pause_btn.pack(side=LEFT, expand=True, fill=X, padx=5)
        
        self.stop_btn = ttk.Button(playback_frame, text="⏹ Stop", 
                                   command=self.stop, state=DISABLED,
                                   style="danger.TButton")
        self.stop_btn.pack(side=LEFT, expand=True, fill=X, padx=(5, 0))
        
        # Seek controls (from second code)
        seek_frame = ttk.Frame(video_card)
        seek_frame.pack(fill=X, pady=5)
        ttk.Button(seek_frame, text="<< 5s", 
                  command=lambda: self.seek_seconds(-5)).pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        ttk.Button(seek_frame, text=">> 5s", 
                  command=lambda: self.seek_seconds(5)).pack(side=LEFT, expand=True, fill=X, padx=(5, 0))
        
        # Time display
        time_frame = ttk.Frame(video_card)
        time_frame.pack(fill=X, pady=(10, 0))
        
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        time_label = ttk.Label(time_frame, textvariable=self.time_var, 
                               font=('Segoe UI', 11, 'bold'))
        time_label.pack()
        
        # Lane Configuration Card
        lane_card = ttk.LabelFrame(parent, text=" Lane Configuration ", padding=15)
        lane_card.pack(fill=X, pady=(0, 10))
        
        # Up Lane (Blue)
        up_frame = ttk.Frame(lane_card)
        up_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(up_frame, text="⬆️ UP LANE", 
                 font=('Segoe UI', 10, 'bold'),
                 foreground='#00a8ff').pack(anchor=W)
        
        self.top_line_scale = ttk.Scale(up_frame, from_=0.2, to=0.6, value=0.4,
                                        command=self.on_top_line_change)
        self.top_line_scale.pack(fill=X, pady=5)
        
        self.top_line_label = ttk.Label(up_frame, text="40%")
        self.top_line_label.pack(anchor=E)
        
        # Down Lane (Green)
        down_frame = ttk.Frame(lane_card)
        down_frame.pack(fill=X)
        
        ttk.Label(down_frame, text="⬇️ DOWN LANE", 
                 font=('Segoe UI', 10, 'bold'),
                 foreground='#00cc88').pack(anchor=W)
        
        self.bottom_line_scale = ttk.Scale(down_frame, from_=0.5, to=0.9, value=0.7,
                                          command=self.on_bottom_line_change)
        self.bottom_line_scale.pack(fill=X, pady=5)
        
        self.bottom_line_label = ttk.Label(down_frame, text="70%")
        self.bottom_line_label.pack(anchor=E)
        
        # Detection Settings Card
        detect_card = ttk.LabelFrame(parent, text=" Detection Settings ", padding=15)
        detect_card.pack(fill=X, pady=(0, 10))
        
        # Motorcycle sensitivity
        sens_frame = ttk.Frame(detect_card)
        sens_frame.pack(fill=X)
        
        ttk.Label(sens_frame, text="🏍️ Motorcycle Sensitivity:").pack(anchor=W)
        
        self.mc_sens_var = tk.DoubleVar(value=0.25)
        self.mc_sens_scale = ttk.Scale(sens_frame, from_=0.1, to=0.5, 
                                       variable=self.mc_sens_var,
                                       command=self.on_sensitivity_change)
        self.mc_sens_scale.pack(fill=X, pady=5)
        
        sens_value_frame = ttk.Frame(sens_frame)
        sens_value_frame.pack(fill=X)
        ttk.Label(sens_value_frame, text="Lower = More sensitive").pack(side=LEFT)
        self.sens_value_label = ttk.Label(sens_value_frame, text="0.25")
        self.sens_value_label.pack(side=RIGHT)
        
        # Motorcycle boost (from second code)
        boost_frame = ttk.Frame(detect_card)
        boost_frame.pack(fill=X, pady=5)
        ttk.Checkbutton(boost_frame, text="🏍️ Motorcycle Boost Mode", 
                       variable=self.var_motorcycle_boost).pack(anchor=W)
        
        # Visualization options (from second code)
        viz_frame = ttk.Frame(detect_card)
        viz_frame.pack(fill=X, pady=5)
        ttk.Checkbutton(viz_frame, text="Show Vehicle Paths", 
                       variable=self.var_show_paths).pack(anchor=W)
        
        # Live Counts Display with Loading Bars
        counts_card = ttk.LabelFrame(parent, text=" Live Counts with Loading Bars ", padding=15)
        counts_card.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(counts_card)
        notebook.pack(fill=BOTH, expand=True)
        
        # Up Lane Tab with Loading Bars
        up_tab = ttk.Frame(notebook)
        notebook.add(up_tab, text="⬆️ UP LANE")
        self.setup_loading_bar_display(up_tab, 'up_lane')
        
        # Down Lane Tab with Loading Bars
        down_tab = ttk.Frame(notebook)
        notebook.add(down_tab, text="⬇️ DOWN LANE")
        self.setup_loading_bar_display(down_tab, 'down_lane')
        
        # Summary Tab
        summary_tab = ttk.Frame(notebook)
        notebook.add(summary_tab, text="📊 SUMMARY")
        self.setup_summary_display(summary_tab)
        
        # Action Buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Button(action_frame, text="🔄 Reset Counts", 
                  command=self.reset_counts,
                  style="secondary.TButton").pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        
        ttk.Button(action_frame, text="💾 Export CSV", 
                  command=self.export_csv,
                  style="info.TButton").pack(side=LEFT, expand=True, fill=X, padx=(5, 0))
        
    def setup_loading_bar_display(self, parent, lane):
        """Setup loading bar display for a lane"""
        # Vehicle types with icons and colors
        vehicles = [
            ("Car", 'Car', '#00ff00'),      # Green
            ("Motorcycle", 'Motorcycle', '#ffaa00'),  # Orange
            ("Bus", 'Bus', '#0066ff'),      # Blue
            ("Truck", 'Truck', '#ff3333')    # Red
        ]
        
        # Container for all bars
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        for icon_text, key, color in vehicles:
            # Create frame for this vehicle type
            bar_frame = LoadingBarFrame(container, key, icon_text, color)
            bar_frame.pack(fill=X, pady=5)
            
            # Store reference
            if lane == 'up_lane':
                self.up_lane_bars[key] = bar_frame
            else:
                self.down_lane_bars[key] = bar_frame
        
        # Separator
        ttk.Separator(container, orient=HORIZONTAL).pack(fill=X, pady=15)
        
        # Lane total with loading bar
        total_frame = ttk.Frame(container)
        total_frame.pack(fill=X)
        
        ttk.Label(total_frame, text="📊 LANE TOTAL", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=W)
        
        # Total bar
        total_bar_frame = ttk.Frame(total_frame)
        total_bar_frame.pack(fill=X, pady=5)
        
        self.total_count_var = tk.IntVar(value=0)
        ttk.Label(total_bar_frame, textvariable=self.total_count_var,
                 font=('Segoe UI', 12, 'bold')).pack(side=RIGHT)
        
        total_progress = ttk.Progressbar(total_bar_frame, orient='horizontal',
                                        length=200, mode='determinate',
                                        style="info.Striped.Horizontal.TProgressbar")
        total_progress.pack(fill=X, expand=True)
        
        if lane == 'up_lane':
            self.up_total_progress = total_progress
            self.up_total_var = self.total_count_var
        else:
            self.down_total_progress = total_progress
            self.down_total_var = self.total_count_var
        
    def setup_summary_display(self, parent):
        """Setup summary tab"""
        # Up Lane Summary
        up_frame = ttk.LabelFrame(parent, text="⬆️ UP LANE", padding=10)
        up_frame.pack(fill=X, pady=5)
        
        self.up_summary_var = tk.StringVar(value="Car: 0 | Motorcycle: 0 | Bus: 0 | Truck: 0")
        ttk.Label(up_frame, textvariable=self.up_summary_var, 
                 font=('Segoe UI', 10)).pack()
        
        # Down Lane Summary
        down_frame = ttk.LabelFrame(parent, text="⬇️ DOWN LANE", padding=10)
        down_frame.pack(fill=X, pady=5)
        
        self.down_summary_var = tk.StringVar(value="Car: 0 | Motorcycle: 0 | Bus: 0 | Truck: 0")
        ttk.Label(down_frame, textvariable=self.down_summary_var, 
                 font=('Segoe UI', 10)).pack()
        
        # Grand Total
        total_frame = ttk.LabelFrame(parent, text="📈 GRAND TOTAL", padding=15)
        total_frame.pack(fill=X, pady=10)
        
        self.grand_total_var = tk.StringVar(value="0")
        ttk.Label(total_frame, textvariable=self.grand_total_var, 
                 font=('Segoe UI', 16, 'bold')).pack()
        
    def setup_video_panel(self, parent):
        """Setup video display panel"""
        # Video canvas
        self.canvas = tk.Canvas(parent, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_var = tk.StringVar(value="✅ Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=SUNKEN, padding=(5, 2))
        status_bar.pack(side=BOTTOM, fill=X)
        
    def on_sensitivity_change(self, val):
        """Update motorcycle sensitivity"""
        self.mc_sensitivity = float(val)
        self.sens_value_label.config(text=f"{self.mc_sensitivity:.2f}")
        
    def browse_video(self):
        """Browse for video file"""
        path = filedialog.askopenfilename(
            title='Select Video File',
            filetypes=[
                ('Video Files', '*.mp4 *.avi *.mov *.mkv'),
                ('All Files', '*.*')
            ]
        )
        
        if path:
            self.video_path = path
            self.load_video(path)
            
    def load_video(self, path):
        """Load video file"""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return
            
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS)))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update duration
        duration = self.frame_count / self.fps
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.total_time_str = f"{minutes:02d}:{seconds:02d}"
        self.update_time_display(0)
        
        # Enable controls
        self.play_btn.config(state=NORMAL)
        self.stop_btn.config(state=NORMAL)
        self.status_var.set(f"✅ Loaded: {Path(path).name}")
        
        # Show first frame
        self.show_current_frame()
        
    def update_time_display(self, frame_num):
        """Update time display"""
        current = frame_num / self.fps
        minutes = int(current // 60)
        seconds = int(current % 60)
        time_str = f"{minutes:02d}:{seconds:02d} / {self.total_time_str}"
        self.time_var.set(time_str)
        
    def seek_seconds(self, seconds):
        """Seek forward/backward by specified seconds"""
        if not self.cap:
            return
            
        try:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            target = cur + int(seconds * self.fps)
            target = max(0, min(self.frame_count - 1, target))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            
            # Update display
            self.show_current_frame()
            self.update_time_display(target)
            
        except Exception as e:
            print(f"Seek error: {e}")
        
    def show_current_frame(self):
        """Show current frame without processing"""
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Draw counting lines
            top_y = int(self.frame_h * self.top_line_y)
            bottom_y = int(self.frame_h * self.bottom_line_y)
            
            # Up lane (blue)
            cv2.line(frame, (0, top_y), (self.frame_w, top_y), (255, 168, 0), 3)
            cv2.putText(frame, "UP LANE", (10, top_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Down lane (green)
            cv2.line(frame, (0, bottom_y), (self.frame_w, bottom_y), (0, 255, 0), 3)
            cv2.putText(frame, "DOWN LANE", (10, bottom_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            self.display_frame(frame)
            
            # Reset position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 
                        max(0, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
            
    def ensure_detector(self):
        """Ensure detector is loaded"""
        if self.detector is None:
            try:
                self.detector = YOLODetector(model_path='yolov8m.pt', conf_threshold=0.25)
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Could not load detector: {e}")
                return False
        return True
        
    def classify_vehicle(self, det):
        """Enhanced classification with per-class thresholds (from second code)"""
        cls_name = det.get('class_name', '').lower()
        confidence = det.get('confidence', 0)
        bbox = det.get('bbox', [0, 0, 0, 0])
        
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Map class name
        mapped_class = None
        for key, value in CLASS_MAP.items():
            if key in cls_name:
                mapped_class = value
                break
        
        if mapped_class is None:
            return None, 0
            
        # Get thresholds for this class
        base_class = 'motorcycle' if mapped_class == 'Motorcycle' else mapped_class.lower()
        conf_threshold = PER_CLASS_CONF.get(base_class, 0.45)
        area_threshold = PER_CLASS_AREA.get(base_class, 120)
        
        # Special handling for motorcycle (use sensitivity setting)
        if mapped_class == 'Motorcycle':
            conf_threshold = self.mc_sensitivity
            
        # Check thresholds
        if confidence >= conf_threshold and area >= area_threshold:
            return mapped_class, confidence
            
        return None, 0
        
    def apply_motorcycle_boost(self, frame, existing_detections):
        """Run additional low-threshold detection for motorcycles (from second code)"""
        try:
            old_conf = getattr(self.detector, 'conf_threshold', None)
            self.detector.conf_threshold = float(self.var_motorcycle_boost_conf.get())
            
            extra = self.detector.detect(frame, imgsz=self.DET_IMG_SIZE, augment=False)
            
            for e in extra:
                ename = e.get('class_name', '').lower()
                if 'motorcycle' in ename or 'motorbike' in ename:
                    x1, y1, x2, y2 = e['bbox']
                    if (x2 - x1) * (y2 - y1) >= int(self.var_motorcycle_boost_area.get()):
                        # Check overlap with existing detections
                        if not any(self.iou_box(e['bbox'], f['bbox']) > 0.4 
                                  for f in existing_detections):
                            e['class_name'] = 'Motorcycle'
                            existing_detections.append(e)
                            
            if old_conf is not None:
                self.detector.conf_threshold = old_conf
                
        except Exception as e:
            print(f"Motorcycle boost error: {e}")
            
        return existing_detections
        
    def iou_box(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
        
    def play(self):
        """Start playback"""
        if not self.cap:
            return
        if not self.ensure_detector():
            return
            
        # Initialize enhanced tracker
        self.enhanced_tracker = EnhancedTracker(max_disappeared=20)
            
        self.is_playing = True
        self.play_btn.config(state=DISABLED)
        self.pause_btn.config(state=NORMAL)
        self.stop_btn.config(state=NORMAL)
        
        self.frame_skip_counter = 0
        self.last_detections = []
        
        self.play_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.play_thread.start()
        
    def pause(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.config(state=NORMAL)
        self.pause_btn.config(state=DISABLED)
        
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        self.play_btn.config(state=NORMAL)
        self.pause_btn.config(state=DISABLED)
        self.stop_btn.config(state=DISABLED)
        
        # Clear track history
        self.track_paths.clear()
        self.track_colors.clear()
        self.track_last_y.clear()
        self.counted_tracks_top.clear()
        self.counted_tracks_bottom.clear()
        
        self.show_current_frame()
        self.status_var.set("⏹ Playback stopped")
        
    def reset_counts(self):
        """Reset all counts"""
        self.counted_tracks_top.clear()
        self.counted_tracks_bottom.clear()
        self.track_last_y.clear()
        self.track_paths.clear()
        self.track_colors.clear()
        self.max_counts.clear()
        
        for lane in ['up_lane', 'down_lane']:
            for cls in self.counts[lane]:
                self.counts[lane][cls] = 0
                
        self.update_count_displays()
        self.status_var.set("🔄 Counts reset")
        
    def on_top_line_change(self, val):
        """Update top line position"""
        self.top_line_y = float(val)
        self.top_line_label.config(text=f"{int(float(val)*100)}%")
        
    def on_bottom_line_change(self, val):
        """Update bottom line position"""
        self.bottom_line_y = float(val)
        self.bottom_line_label.config(text=f"{int(float(val)*100)}%")
        
    def update_max_counts(self):
        """Update maximum counts for progress bar scaling"""
        # Update max for each class
        for cls in self.counts['up_lane']:
            self.max_counts[cls] = max(
                self.max_counts.get(cls, 0),
                self.counts['up_lane'][cls],
                self.counts['down_lane'][cls]
            )
        
    def update_count_displays(self):
        """Update all count displays including loading bars"""
        # Update max counts first
        self.update_max_counts()
        
        # Update up lane loading bars
        up_total = 0
        for cls, bar in self.up_lane_bars.items():
            count = self.counts['up_lane'][cls]
            bar.set_max_value(max(1, self.max_counts.get(cls, 100)))
            bar.update_count(count)
            up_total += count
        
        # Update down lane loading bars
        down_total = 0
        for cls, bar in self.down_lane_bars.items():
            count = self.counts['down_lane'][cls]
            bar.set_max_value(max(1, self.max_counts.get(cls, 100)))
            bar.update_count(count)
            down_total += count
        
        # Update total progress bars
        grand_total = up_total + down_total
        max_total = max(up_total, down_total, 1)
        
        if hasattr(self, 'up_total_progress'):
            self.up_total_var.set(up_total)
            self.up_total_progress['value'] = (up_total / max_total) * 100
        
        if hasattr(self, 'down_total_progress'):
            self.down_total_var.set(down_total)
            self.down_total_progress['value'] = (down_total / max_total) * 100
        
        # Update summary strings
        up = self.counts['up_lane']
        self.up_summary_var.set(f"Car: {up['Car']} | Motorcycle: {up['Motorcycle']} | Bus: {up['Bus']} | Truck: {up['Truck']}")
        
        down = self.counts['down_lane']
        self.down_summary_var.set(f"Car: {down['Car']} | Motorcycle: {down['Motorcycle']} | Bus: {down['Bus']} | Truck: {down['Truck']}")
        
        # Update grand total
        self.grand_total_var.set(str(grand_total))
        
    def playback_loop(self):
        """Main playback loop with enhanced tracking"""
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_skip_counter += 1
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.update_time_display(self.current_frame_num)
            
            # Run detection every N frames
            if self.frame_skip_counter >= self.process_every_n_frames:
                self.frame_skip_counter = 0
                
                detections = self.detector.detect(frame, imgsz=self.DET_IMG_SIZE, augment=False)
                
                filtered = []
                for det in detections:
                    vehicle_type, confidence = self.classify_vehicle(det)
                    if vehicle_type:
                        det['class_name'] = vehicle_type
                        det['confidence'] = confidence
                        filtered.append(det)
                
                # Apply motorcycle boost if enabled
                if self.var_motorcycle_boost.get():
                    filtered = self.apply_motorcycle_boost(frame, filtered)
                        
                self.last_detections = filtered
            else:
                filtered = self.last_detections
                
            # Update enhanced tracker
            tracks = self.enhanced_tracker.update(filtered, self.current_frame_num)
            
            # Process counting
            top_y = int(self.frame_h * self.top_line_y)
            bottom_y = int(self.frame_h * self.bottom_line_y)
            
            for track in tracks:
                tid = track.get('track_id')
                bbox = track.get('bbox')
                if not bbox or not tid:
                    continue
                    
                x1, y1, x2, y2 = bbox
                cy = (y1 + y2) / 2
                cls = track.get('class_name', '')
                
                if cls not in self.counts['up_lane']:
                    continue
                
                # Assign color for this track if not already
                if tid not in self.track_colors:
                    self.track_colors[tid] = tuple(np.random.randint(100, 255, 3).tolist())
                
                # Store path for visualization
                if self.var_show_paths.get():
                    self.track_paths[tid].append((int((x1+x2)/2), int((y1+y2)/2)))
                    if len(self.track_paths[tid]) > 30:
                        self.track_paths[tid].pop(0)
                
                # Check for line crossings
                if tid in self.track_last_y:
                    last_y = self.track_last_y[tid]
                    
                    # Up lane crossing (below to above)
                    if last_y > top_y and cy <= top_y:
                        if tid not in self.counted_tracks_top:
                            self.counts['up_lane'][cls] += 1
                            self.counted_tracks_top.add(tid)
                            self.status_var.set(f"⬆️ Up lane: {cls} counted")
                    
                    # Down lane crossing (above to below)
                    if last_y < bottom_y and cy >= bottom_y:
                        if tid not in self.counted_tracks_bottom:
                            self.counts['down_lane'][cls] += 1
                            self.counted_tracks_bottom.add(tid)
                            self.status_var.set(f"⬇️ Down lane: {cls} counted")
                
                self.track_last_y[tid] = cy
            
            # Update displays
            self.update_count_displays()
            
            # Draw on frame
            self.draw_on_frame(frame, tracks, top_y, bottom_y)
            
            # Display
            self.display_frame(frame)
            
            # Control speed for real-time playback
            time.sleep(0.03)  # ~30 FPS
            
        self.is_playing = False
        self.root.after(0, self.on_playback_end)
        
    def draw_on_frame(self, frame, tracks, top_y, bottom_y):
        """Enhanced drawing with paths and better visualization"""
        h, w = frame.shape[:2]
        
        # Draw lane lines
        # Up lane (blue)
        cv2.line(frame, (0, top_y), (w, top_y), (255, 168, 0), 3)
        cv2.putText(frame, "UP LANE", (10, top_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Down lane (green)
        cv2.line(frame, (0, bottom_y), (w, bottom_y), (0, 255, 0), 3)
        cv2.putText(frame, "DOWN LANE", (10, bottom_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw vehicle paths first (so they appear behind boxes)
        if self.var_show_paths.get():
            for tid, path in self.track_paths.items():
                if len(path) > 1:
                    color = self.track_colors.get(tid, (255, 255, 255))
                    for i in range(1, len(path)):
                        thickness = int(np.sqrt(i) + 1)
                        cv2.line(frame, path[i-1], path[i], color, thickness)
        
        # Draw vehicles
        for track in tracks:
            tid = track.get('track_id')
            bbox = track.get('bbox')
            cls = track.get('class_name', '')
            occluded = track.get('occluded', False)
            
            if bbox and cls in COLORS:
                x1, y1, x2, y2 = map(int, bbox)
                color = COLORS.get(cls, (255, 255, 255))
                
                # Draw bounding box (dashed if occluded)
                if occluded:
                    self.draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Determine lane
                cy = (y1 + y2) / 2
                lane_symbol = "⬆️" if cy < (top_y + bottom_y)/2 else "⬇️"
                
                # Label
                label = f"{cls[:3]}{tid}"
                if occluded:
                    label += "?"  # Mark occluded tracks
                
                # Text background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
                
                # Text
                cv2.putText(frame, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw live counts overlay
        y_offset = 40
        cv2.putText(frame, "LIVE COUNTS", (w-200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Up lane counts
        up = self.counts['up_lane']
        cv2.putText(frame, f"UP: C{up['Car']} M{up['Motorcycle']} B{up['Bus']} T{up['Truck']}", 
                   (w-200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 168, 0), 1)
        y_offset += 25
        
        # Down lane counts
        down = self.counts['down_lane']
        cv2.putText(frame, f"DOWN: C{down['Car']} M{down['Motorcycle']} B{down['Bus']} T{down['Truck']}", 
                   (w-200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        """Draw a dashed rectangle (for occluded objects)"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
            
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
        
    def display_frame(self, frame):
        """Display frame on canvas"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            h, w = frame.shape[:2]
            scale = min(canvas_w/w, canvas_h/h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if scale != 1:
                frame = cv2.resize(frame, (new_w, new_h))
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            
            x_offset = (canvas_w - new_w) // 2
            y_offset = (canvas_h - new_h) // 2
            
            self.canvas.delete('all')
            self.canvas.create_image(x_offset, y_offset, image=photo, anchor=NW)
            self.canvas.image = photo
            
    def on_playback_end(self):
        """Handle playback end"""
        self.play_btn.config(state=NORMAL)
        self.pause_btn.config(state=DISABLED)
        self.stop_btn.config(state=DISABLED)
        self.status_var.set("✅ Playback finished")
        
    def export_csv(self):
        """Export counts to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')],
            title='Save Counts'
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Lane', 'Direction', 'Vehicle Type', 'Count'])
                    
                    # Up lane
                    for vehicle, count in self.counts['up_lane'].items():
                        writer.writerow(['Up Lane', '⬆️', vehicle, count])
                    
                    # Down lane
                    for vehicle, count in self.counts['down_lane'].items():
                        writer.writerow(['Down Lane', '⬇️', vehicle, count])
                    
                    # Totals
                    up_total = sum(self.counts['up_lane'].values())
                    down_total = sum(self.counts['down_lane'].values())
                    
                    writer.writerow([])
                    writer.writerow(['SUMMARY', '', '', ''])
                    writer.writerow(['Up Lane Total', '⬆️', '', up_total])
                    writer.writerow(['Down Lane Total', '⬇️', '', down_total])
                    writer.writerow(['Grand Total', '📊', '', up_total + down_total])
                    
                self.status_var.set(f"✅ Saved to {Path(filename).name}")
                messagebox.showinfo("Success", "Counts exported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
                
    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == '__main__':
    app = ModernEnhancedVehicleCounter()
    app.run()
