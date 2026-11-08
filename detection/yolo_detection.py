"""
YOLO-based vehicle detection module.

Detects vehicles (cars, trucks, motorcycles) in video frames.
"""

from ultralytics import YOLO
import numpy as np


class YOLODetector:
    """YOLO detector for vehicle detection in traffic scenes."""
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (yolov8n, yolov8s, yolov8m, etc.)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = [2, 3, 5, 7]  # car, truck, motorcycle, bus
        
    def detect(self, frame, imgsz=None, augment=False):
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format:
            [{
                'bbox': [x1, y1, x2, y2],  # xyxy format
                'confidence': float,
                'class': int,
                'class_name': str
            }]
        """
        # Allow overriding inference image size and enable augmentation (TTA)
        kwargs = {}
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        if augment:
            kwargs['augment'] = True

        results = self.model(frame, conf=self.conf_threshold, verbose=False, **kwargs)
        detections = []
        
        for r in results[0].boxes:
            cls = int(r.cls.cpu().numpy()[0])
            
            # Only keep vehicle classes
            if cls in self.vehicle_classes:
                bbox = r.xyxy.cpu().numpy().tolist()[0]
                confidence = float(r.conf.cpu().numpy()[0])
                class_name = self.model.names[cls]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class': cls,
                    'class_name': class_name
                })
        
        return detections
    
    def filter_detections(self, detections, min_conf=0.5, min_area=100):
        """
        Filter detections by confidence and bounding box area.
        
        Args:
            detections: List of detections
            min_conf: Minimum confidence threshold
            min_area: Minimum bounding box area (pixels²)
            
        Returns:
            Filtered detections list
        """
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            if det['confidence'] >= min_conf and area >= min_area:
                filtered.append(det)
        
        return filtered
