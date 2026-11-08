"""
Simplified multi-object tracking module.

Uses centroid-based tracking with Hungarian algorithm for association.
This is a simpler alternative to ByteTrack that works better with Python 3.8.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List


class SimpleTracker:
    """Simplified multi-object tracker using centroid tracking."""
    
    def __init__(self, max_disappeared=30):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Max frames to keep track alive
        """
        self.next_track_id = 0
        self.tracks = {}  # {track_id: {centroid, bbox, frames_disappeared, ...}}
        self.max_disappeared = max_disappeared
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections with format:
                [{
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class': int,
                    'class_name': str
                }]
                
        Returns:
            List of tracked objects with format:
            [{
                'track_id': int,
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': int,
                'class_name': str
            }]
        """
        # Get current centroids from detections
        input_centroids = np.zeros((len(detections), 2))
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids[i] = [cx, cy]
        
        # If no tracks yet, create new ones
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                self.tracks[self.next_track_id] = {
                    'centroid': np.array([cx, cy]),
                    'bbox': det['bbox'],
                    'frames_disappeared': 0,
                    'confidence': det['confidence'],
                    'class': det['class'],
                    'class_name': det['class_name']
                }
                self.next_track_id += 1
        
        else:
            # Match detections to existing tracks
            tracked_tracks = list(self.tracks.keys())
            track_centroids = np.array([self.tracks[tid]['centroid'] for tid in tracked_tracks])
            
            # Initialize matched arrays
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)
            used_cols = set()
            
            # Calculate distances between centroids
            if len(input_centroids) > 0:
                distances = self._compute_distances(track_centroids, input_centroids)
                
                # Hungarian algorithm for optimal assignment
                row_idx, col_idx = linear_sum_assignment(distances)
                
                # Track matched detections
                for row, col in zip(row_idx, col_idx):
                    if distances[row, col] > 100:  # Max distance threshold (pixels)
                        continue
                    
                    track_id = tracked_tracks[row]
                    det = detections[col]
                    
                    # Update track
                    x1, y1, x2, y2 = det['bbox']
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    self.tracks[track_id]['centroid'] = np.array([cx, cy])
                    self.tracks[track_id]['bbox'] = det['bbox']
                    self.tracks[track_id]['frames_disappeared'] = 0
                    self.tracks[track_id]['confidence'] = det['confidence']
                    self.tracks[track_id]['class'] = det['class']
                    self.tracks[track_id]['class_name'] = det['class_name']
                    
                    used_cols.add(col)
                
                # Check for unmatched detections (new tracks)
                for col, det in enumerate(detections):
                    if col not in used_cols:
                        x1, y1, x2, y2 = det['bbox']
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        
                        self.tracks[self.next_track_id] = {
                            'centroid': np.array([cx, cy]),
                            'bbox': det['bbox'],
                            'frames_disappeared': 0,
                            'confidence': det['confidence'],
                            'class': det['class'],
                            'class_name': det['class_name']
                        }
                        self.next_track_id += 1
            
            # Mark unmatched tracks as disappeared
            matched_track_ids = set(tracked_tracks[row] for row, col in zip(row_idx, col_idx))
            for track_id in tracked_tracks:
                if track_id not in matched_track_ids:
                    self.tracks[track_id]['frames_disappeared'] += 1
        
        # Remove tracks that have disappeared too long
        deregistered_ids = []
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_disappeared'] > self.max_disappeared:
                deregistered_ids.append(track_id)
                del self.tracks[track_id]
        
        # Format output
        result = []
        for track_id in sorted(self.tracks.keys()):
            track = self.tracks[track_id]
            result.append({
                'track_id': track_id,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'class': track['class'],
                'class_name': track['class_name']
            })
        
        return result
    
    @staticmethod
    def _compute_distances(centroids_a: np.ndarray, centroids_b: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between two sets of centroids.
        
        Args:
            centroids_a: (N1, 2) array
            centroids_b: (N2, 2) array
            
        Returns:
            (N1, N2) distance matrix
        """
        # Expand dimensions for broadcasting
        ca = centroids_a[:, np.newaxis, :]  # (N1, 1, 2)
        cb = centroids_b[np.newaxis, :, :]  # (1, N2, 2)
        
        # Compute distances
        distances = np.sqrt(np.sum((ca - cb) ** 2, axis=2))  # (N1, N2)
        
        return distances
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 0


# Aliases for compatibility
ByteTracker = SimpleTracker

