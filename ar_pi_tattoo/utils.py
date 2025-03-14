#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for AR Pi Tattoo
"""

import cv2
import numpy as np
import time


class FPS:
    """FPS counter for performance monitoring"""
    
    def __init__(self, avg_frames=30):
        """Initialize FPS counter
        
        Args:
            avg_frames: Number of frames to average FPS over
        """
        self.frame_times = []
        self.avg_frames = avg_frames
        self.prev_time = time.time()
        
    def update(self):
        """Update FPS counter with current frame time"""
        current_time = time.time()
        self.frame_times.append(current_time - self.prev_time)
        self.prev_time = current_time
        
        # Keep only the last avg_frames frame times
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
            
    def get_fps(self):
        """Get current FPS
        
        Returns:
            Current FPS as float
        """
        if not self.frame_times:
            return 0.0
            
        # Calculate average frame time
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # Calculate FPS
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return fps

def draw_fps(frame, fps):
    """Draw FPS counter on frame
    
    Args:
        frame: Input frame
        fps: FPS value to display
        
    Returns:
        Frame with FPS counter
    """
    # Draw FPS text
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame
    
def draw_instructions(frame, text):
    """Draw instructions on frame
    
    Args:
        frame: Input frame
        text: Instruction text to display
        
    Returns:
        Frame with instructions
    """
    # Draw instruction text at bottom of frame
    height = frame.shape[0]
    cv2.putText(frame, text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame 