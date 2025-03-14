#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AR Pi Tattoo - Main application module
"""

import cv2
import time
import numpy as np
import os

from .face_detector import FaceDetector
from .hand_gesture import HandGestureDetector
from .tattoo_renderer import TattooRenderer
from .utils import FPS


class ARPiTattooApp:
    """AR Pi Tattoo Application"""

    def __init__(self):
        """Initialize the application"""
        self.face_detector = FaceDetector()
        self.hand_detector = HandGestureDetector()
        self.tattoo_renderer = TattooRenderer()
        self.fps = FPS()
        self.cap = None
        
        # 调试模式
        self.debug = True
        
    def setup(self):
        """Setup the application"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
            
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 打印调试信息 - 检查资源目录
        if self.debug:
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
            pi_symbols_dir = os.path.join(assets_dir, "pi_symbols")
            print(f"Assets directory: {assets_dir}")
            print(f"Pi symbols directory: {pi_symbols_dir}")
            
            if os.path.exists(pi_symbols_dir):
                print(f"Pi symbols directory exists, contains files: {os.listdir(pi_symbols_dir)}")
                
                # 检查small目录
                small_dir = os.path.join(pi_symbols_dir, "small")
                if os.path.exists(small_dir):
                    print(f"Small directory exists, contains files: {os.listdir(small_dir)}")
                else:
                    print("Small directory does not exist")
                    
                # 检查digits目录
                digits_dir = os.path.join(pi_symbols_dir, "digits")
                if os.path.exists(digits_dir):
                    print(f"Digits directory exists, contains files: {os.listdir(digits_dir)}")
                else:
                    print("Digits directory does not exist")
            else:
                print("Pi symbols directory does not exist")
        
    def run(self):
        """Run the application"""
        if self.cap is None:
            raise ValueError("Camera not initialized. Call setup() first.")
            
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Update FPS counter
            self.fps.update()
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Display FPS
            fps_text = f"FPS: {self.fps.get_fps():.1f}"
            cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('AR Pi Tattoo', result_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        self.cleanup()
        
    def process_frame(self, frame):
        """Process a single frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with AR effects
        """
        # Detect face
        face_data = self.face_detector.detect_face(frame)
        
        # 调试信息 - 人脸检测
        if self.debug and face_data:
            # 显示人脸检测信息
            forehead_pos = face_data.get('forehead')
            mouth_open = face_data.get('mouth_open')
            
            debug_text = []
            if forehead_pos:
                debug_text.append(f"forehead position: {forehead_pos}")
                # 在额头位置画一个小圆点
                cv2.circle(frame, forehead_pos, 5, (0, 255, 0), -1)
            
            if mouth_open:
                debug_text.append("mouth: open")
            else:
                debug_text.append("mouth: closed")
                
            # 显示调试文本
            for i, text in enumerate(debug_text):
                cv2.putText(frame, text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detect hand gesture
        hand_data = self.hand_detector.detect(frame)
        
        # 调试信息 - 手势检测
        if self.debug and hand_data:
            gesture = hand_data.get('gesture')
            landmarks = hand_data.get('landmarks')
            
            debug_text = []
            if gesture:
                debug_text.append(f"gesture: {gesture}")
            
            if landmarks and 4 in landmarks:  # 检查拇指尖点
                thumb_tip = landmarks[4]
                thumb_x = int(thumb_tip['x'] * frame.shape[1])
                thumb_y = int(thumb_tip['y'] * frame.shape[0])
                # 在拇指尖画一个小圆点
                cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 0, 0), -1)
                debug_text.append(f"thumb position: ({thumb_x}, {thumb_y})")
            
            # 显示调试文本
            for i, text in enumerate(debug_text):
                cv2.putText(frame, text, (10, 150 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Render tattoo
        result = self.tattoo_renderer.render(frame, face_data, hand_data)
        
        # 调试信息 - 渲染状态
        if self.debug:
            # 显示加载的图像信息
            debug_text = []
            debug_text.append(f"Basic Pi symbol: {'loaded' if self.tattoo_renderer.pi_symbol is not None else 'not loaded'}")
            debug_text.append(f"Pi symbol style count: {len(self.tattoo_renderer.pi_symbols)}")
            debug_text.append(f"Small Pi symbol count: {len(self.tattoo_renderer.small_pi_symbols)}")
            debug_text.append(f"Pi digit image count: {len(self.tattoo_renderer.digit_images)}")
            
            # 显示动画状态
            heart_active = self.tattoo_renderer.animation_state['heart']['active']
            mouth_active = self.tattoo_renderer.animation_state['mouth']['active']
            
            debug_text.append(f"Heart animation: {'active' if heart_active else 'not active'}")
            debug_text.append(f"Mouth animation: {'active' if mouth_active else 'not active'}")
            
            if mouth_active:
                digit = self.tattoo_renderer.animation_state['mouth']['digit']
                debug_text.append(f"Current digit: {digit}")
            
            # 显示调试文本
            for i, text in enumerate(debug_text):
                cv2.putText(result, text, (frame.shape[1] - 350, 60 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return result
        
    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.face_detector.cleanup()
        self.hand_detector.cleanup() 