#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hand gesture recognition module for AR Pi Tattoo
"""

import cv2
import mediapipe as mp
import numpy as np
import math


class HandGestureDetector:
    """Hand gesture recognition using MediaPipe Hands"""
    
    def __init__(self):
        """Initialize the hand gesture recognizer"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Last detected gesture
        self.last_gesture = None
        
        # Debug mode
        self.debug = True
        
    def detect(self, frame):
        """Detect hand gestures in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with gesture name and landmarks, or None
        """
        # Convert to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(rgb_frame)
        
        # If no hands detected, return None
        if not results.multi_hand_landmarks:
            return None
            
        # Get landmarks for the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 在原始帧上绘制手部关键点
        if self.debug:
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        # Recognize gesture
        gesture = self._recognize_gesture(hand_landmarks, frame.shape)
        
        # Update last detected gesture
        if gesture:
            self.last_gesture = gesture
            if self.debug:
                print(f"Detected gesture: {gesture}")
            
        # 将landmarks转换为字典格式
        landmarks_dict = {}
        height, width, _ = frame.shape
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmarks_dict[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
            
        # 返回包含手势和landmarks的字典
        return {
            'gesture': gesture,
            'landmarks': landmarks_dict
        }
        
    def _recognize_gesture(self, hand_landmarks, frame_shape):
        """Recognize specific gestures from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the input frame
            
        Returns:
            Gesture name or None
        """
        # Extract coordinates for all landmarks
        height, width, _ = frame_shape
        points = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            
        # Detect circle gesture
        if self._is_circle_gesture(points):
            return "circle"
            
        # Detect heart gesture (thumb and index finger crossed)
        if self._is_heart_gesture(points):
            return "heart"
            
        return None
        
    def _is_circle_gesture(self, points):
        """Check if the hand is forming a circle gesture
        
        Args:
            points: List of (x, y) coordinates for hand landmarks
            
        Returns:
            True if circle gesture detected, False otherwise
        """
        # Get coordinates for thumb tip and index finger tip
        thumb_tip = points[4]
        index_tip = points[8]
        
        # Calculate distance between thumb tip and index finger tip
        distance = math.sqrt(
            (thumb_tip[0] - index_tip[0])**2 + 
            (thumb_tip[1] - index_tip[1])**2
        )
        
        # Calculate hand size as reference (using distance from index base to pinky base)
        hand_size = math.sqrt(
            (points[5][0] - points[17][0])**2 + 
            (points[5][1] - points[17][1])**2
        )
        
        # If distance between thumb and index finger is less than 30% of hand size, consider it a circle gesture
        if distance < hand_size * 0.3:
            if self.debug:
                print(f"Detected circle gesture: distance={distance:.2f}, hand size={hand_size:.2f}, ratio={distance/hand_size:.2f}")
            return True
            
        return False
        
    def _is_heart_gesture(self, points):
        """Check if the hand is forming a heart gesture (thumb and index finger crossed)
        
        Args:
            points: List of (x, y) coordinates for hand landmarks
            
        Returns:
            True if heart gesture detected, False otherwise
        """
        # Get coordinates for thumb and index finger
        thumb_tip = points[4]
        thumb_ip = points[3]  # Interphalangeal joint
        thumb_mcp = points[2]  # Metacarpophalangeal joint
        
        index_tip = points[8]
        index_pip = points[6]  # Proximal interphalangeal joint
        index_mcp = points[5]  # Metacarpophalangeal joint
        
        # 检查拇指和食指是否形成心形
        # 简化的检测方法：检查拇指和食指是否交叉，并且指尖距离适中
        
        # 计算向量
        thumb_vector = (thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1])
        index_vector = (index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1])
        
        # 计算叉积，判断是否交叉
        cross_product = thumb_vector[0] * index_vector[1] - thumb_vector[1] * index_vector[0]
        
        # 计算指尖之间的距离
        tip_distance = math.sqrt(
            (thumb_tip[0] - index_tip[0])**2 + 
            (thumb_tip[1] - index_tip[1])**2
        )
        
        # 计算手部大小作为参考
        hand_size = math.sqrt(
            (points[0][0] - points[17][0])**2 + 
            (points[0][1] - points[17][1])**2
        )
        
        # 大幅降低检测阈值，使心形手势更容易被触发
        # 叉积的符号告诉我们它们是否交叉
        # 我们还检查指尖是否相对接近
        # 降低叉积阈值从500到300，增加距离阈值从0.6到0.8
        is_heart = abs(cross_product) > 300 and tip_distance < hand_size * 0.8
        
        if self.debug:
            print(f"Heart gesture detection: cross product={cross_product:.2f}, tip distance={tip_distance:.2f}, hand size={hand_size:.2f}, ratio={tip_distance/hand_size:.2f}, is heart={is_heart}")
            
            # 如果接近心形但未检测到，提供更多信息
            if tip_distance < hand_size * 0.9 and not is_heart:
                print(f"  Almost heart but not detected: cross product condition={abs(cross_product) > 300}, distance condition={tip_distance < hand_size * 0.8}")
        
        return is_heart
        
    def cleanup(self):
        """Release resources"""
        if self.hands:
            self.hands.close() 