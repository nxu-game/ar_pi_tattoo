#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Face detection module for AR Pi Tattoo
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Face detection using MediaPipe Face Mesh"""
    
    def __init__(self):
        """Initialize the face detector"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Forehead region keypoint indices (based on MediaPipe Face Mesh's 468 landmarks)
        # These points roughly outline the forehead area
        self.forehead_indices = [
            10, 67, 69, 104, 108, 151, 337, 338, 
            339, 340, 297, 299, 296, 334
        ]
        
        # Debug mode
        self.debug = True
        
    def setup(self):
        """Setup the face mesh detector"""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face(self, frame):
        """Detect face in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with face landmarks and forehead position, or None if no face detected
        """
        # Convert to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_frame)
        
        # If no face detected, return None
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # 在原始帧上绘制面部网格
        if self.debug:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        # Get forehead position (using the middle of the forehead)
        forehead_landmark = face_landmarks.landmark[10]  # Middle of forehead
        height, width, _ = frame.shape
        forehead_x = int(forehead_landmark.x * width)
        forehead_y = int(forehead_landmark.y * height)
        
        if self.debug:
            print(f"forehead position: ({forehead_x}, {forehead_y})")
        
        # Check if mouth is open
        mouth_open = self._is_mouth_open(face_landmarks, frame.shape)
        
        if self.debug:
            print(f"mouth state: {'open' if mouth_open else 'closed'}")
        
        # 将landmarks转换为字典格式
        landmarks_dict = {}
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks_dict[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
        
        # 获取嘴巴位置信息
        upper_lip = face_landmarks.landmark[13]  # Upper lip
        lower_lip = face_landmarks.landmark[14]  # Lower lip
        left_mouth = face_landmarks.landmark[78]  # Left corner of mouth
        right_mouth = face_landmarks.landmark[308]  # Right corner of mouth
        
        mouth_top = (int(upper_lip.x * width), int(upper_lip.y * height))
        mouth_bottom = (int(lower_lip.x * width), int(lower_lip.y * height))
        mouth_left = (int(left_mouth.x * width), int(left_mouth.y * height))
        mouth_right = (int(right_mouth.x * width), int(right_mouth.y * height))
        
        return {
            'landmarks': landmarks_dict,
            'forehead': (forehead_x, forehead_y),
            'mouth_open': mouth_open,
            'mouth_top': mouth_top,
            'mouth_bottom': mouth_bottom,
            'mouth_left': mouth_left,
            'mouth_right': mouth_right
        }
        
    def _is_mouth_open(self, face_landmarks, frame_shape):
        """Check if the mouth is open
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the input frame
            
        Returns:
            True if mouth is open, False otherwise
        """
        height, width, _ = frame_shape
        
        # Get upper and lower lip landmarks
        # Using landmarks for the middle of the upper and lower lips
        upper_lip = face_landmarks.landmark[13]  # Upper lip
        lower_lip = face_landmarks.landmark[14]  # Lower lip
        
        # Convert to pixel coordinates
        upper_lip_y = int(upper_lip.y * height)
        lower_lip_y = int(lower_lip.y * height)
        
        # Calculate vertical distance between lips
        lip_distance = abs(upper_lip_y - lower_lip_y)
        
        # Calculate face height as reference (using distance from chin to forehead)
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]
        face_height = abs(int(forehead.y * height) - int(chin.y * height))
        
        # 计算嘴巴张开比例
        mouth_ratio = lip_distance / face_height
        
        if self.debug:
            print(f"mouth detection: upper_lip_y={upper_lip_y}, lower_lip_y={lower_lip_y}, distance={lip_distance}, face_height={face_height}, ratio={mouth_ratio:.3f}")
        
        # If distance between lips is more than 5% of face height, consider mouth open
        return lip_distance > (face_height * 0.05)
        
    def get_forehead_rect(self, landmarks, frame_shape):
        """Get a rectangular region covering the forehead
        
        Args:
            landmarks: Face landmarks from detect()
            frame_shape: Shape of the input frame
            
        Returns:
            Tuple of (x, y, w, h) defining the forehead rectangle
        """
        if landmarks is None or 'forehead_points' not in landmarks:
            return None
            
        forehead_points = landmarks['forehead_points']
        
        # Calculate bounding box for forehead region
        x, y, w, h = cv2.boundingRect(forehead_points)
        
        # Slightly expand the area to ensure it covers the entire forehead
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(frame_shape[1] - x, w + 2 * padding_x)
        h = min(frame_shape[0] - y, h + 2 * padding_y)
        
        return (x, y, w, h)
        
    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()