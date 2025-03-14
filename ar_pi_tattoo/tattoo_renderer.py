#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tattoo rendering module for AR Pi Tattoo
"""

import os
import cv2
import numpy as np
import math
from pathlib import Path
import random
import time


class TattooRenderer:
    """Renderer for AR tattoo effects"""
    
    def __init__(self, assets_dir='assets'):
        """Initialize the tattoo renderer
        
        Args:
            assets_dir: Directory containing assets
        """
        # 始终使用项目根目录下的assets目录
        module_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(os.path.dirname(module_dir), assets_dir)
        
        self.assets_dir = assets_dir
        self.pi_symbol = None
        self.pi_symbols = []
        self.small_pi_symbols = []
        self.digit_images = {}
        self.animation_state = {
            'heart': {
                'active': False,
                'start_time': 0,
                'duration': 1.0,  # 减少动画持续时间，加快显示速度
                'current_frame': 0,
                'selected_symbol': None,  # 存储选中的符号
                'rotation_angle': 0,      # 旋转角度
                'scale_factor': 1.0,      # 缩放因子
                'position_offset': (0, 0) # 位置偏移
            },
            'mouth': {
                'active': False,
                'start_time': 0,
                'duration': 10.0,  # 增加嘴部动画持续时间到10秒
                'digit': None,
                'frame_count': 0,
                'max_frames': 200,  # 增加最大帧数到200
                'digit_index': 0,   # 跟踪当前显示的π数字索引
                'scale_factor': 1.0,      # 缩放因子
                'rotation_angle': 0,      # 旋转角度
                'position_offset': (0, 0), # 位置偏移
                'active_digits': []  # 存储当前活跃的数字及其状态
            }
        }
        
        # π的数字序列 - 增加到100位
        self.pi_digits = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
        
        # 调试模式
        self.debug = True
        
        # Load pi symbols
        self._load_pi_symbols()
        
    def _load_pi_symbols(self):
        """Load pi symbols from assets directory"""
        self.pi_symbols = []
        self.small_pi_symbols = []
        self.digit_images = {}
        
        # 加载主要π符号
        pi_symbols_dir = os.path.join(self.assets_dir, "pi_symbols")
        if os.path.exists(pi_symbols_dir):
            for filename in os.listdir(pi_symbols_dir):
                if filename.startswith("pi_symbol_") and filename.endswith((".png", ".jpg")):
                    filepath = os.path.join(pi_symbols_dir, filename)
                    try:
                        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            # 确保有alpha通道
                            if img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            self.pi_symbols.append(img)
                            if self.debug:
                                print(f"已加载π符号: {filepath}")
                    except Exception as e:
                        print(f"无法加载π符号 {filepath}: {e}")
        
        # 加载小π符号（用于动画）
        small_pi_dir = os.path.join(pi_symbols_dir, "small")
        if os.path.exists(small_pi_dir):
            for filename in sorted(os.listdir(small_pi_dir)):
                if filename.startswith("small_pi_") and filename.endswith((".png", ".jpg")):
                    filepath = os.path.join(small_pi_dir, filename)
                    try:
                        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            # 确保有alpha通道
                            if img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            self.small_pi_symbols.append(img)
                            if self.debug:
                                print(f"已加载小π符号: {filepath}")
                    except Exception as e:
                        print(f"无法加载小π符号 {filepath}: {e}")
        
        # 加载数字图像（用于嘴部动画）
        digits_dir = os.path.join(pi_symbols_dir, "digits")
        if os.path.exists(digits_dir):
            for filename in os.listdir(digits_dir):
                # 支持数字和小数点图像
                if filename.startswith("pi_digit_") and filename.endswith((".png", ".jpg")):
                    try:
                        # 从文件名中提取数字
                        digit_str = filename.replace("pi_digit_", "").split(".")[0]
                        # 处理小数点的特殊情况
                        if digit_str == "dot":
                            digit_key = "."
                        else:
                            digit_key = digit_str
                            
                        filepath = os.path.join(digits_dir, filename)
                        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            # 确保有alpha通道
                            if img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            self.digit_images[digit_key] = img
                            if self.debug:
                                print(f"已加载数字图像 {digit_key}: {filepath}")
                    except Exception as e:
                        print(f"无法加载数字图像 {filename}: {e}")
        
        # 如果没有加载到任何π符号，创建一个基本的
        if not self.pi_symbols:
            print("未找到π符号图像，创建基本符号")
            self.pi_symbols.append(self._create_pi_symbol(200))
        
        # 如果没有加载到任何小π符号，创建一些基本的
        if not self.small_pi_symbols:
            print("未找到小π符号图像，创建基本符号")
            for _ in range(5):
                self.small_pi_symbols.append(self._create_pi_symbol(100))
        
        # 如果没有加载到任何数字图像，创建一些基本的
        if not self.digit_images:
            print("未找到数字图像，创建基本数字")
            for digit in "0123456789.":
                self.digit_images[digit] = self._create_digit_image(digit, 100)
        
        if self.debug:
            print(f"已加载 {len(self.pi_symbols)} 个π符号")
            print(f"已加载 {len(self.small_pi_symbols)} 个小π符号")
            print(f"已加载 {len(self.digit_images)} 个数字图像: {list(self.digit_images.keys())}")
        
    def _create_pi_symbol(self, size):
        """Create a basic pi symbol image
        
        Args:
            size: Size of the output image
            
        Returns:
            BGRA image with pi symbol
        """
        # Create a transparent image
        img = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Draw pi symbol
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size / 80
        thickness = int(size / 40)
        text = 'π'
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position to center the text
        x = (size - text_width) // 2
        y = (size + text_height) // 2
        
        # Draw white text
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255, 255), thickness)
        
        return img
        
    def render(self, frame, face_data=None, hand_data=None):
        """Render AR effects on the frame
        
        Args:
            frame: Input video frame
            face_data: Face detection data
            hand_data: Hand gesture data
            
        Returns:
            Frame with AR effects rendered
        """
        # 创建一个副本，避免修改原始帧
        result = frame.copy()
        
        # 如果检测到人脸，在额头上渲染π符号
        if face_data and 'forehead' in face_data:
            forehead_pos = face_data['forehead']
            self._render_on_forehead(result, forehead_pos)
            
        # 如果检测到心形手势，渲染动画
        if hand_data and hand_data.get('gesture') == 'heart':
            # 如果动画未激活，开始动画
            if not self.animation_state['heart']['active']:
                self.animation_state['heart']['active'] = True
                self.animation_state['heart']['start_time'] = time.time()
                self.animation_state['heart']['current_frame'] = 0
                if self.debug:
                    print("Heart gesture detected, starting animation")
                    
        # 如果心形动画正在进行，渲染它
        if self.animation_state['heart']['active']:
            self._render_heart_animation(result, hand_data)
            
        # 如果检测到嘴巴张开，渲染动画
        if face_data and face_data.get('mouth_open'):
            # 如果动画未激活，开始动画
            if not self.animation_state['mouth']['active']:
                self.animation_state['mouth']['active'] = True
                self.animation_state['mouth']['start_time'] = time.time()
                self.animation_state['mouth']['frame_count'] = 0
                self.animation_state['mouth']['digit'] = None
                if self.debug:
                    print("Mouth open, starting mouth animation")
                    
        # 如果嘴巴动画正在进行，渲染它
        if self.animation_state['mouth']['active'] and face_data:
            # 确保face_data包含必要的嘴巴位置信息
            if all(key in face_data for key in ['mouth_top', 'mouth_bottom', 'mouth_left', 'mouth_right']):
                self._render_mouth_animation(result, face_data)
            else:
                # 如果缺少必要的嘴巴位置信息，停止动画
                if self.debug:
                    print("Missing mouth position data, stopping mouth animation")
                self.animation_state['mouth']['active'] = False
            
        return result
        
    def _render_on_forehead(self, frame, forehead_pos):
        """Render pi symbol on forehead
        
        Args:
            frame: Frame to render on
            forehead_pos: Position of the forehead (x, y)
        """
        if self.debug:
            print(f"Rendering forehead pi symbol, position: {forehead_pos}")
            
        # 如果没有π符号图像，无法渲染
        if not self.pi_symbols:
            if self.debug:
                print("Unable to render forehead pi symbol: no pi symbol images available")
            return
            
        # 随机选择一个π符号
        pi_symbol = random.choice(self.pi_symbols)
        
        # 确保pi_symbol不为None
        if pi_symbol is None:
            if self.debug:
                print("Unable to render forehead pi symbol: selected pi symbol is None")
            return
            
        # 调整π符号大小
        symbol_size = 100  # 固定大小
        pi_symbol_resized = self._resize_image(pi_symbol, symbol_size, symbol_size)
        
        # 确保调整大小成功
        if pi_symbol_resized is None:
            if self.debug:
                print("Unable to render forehead pi symbol: failed to resize image")
            return
            
        # 计算位置（居中于额头）
        pos_x = forehead_pos[0] - pi_symbol_resized.shape[1] // 2
        pos_y = forehead_pos[1] - pi_symbol_resized.shape[0] // 2
        
        # 叠加图像
        self._overlay_image(frame, pi_symbol_resized, pos_x, pos_y)
        
    def _render_heart_animation(self, frame, hand_data):
        """Render animation when heart gesture is detected
        
        Args:
            frame: Frame to render on
            hand_data: Hand gesture data
        """
        # 检查动画状态
        animation = self.animation_state['heart']
        
        # 如果动画正在进行，但没有手部数据，仍然继续动画
        if animation['active']:
            # 计算动画进度
            elapsed_time = time.time() - animation['start_time']
            progress = min(elapsed_time / animation['duration'], 1.0)
            
            # 如果动画结束，重置状态
            if progress >= 1.0:
                animation['active'] = False
                animation['selected_symbol'] = None
                if self.debug:
                    print("Heart animation ended")
                return
                
            # 选择一个π符号图像
            if not self.small_pi_symbols:
                if self.debug:
                    print("Unable to render heart animation: no small pi symbols available")
                return
                
            # 如果还没有选择符号，随机选择一个
            if animation['selected_symbol'] is None:
                animation['selected_symbol'] = random.choice(self.small_pi_symbols)
                # 随机初始化变换参数
                animation['rotation_angle'] = random.uniform(0, 360)
                animation['scale_factor'] = random.uniform(0.8, 1.2)
                animation['position_offset'] = (
                    random.randint(-50, 50),
                    random.randint(-50, 50)
                )
                
            pi_symbol = animation['selected_symbol']
            
            # 计算动画参数
            # 使用非线性缓动函数使动画更加生动
            # 使用二次缓动函数：progress^2 用于加速开始，(1-(1-progress)^2) 用于减速结束
            if progress < 0.5:
                # 加速阶段
                ease = 2 * progress * progress
            else:
                # 减速阶段
                ease = 1 - pow(-2 * progress + 2, 2) / 2
                
            # 计算大小变化：从小到大
            start_size = 20
            end_size = 150
            current_size = int(start_size + (end_size - start_size) * ease)
            
            # 如果有手部数据，使用拇指位置作为起点
            if hand_data and 'landmarks' in hand_data and 4 in hand_data['landmarks']:
                thumb_tip = hand_data['landmarks'][4]
                thumb_x = int(thumb_tip['x'] * frame.shape[1])
                thumb_y = int(thumb_tip['y'] * frame.shape[0])
            else:
                # 如果没有手部数据，使用屏幕中心下方作为起点
                thumb_x = frame.shape[1] // 2
                thumb_y = int(frame.shape[0] * 0.7)  # 屏幕70%高度处
            
            # 计算位置：从拇指位置向上移动，加上随机偏移
            offset_y = int(100 * ease)  # 向上移动的距离
            pos_x = thumb_x - current_size // 2 + int(animation['position_offset'][0] * ease)
            pos_y = thumb_y - current_size - offset_y + int(animation['position_offset'][1] * ease)
            
            # 计算透明度：从透明到不透明
            alpha = min(1.0, progress * 3)  # 更快地变为不透明
            
            # 调整图像大小，考虑缩放因子
            scaled_size = int(current_size * animation['scale_factor'])
            pi_symbol_resized = self._resize_image(pi_symbol, scaled_size, scaled_size)
            
            # 应用旋转变换
            # 计算当前旋转角度，随时间变化
            current_angle = animation['rotation_angle'] + (progress * 360 % 360)
            
            # 获取旋转矩阵
            center = (pi_symbol_resized.shape[1] // 2, pi_symbol_resized.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            
            # 应用旋转
            rotated_image = cv2.warpAffine(
                pi_symbol_resized, 
                rotation_matrix, 
                (pi_symbol_resized.shape[1], pi_symbol_resized.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # 应用透明度
            if alpha < 1.0:
                # 创建透明度蒙版
                alpha_mask = np.ones(rotated_image.shape[:2], dtype=np.float32) * alpha
                # 应用透明度
                rotated_image[:,:,3] = (rotated_image[:,:,3] * alpha_mask).astype(np.uint8)
            
            # 将图像叠加到帧上
            self._overlay_image(frame, rotated_image, pos_x, pos_y)
            
            # 更新当前帧
            animation['current_frame'] += 1
            return
            
        # 如果动画未激活，检查是否需要开始新动画
        if hand_data and hand_data.get('gesture') == 'heart':
            # 获取拇指位置
            landmarks = hand_data.get('landmarks', {})
            if not landmarks or 4 not in landmarks:
                return
                
            # 开始新动画
            animation['active'] = True
            animation['start_time'] = time.time()
            animation['current_frame'] = 0
            animation['selected_symbol'] = None  # 将在第一帧选择
            if self.debug:
                print("Heart gesture detected, starting animation")
        
    def _render_mouth_animation(self, frame, face_data):
        """Render pi digits animation when mouth is open
        
        Args:
            frame: Frame to render on
            face_data: Face detection data
        """
        if not face_data:
            return
            
        # 获取嘴巴位置
        mouth_top = face_data.get('mouth_top')
        mouth_bottom = face_data.get('mouth_bottom')
        mouth_left = face_data.get('mouth_left')
        mouth_right = face_data.get('mouth_right')
        
        if not all([mouth_top, mouth_bottom, mouth_left, mouth_right]):
            if self.debug:
                print("Unable to render mouth animation: missing mouth position info")
            return
            
        # 计算嘴部中心和大小
        mouth_center = (
            int((mouth_left[0] + mouth_right[0]) / 2),
            int((mouth_top[1] + mouth_bottom[1]) / 2)
        )
        mouth_width = int(mouth_right[0] - mouth_left[0])
        mouth_height = int(mouth_bottom[1] - mouth_top[1])
        
        # 确保宽度和高度不为零
        if mouth_width <= 0 or mouth_height <= 0:
            if self.debug:
                print("Unable to render mouth animation: invalid mouth dimensions")
            return
            
        # 检查动画状态
        animation = self.animation_state['mouth']
        
        # 如果检测到嘴巴张开，开始动画
        if face_data.get('mouth_open') and not animation['active']:
            animation['active'] = True
            animation['start_time'] = time.time()
            animation['frame_count'] = 0
            animation['digit_index'] = 0  # 重置数字索引，从3.14开始
            animation['active_digits'] = []  # 清空活跃数字列表
            if self.debug:
                print("Mouth open, starting mouth animation")
        
        # 如果嘴巴关闭，停止动画并重置索引
        if not face_data.get('mouth_open') and animation['active']:
            animation['active'] = False
            animation['digit_index'] = 0  # 重置数字索引，下次从3.14开始
            animation['active_digits'] = []  # 清空活跃数字列表
            if self.debug:
                print("Mouth closed, stopping animation")
            return
        
        # 如果动画正在进行
        if animation['active']:
            if self.debug and animation['frame_count'] % 10 == 0:  # 减少日志输出频率
                print(f"Rendering mouth animation, frame: {animation['frame_count']}, active digits: {len(animation['active_digits'])}")
                
            # 如果已经显示了足够的帧数，结束动画
            if animation['frame_count'] >= animation['max_frames']:
                animation['active'] = False
                animation['digit_index'] = 0  # 重置数字索引
                animation['active_digits'] = []  # 清空活跃数字列表
                if self.debug:
                    print("Mouth animation ended")
                return
            
            # 每5帧添加一个新数字
            if animation['frame_count'] % 5 == 0:
                try:
                    # 获取下一个π数字
                    digit_index = animation['digit_index']
                    digit = self.pi_digits[digit_index]
                    
                    # 确保digit是字符串类型
                    digit = str(digit)
                    
                    # 更新索引，循环使用π数字序列
                    animation['digit_index'] = (digit_index + 1) % len(self.pi_digits)
                    
                    # 检查是否有这个数字的图像
                    digit_img = None
                    if digit in self.digit_images:
                        digit_img = self.digit_images[digit]
                    else:
                        # 如果没有特定数字的图像，使用随机可用数字
                        available_digits = list(self.digit_images.keys())
                        if available_digits:
                            random_digit = random.choice(available_digits)
                            digit_img = self.digit_images[random_digit]
                            # 更新当前数字为实际使用的数字
                            digit = random_digit
                            if self.debug:
                                print(f"  Digit {digit} image not found, using alternative digit {random_digit}")
                        else:
                            if self.debug:
                                print("  No digit images available")
                            return
                    
                    # 为新数字创建随机变换参数
                    new_digit = {
                        'digit': digit,
                        'start_frame': animation['frame_count'],
                        'lifetime': random.randint(30, 60),  # 每个数字显示30-60帧
                        'rotation_angle': random.uniform(0, 360),
                        'scale_factor': random.uniform(0.8, 1.8),
                        'position_offset': (
                            random.randint(-int(mouth_width * 0.8), int(mouth_width * 0.8)),
                            random.randint(-int(mouth_height * 0.8), int(mouth_height * 0.8))
                        ),
                        'direction': (
                            random.uniform(-1, 1),
                            random.uniform(-1, 1)
                        ),
                        'speed': random.uniform(1, 3)
                    }
                    
                    # 添加到活跃数字列表
                    animation['active_digits'].append(new_digit)
                    
                    if self.debug and animation['frame_count'] % 10 == 0:
                        print(f"  Added new digit: {digit}, total active: {len(animation['active_digits'])}")
                except Exception as e:
                    if self.debug:
                        print(f"Error adding new digit: {e}")
            
            # 更新和渲染所有活跃数字
            updated_active_digits = []
            for digit_info in animation['active_digits']:
                # 计算数字已经显示的帧数
                frames_active = animation['frame_count'] - digit_info['start_frame']
                
                # 如果数字已经显示足够长时间，不再继续显示
                if frames_active >= digit_info['lifetime']:
                    continue
                
                # 计算显示进度 (0.0 到 1.0)
                progress = frames_active / digit_info['lifetime']
                
                # 使用缓动函数使动画更加生动
                if progress < 0.2:
                    # 快速出现
                    ease = progress / 0.2
                elif progress > 0.8:
                    # 慢慢消失
                    ease = (1.0 - progress) / 0.2
                else:
                    # 保持最大尺寸
                    ease = 1.0
                
                # 获取数字图像
                try:
                    digit = digit_info['digit']
                    # 确保digit是字符串类型
                    digit = str(digit)
                    
                    digit_img = None
                    if digit in self.digit_images:
                        digit_img = self.digit_images[digit]
                    else:
                        continue
                        
                    if digit_img is None:
                        continue
                    
                    # 计算大小变化：从小到大再到小，加入随机缩放因子
                    base_size = max(mouth_width, mouth_height) * 1.5  # 基础大小，更大于嘴部
                    current_size = int(base_size * ease * digit_info['scale_factor'])
                    
                    # 确保尺寸至少为10像素
                    current_size = max(current_size, 10)
                    
                    # 更新位置：数字会随时间移动
                    # 基础位置是嘴部中心
                    base_x = mouth_center[0] + digit_info['position_offset'][0]
                    base_y = mouth_center[1] + digit_info['position_offset'][1]
                    
                    # 根据方向和速度更新位置
                    move_x = digit_info['direction'][0] * digit_info['speed'] * frames_active
                    move_y = digit_info['direction'][1] * digit_info['speed'] * frames_active
                    
                    pos_x = int(base_x + move_x)
                    pos_y = int(base_y + move_y)
                    
                    # 调整数字大小
                    digit_img_resized = self._resize_image(digit_img, current_size, current_size)
                    
                    # 应用旋转变换
                    if digit_img_resized is None:
                        continue
                        
                    center = (digit_img_resized.shape[1] // 2, digit_img_resized.shape[0] // 2)
                    # 随时间增加旋转角度
                    current_angle = digit_info['rotation_angle'] + (frames_active * 2 % 360)
                    rotation_matrix = cv2.getRotationMatrix2D(center, current_angle, 1.0)
                    
                    # 应用旋转
                    if len(digit_img_resized.shape) < 3 or digit_img_resized.shape[2] < 4:
                        if self.debug:
                            print(f"Invalid digit image shape: {digit_img_resized.shape}")
                        continue
                        
                    # 分离通道
                    b, g, r, a = cv2.split(digit_img_resized)
                    
                    # 旋转RGB通道
                    rgb = cv2.merge([b, g, r])
                    rgb_rotated = cv2.warpAffine(
                        rgb, 
                        rotation_matrix, 
                        (digit_img_resized.shape[1], digit_img_resized.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT
                    )
                    
                    # 旋转Alpha通道
                    a_rotated = cv2.warpAffine(
                        a, 
                        rotation_matrix, 
                        (digit_img_resized.shape[1], digit_img_resized.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT
                    )
                    
                    # 合并通道
                    b_rotated, g_rotated, r_rotated = cv2.split(rgb_rotated)
                    rotated_image = cv2.merge([b_rotated, g_rotated, r_rotated, a_rotated])
                except Exception as e:
                    if self.debug:
                        print(f"Error processing digit: {e}")
                    continue
                
                # 将数字放置在计算的位置
                self._overlay_image(
                    frame, 
                    rotated_image, 
                    pos_x - rotated_image.shape[1] // 2,
                    pos_y - rotated_image.shape[0] // 2
                )
                
                # 保留这个数字用于下一帧
                updated_active_digits.append(digit_info)
            
            # 更新活跃数字列表
            animation['active_digits'] = updated_active_digits
            
            # 更新帧计数
            animation['frame_count'] += 1
        
    def _blend_image(self, background, foreground, position):
        """Blend foreground image with alpha channel onto background
        
        Args:
            background: Background image
            foreground: Foreground image with alpha channel
            position: Position (x, y) to place the foreground
            
        Returns:
            Blended image
        """
        x, y = position
        h, w = foreground.shape[:2]
        
        # Check if the foreground is completely outside the background
        if x >= background.shape[1] or y >= background.shape[0] or x + w <= 0 or y + h <= 0:
            if self.debug:
                print(f"Image blending: foreground completely outside background, position=({x}, {y}), size=({w}, {h}), background size={background.shape}")
            return background
            
        # Calculate the valid region to blend
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(background.shape[1], x + w)
        y2 = min(background.shape[0], y + h)
        
        # Calculate corresponding region in foreground
        fx1 = x1 - x
        fy1 = y1 - y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)
        
        # Get regions from both images
        bg_region = background[y1:y2, x1:x2]
        fg_region = foreground[fy1:fy2, fx1:fx2]
        
        # Extract alpha channel and normalize to range 0-1
        alpha = fg_region[:, :, 3] / 255.0
        
        # Reshape alpha for broadcasting
        alpha = alpha[:, :, np.newaxis]
        
        # Blend images
        blended = bg_region * (1 - alpha) + fg_region[:, :, :3] * alpha
        
        # Update background with blended region
        result = background.copy()
        result[y1:y2, x1:x2] = blended
        
        return result
        
    def _render_basic(self, frame, position, size=100):
        """Render a basic pi symbol
        
        Args:
            frame: Input frame
            position: Position (x, y) to render the symbol
            size: Size of the symbol
            
        Returns:
            Frame with pi symbol rendered
        """
        # Create pi symbol if not already created
        if self.pi_symbol is None:
            self.pi_symbol = self._create_pi_symbol(size)
            
        # Resize pi symbol if needed
        if self.pi_symbol.shape[0] != size:
            pi_symbol = cv2.resize(self.pi_symbol, (size, size))
        else:
            pi_symbol = self.pi_symbol
            
        # Calculate position to center the symbol
        x, y = position
        x = x - size // 2
        y = y - size // 2
        
        # Blend the pi symbol onto the frame
        return self._blend_image(frame, pi_symbol, (x, y))
        
    def cleanup(self):
        """Release resources"""
        pass  # No resources to clean up

    def _create_digit_image(self, digit, size):
        """Create a basic digit image
        
        Args:
            digit: The digit to create (0-9 or '.')
            size: Size of the image
            
        Returns:
            Image with the digit
        """
        # 创建透明背景
        img = np.zeros((size, size, 4), dtype=np.uint8)
        
        # 设置字体
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size / 100 * 2.5  # 较大的字体
        thickness = int(size / 30)
        color = (255, 255, 255, 255)  # 白色，带透明度
        
        # 获取文本大小
        text = str(digit)
        text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        
        # 计算文本位置（居中）
        x = (size - text_size[0]) // 2
        y = (size + text_size[1]) // 2
        
        # 绘制文本
        cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness)
        
        # 添加发光效果
        blur_size = int(size / 10)
        if blur_size % 2 == 0:
            blur_size += 1  # 确保是奇数
        img_blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # 合并原始图像和发光效果
        result = np.zeros_like(img)
        for c in range(3):  # RGB通道
            result[:,:,c] = np.minimum(img[:,:,c] + img_blurred[:,:,c], 255)
        result[:,:,3] = img[:,:,3]  # 保持原始alpha通道
        
        return result
        
    def _resize_image(self, image, width, height):
        """Resize image while preserving aspect ratio
        
        Args:
            image: Input image
            width: Target width
            height: Target height
            
        Returns:
            Resized image
        """
        if image is None:
            return None
            
        # 检查图像是否有正确的形状和通道
        if len(image.shape) < 3:
            if self.debug:
                print(f"Invalid image shape: {image.shape}, must have at least 3 dimensions")
            return None
            
        # 计算宽高比
        aspect_ratio = image.shape[1] / image.shape[0]
        
        # 根据目标宽高比调整大小
        if width / height > aspect_ratio:
            # 目标更宽，以高度为基准
            new_height = height
            new_width = int(height * aspect_ratio)
        else:
            # 目标更高，以宽度为基准
            new_width = width
            new_height = int(width / aspect_ratio)
            
        # 调整大小
        try:
            resized = cv2.resize(image, (new_width, new_height))
            # 确保图像有alpha通道
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            return resized
        except Exception as e:
            if self.debug:
                print(f"Error resizing image: {e}")
            return None
        
    def _overlay_image(self, background, overlay, x, y):
        """Overlay an image with alpha channel onto background
        
        Args:
            background: Background image
            overlay: Overlay image with alpha channel
            x, y: Position to place the overlay
            
        Returns:
            Background with overlay applied
        """
        if overlay is None:
            return
            
        # 检查overlay是否有正确的形状和通道
        if len(overlay.shape) < 3 or overlay.shape[2] < 4:
            if self.debug:
                print(f"Invalid overlay image shape: {overlay.shape}, must have alpha channel")
            return
            
        # 获取覆盖图像的尺寸
        h, w = overlay.shape[:2]
        
        # 检查位置是否在背景图像范围内
        if x < 0:
            overlay = overlay[:, -x:]
            w += x
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h += y
            y = 0
        if x + w > background.shape[1]:
            overlay = overlay[:, :background.shape[1] - x]
            w = overlay.shape[1]
        if y + h > background.shape[0]:
            overlay = overlay[:background.shape[0] - y, :]
            h = overlay.shape[0]
            
        if w <= 0 or h <= 0:
            return
            
        # 获取覆盖区域
        roi = background[y:y+h, x:x+w]
        
        # 创建蒙版
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # 应用覆盖图像
        for c in range(3):  # RGB通道
            background[y:y+h, x:x+w, c] = (overlay[:, :, c] * alpha[:, :, 0] + 
                                          roi[:, :, c] * (1 - alpha[:, :, 0]))
                                          
        return background 