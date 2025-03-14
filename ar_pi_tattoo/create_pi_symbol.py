import cv2
import numpy as np
import os
import sys

def create_basic_pi_symbol(size=200, output_path=None):
    """Create a basic pi symbol image
    
    Args:
        size: Size of the output image
        output_path: Path to save the image, if None, the image is not saved
        
    Returns:
        The created image
    """
    # 创建透明背景
    img = np.zeros((size, size, 4), dtype=np.uint8)
    
    # 设置字体
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size / 80  # 增大字体
    thickness = int(size / 40)  # 增加线条粗细
    color = (255, 255, 255, 255)  # 白色，带透明度
    
    # 绘制π符号
    text = "π"
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    # 计算文本位置（居中）
    x = (size - text_size[0]) // 2
    y = (size + text_size[1]) // 2
    
    # 绘制文本
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness)
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"已保存基本π符号图像到 {output_path}")
    
    return img

def create_stylized_pi_symbol(size=200, output_path=None):
    """Create a stylized pi symbol with edge enhancement
    
    Args:
        size: Size of the output image
        output_path: Path to save the image, if None, the image is not saved
        
    Returns:
        The created image
    """
    # 首先创建基本符号
    img = create_basic_pi_symbol(size)
    
    # 应用边缘增强
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    img_filtered = cv2.filter2D(img, -1, kernel)
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_filtered)
        print(f"已保存风格化π符号图像到 {output_path}")
    
    return img_filtered

def extract_pi_from_image(image_path, output_path=None, size=200):
    """Extract pi symbol from an image
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the extracted image, if None, the image is not saved
        size: Size of the output image
        
    Returns:
        The extracted image or None if extraction failed
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
            
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"未在图像中找到轮廓: {image_path}")
            return None
            
        # 找到最大的轮廓（假设是π符号）
        max_contour = max(contours, key=cv2.contourArea)
        
        # 创建掩码
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
        
        # 应用掩码
        result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        result[:,:,0:3] = img
        result[:,:,3] = mask
        
        # 裁剪到符号区域
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = result[y:y+h, x:x+w]
        
        # 调整大小
        resized = cv2.resize(cropped, (size, size))
        
        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, resized)
            print(f"已保存提取的π符号图像到 {output_path}")
        
        return resized
    except Exception as e:
        print(f"提取π符号时出错: {e}")
        return None

def create_glowing_pi_symbol(base_image=None, size=200, output_path=None):
    """Create a glowing pi symbol
    
    Args:
        base_image: Base image to apply glow effect, if None, a basic pi symbol is created
        size: Size of the output image
        output_path: Path to save the image, if None, the image is not saved
        
    Returns:
        The created image
    """
    # 如果没有提供基础图像，创建一个基本符号
    if base_image is None:
        base_image = create_basic_pi_symbol(size)
    elif isinstance(base_image, str):
        # 如果提供的是路径，读取图像
        base_image = cv2.imread(base_image, cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"无法读取基础图像，使用默认图像")
            base_image = create_basic_pi_symbol(size)
        else:
            # 调整大小
            base_image = cv2.resize(base_image, (size, size))
    
    # 确保图像有alpha通道
    if base_image.shape[2] == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
    
    # 创建发光效果
    blur_size = int(size / 10)
    if blur_size % 2 == 0:
        blur_size += 1  # 确保是奇数
    glow = cv2.GaussianBlur(base_image, (blur_size, blur_size), 0)
    
    # 增强发光效果
    glow = cv2.addWeighted(glow, 1.5, glow, 0, 0)
    
    # 合并原始图像和发光效果
    result = np.zeros_like(base_image)
    for c in range(3):  # RGB通道
        result[:,:,c] = np.minimum(base_image[:,:,c] + glow[:,:,c], 255)
    result[:,:,3] = base_image[:,:,3]  # 保持原始alpha通道
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"已保存发光π符号图像到 {output_path}")
    
    return result

def create_colorful_pi_symbol(base_image=None, size=200, output_path=None):
    """Create a colorful pi symbol
    
    Args:
        base_image: Base image to apply color effect, if None, a basic pi symbol is created
        size: Size of the output image
        output_path: Path to save the image, if None, the image is not saved
        
    Returns:
        The created image
    """
    # 如果没有提供基础图像，创建一个基本符号
    if base_image is None:
        base_image = create_basic_pi_symbol(size)
    elif isinstance(base_image, str):
        # 如果提供的是路径，读取图像
        base_image = cv2.imread(base_image, cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"无法读取基础图像，使用默认图像")
            base_image = create_basic_pi_symbol(size)
        else:
            # 调整大小
            base_image = cv2.resize(base_image, (size, size))
    
    # 确保图像有alpha通道
    if base_image.shape[2] == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
    
    # 创建彩色渐变
    gradient = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            # 创建彩虹渐变
            r = int(255 * (0.5 + 0.5 * np.sin(0.1 * x + 0.1 * y)))
            g = int(255 * (0.5 + 0.5 * np.sin(0.1 * x + 0.1 * y + 2)))
            b = int(255 * (0.5 + 0.5 * np.sin(0.1 * x + 0.1 * y + 4)))
            gradient[y, x] = [b, g, r]
    
    # 应用渐变到符号
    result = np.zeros_like(base_image)
    for c in range(3):  # RGB通道
        result[:,:,c] = np.where(base_image[:,:,3] > 0, gradient[:,:,c], 0)
    result[:,:,3] = base_image[:,:,3]  # 保持原始alpha通道
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"已保存彩色π符号图像到 {output_path}")
    
    return result

def create_neon_pi_symbol(base_image=None, size=200, output_path=None):
    """Create a neon pi symbol
    
    Args:
        base_image: Base image to apply neon effect, if None, a basic pi symbol is created
        size: Size of the output image
        output_path: Path to save the image, if None, the image is not saved
        
    Returns:
        The created image
    """
    # 如果没有提供基础图像，创建一个基本符号
    if base_image is None:
        base_image = create_basic_pi_symbol(size)
    elif isinstance(base_image, str):
        # 如果提供的是路径，读取图像
        base_image = cv2.imread(base_image, cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"无法读取基础图像，使用默认图像")
            base_image = create_basic_pi_symbol(size)
        else:
            # 调整大小
            base_image = cv2.resize(base_image, (size, size))
    
    # 确保图像有alpha通道
    if base_image.shape[2] == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
    
    # 创建霓虹效果
    # 首先创建一个蓝色版本
    neon = np.zeros_like(base_image)
    neon[:,:,0] = 255  # 蓝色通道
    neon[:,:,3] = base_image[:,:,3]  # 使用原始alpha
    
    # 添加发光效果
    blur_size = int(size / 8)
    if blur_size % 2 == 0:
        blur_size += 1  # 确保是奇数
    glow = cv2.GaussianBlur(neon, (blur_size, blur_size), 0)
    
    # 增强发光效果
    glow = cv2.addWeighted(glow, 2.0, glow, 0, 0)
    
    # 合并原始图像和发光效果
    result = glow.copy()
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"已保存霓虹π符号图像到 {output_path}")
    
    return result

def create_small_pi_symbols(base_image=None, count=20, size=50, output_dir=None):
    """Create multiple small pi symbols for animation
    
    Args:
        base_image: Base image to create variations, if None, a basic pi symbol is created
        count: Number of small symbols to create
        size: Size of each small symbol
        output_dir: Directory to save images, if None, images are not saved
        
    Returns:
        List of created images
    """
    # 如果没有提供基础图像，创建一个基本符号
    if base_image is None:
        base_image = create_basic_pi_symbol(size*4)  # 创建较大的图像，然后缩小
    elif isinstance(base_image, str):
        # 如果提供的是路径，读取图像
        base_image = cv2.imread(base_image, cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"无法读取基础图像，使用默认图像")
            base_image = create_basic_pi_symbol(size*4)
    
    # 确保图像有alpha通道
    if base_image.shape[2] == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
    
    # 创建多个变体
    small_symbols = []
    
    for i in range(count):
        # 创建变体
        variant = base_image.copy()
        
        # 应用随机变换
        # 1. 随机旋转
        angle = np.random.uniform(-30, 30)
        center = (variant.shape[1] // 2, variant.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variant = cv2.warpAffine(variant, M, (variant.shape[1], variant.shape[0]))
        
        # 2. 随机缩放
        scale = np.random.uniform(0.8, 1.2)
        variant = cv2.resize(variant, None, fx=scale, fy=scale)
        
        # 确保图像大小正确
        if variant.shape[0] != variant.shape[1]:
            # 创建一个正方形画布
            square_size = max(variant.shape[0], variant.shape[1])
            square = np.zeros((square_size, square_size, 4), dtype=np.uint8)
            
            # 将变体放在中心
            x_offset = (square_size - variant.shape[1]) // 2
            y_offset = (square_size - variant.shape[0]) // 2
            
            # 确保不会越界
            h, w = min(variant.shape[0], square_size), min(variant.shape[1], square_size)
            square[y_offset:y_offset+h, x_offset:x_offset+w] = variant[:h, :w]
            variant = square
        
        # 调整到最终大小
        variant = cv2.resize(variant, (size, size))
        
        small_symbols.append(variant)
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"small_pi_{i:02d}.png")
            cv2.imwrite(output_path, variant)
            print(f"已保存小π符号 {i} 到 {output_path}")
    
    return small_symbols

def create_digit_images(size=100, output_dir=None):
    """Create images for pi digits (0-9 and decimal point)
    
    Args:
        size: Size of each digit image
        output_dir: Directory to save images, if None, images are not saved
        
    Returns:
        Dictionary of digit images
    """
    digits = {}
    
    # 创建0-9的数字图像
    for digit in range(10):
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
        
        digits[str(digit)] = result
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"digit_{digit}.png")
            cv2.imwrite(output_path, result)
            print(f"已保存数字 {digit} 到 {output_path}")
    
    # 创建小数点图像
    img = np.zeros((size, size, 4), dtype=np.uint8)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size / 100 * 2.5
    thickness = int(size / 30)
    color = (255, 255, 255, 255)
    
    text = "."
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    x = (size - text_size[0]) // 2
    y = (size + text_size[1]) // 2
    
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness)
    
    blur_size = int(size / 10)
    if blur_size % 2 == 0:
        blur_size += 1
    img_blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    result = np.zeros_like(img)
    for c in range(3):
        result[:,:,c] = np.minimum(img[:,:,c] + img_blurred[:,:,c], 255)
    result[:,:,3] = img[:,:,3]
    
    digits["."] = result
    
    # 保存小数点图像
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "digit_dot.png")
        cv2.imwrite(output_path, result)
        print(f"已保存小数点到 {output_path}")
    
    return digits

if __name__ == "__main__":
    # 设置输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "assets")
    pi_symbols_dir = os.path.join(assets_dir, "pi_symbols")
    small_pi_dir = os.path.join(pi_symbols_dir, "small")
    digits_dir = os.path.join(pi_symbols_dir, "digits")
    
    # 确保目录存在
    os.makedirs(pi_symbols_dir, exist_ok=True)
    os.makedirs(small_pi_dir, exist_ok=True)
    os.makedirs(digits_dir, exist_ok=True)
    
    # 尝试从images.png提取π符号
    images_path = os.path.join(assets_dir, "images.png")
    extracted_path = os.path.join(pi_symbols_dir, "pi_symbol_extracted.png")
    extracted_image = None
    
    if os.path.exists(images_path):
        print(f"从 {images_path} 提取π符号")
        extracted_image = extract_pi_from_image(images_path, extracted_path, 200)
    else:
        print(f"未找到 {images_path}，将使用基本π符号")
    
    # 创建基本π符号
    basic_path = os.path.join(pi_symbols_dir, "pi_symbol_basic.png")
    basic_image = create_basic_pi_symbol(200, basic_path)
    
    # 创建风格化π符号
    stylized_path = os.path.join(pi_symbols_dir, "pi_symbol_stylized.png")
    # 使用提取的图像（如果可用）或基本图像
    base_image = extracted_image if extracted_image is not None else basic_image
    stylized_image = create_stylized_pi_symbol(200, stylized_path)
    
    # 创建发光π符号
    glowing_path = os.path.join(pi_symbols_dir, "pi_symbol_glowing.png")
    create_glowing_pi_symbol(base_image, 200, glowing_path)
    
    # 创建彩色π符号
    colorful_path = os.path.join(pi_symbols_dir, "pi_symbol_colorful.png")
    create_colorful_pi_symbol(base_image, 200, colorful_path)
    
    # 创建霓虹π符号
    neon_path = os.path.join(pi_symbols_dir, "pi_symbol_neon.png")
    create_neon_pi_symbol(base_image, 200, neon_path)
    
    # 创建小π符号（用于动画）
    create_small_pi_symbols(base_image, 20, 50, small_pi_dir)
    
    # 创建数字图像（用于嘴部动画）
    create_digit_images(100, digits_dir)
    
    print("所有π符号图像创建完成！") 