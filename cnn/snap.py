import cv2
import numpy as np
import pyautogui
import time
import os
import sys
import argparse
from glob import glob
from human_control import HumanMouse


def capture_screen():
    """捕获整个屏幕的截图并返回，转换屏幕截图为OpenCV可处理的BGR格式。"""
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
    return screen_image

def rotate_screen(mouse, step_index, center_x, center_y, h_step=30, v_step=20, duration=0.25):
    """交替进行左右/上下拖拽，自动旋转视角。"""
    pattern = step_index % 4
    if pattern == 0:
        dx, dy = h_step, 0       # 右
    elif pattern == 1:
        dx, dy = -h_step, 0      # 左
    elif pattern == 2:
        dx, dy = 0, v_step       # 下
    else:
        dx, dy = 0, -v_step      # 上

    mouse.move_to(center_x, center_y, duration=duration * 0.45)
    mouse.press_left()
    mouse.move_to(center_x + dx, center_y + dy, duration=duration, curvature=24)
    mouse.release_left()


def find_and_save_icon(template_filename, save_folder, capture_interval=0.5, num_captures=310):
    """使用模板匹配技术在屏幕截图中查找图标，并在找到后保存到指定文件夹。"""
    template_path = os.path.join('../icon', template_filename)  # 构建模板路径
    base_filename = os.path.splitext(template_filename)[0]  # 从文件名中提取基本名字，不含扩展名

    # 创建以图标名命名的文件夹
    icon_save_folder = os.path.join(save_folder, base_filename)
    if not os.path.exists(icon_save_folder):
        os.makedirs(icon_save_folder)

    # 计算当前文件夹中已存在的文件序号，以便继续编号
    existing_files = glob(os.path.join(icon_save_folder, f'{base_filename}-*.png'))
    max_index = 0
    if existing_files:
        max_index = max([int(file.split('-')[-1].split('.')[0]) for file in existing_files])

    save_count = max_index + 1  # 从最大序号后继续开始

    # 加载模板图标
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    screen_w, screen_h = pyautogui.size()
    center_x, center_y = screen_w // 2, screen_h // 2
    mouse = HumanMouse(human_factor=0.92, default_duration=0.35, default_offset=2)

    for i in range(num_captures):
        rotate_screen(mouse, i, center_x, center_y)
        time.sleep(0.3)
        
        screen_image = capture_screen()
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.8:  # 如果匹配度高于0.8，则认为找到了图标
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # 截取图标区域
            icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # 保存图标图像
            save_path = os.path.join(icon_save_folder, f'{base_filename}-{save_count}.png')
            cv2.imwrite(save_path, icon_image)
            print(f"Saved: {save_path}")
            save_count += 1
            
        else:
            print("Icon not found")

        time.sleep(capture_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动旋转屏幕并采集匹配到的图标样本")
    parser.add_argument("template", help="模板图标文件名，例如 jump1-1.png")
    parser.add_argument("--count", type=int, default=310, help="采集轮数，默认 310")
    args = parser.parse_args()

    if args.count <= 0:
        print("--count 必须是正整数")
        sys.exit(1)

    template_filename = args.template
    save_folder = '../traindata'  # 指定保存图标图像的根文件夹
    find_and_save_icon(template_filename, save_folder, num_captures=args.count)