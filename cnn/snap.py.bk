import cv2
import numpy as np
import pyautogui
import time
import os
import sys
from glob import glob


def capture_screen():
    """捕获整个屏幕的截图并返回，转换屏幕截图为OpenCV可处理的BGR格式。"""
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
    return screen_image

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

    for i in range(num_captures):
        # 周期性重置视角（每50次循环拖动反方向）
        # if i % 50 == 0:
        #     pyautogui.moveTo(250, 700)
        #     pyautogui.dragRel(30, 0, 0.5, pyautogui.easeOutQuad)
        #     time.sleep(1.5)
        
        # # 水平旋转屏幕
        # scollscreen()
        time.sleep(0.5)
        
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
    if len(sys.argv) < 2:
        print("Usage: python snap.py jump1-1.png")
        sys.exit(1)

    template_filename = sys.argv[1]  # 从命令行参数获取图标文件名
    save_folder = '../traindata'  # 指定保存图标图像的根文件夹
    find_and_save_icon(template_filename, save_folder)