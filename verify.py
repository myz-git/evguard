import os
import sys
import glob
import logging
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pyautogui

# Updated IconCNN with Adaptive Pooling
class IconCNN(nn.Module):
    def __init__(self):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Adaptive pooling to 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Screen capture function (placeholder, implement as needed)
def capture_screen():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Debug image saving function (placeholder, implement as needed)
def save_debug_image(image, filename):
    cv2.imwrite(f"debug_{filename}.png", image)

def verify_icon(icon_name, max_attempts=3, template_threshold=0.72, model_threshold=0.85):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IconCNN().to(device)
    model_path = os.path.join('model_cnn', f"{icon_name}_classifier.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found.")
        sys.exit(1)
    except RuntimeError as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    
    # Transform without resizing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    template_path = os.path.join('icon', f"{icon_name}-1.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        logging.error(f"Template image {template_path} not found")
        sys.exit(1)
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    
    for attempt in range(max_attempts):
        screen_image = capture_screen()
        #save_debug_image(screen_image, f"screen_attempt_{attempt}")
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(res)
        logging.info(f"Attempt {attempt+1}: Template matching max_val = {max_val}")
        
        loc = np.where(res >= template_threshold)
        matches = list(zip(*loc[::-1]))
        
        if matches:
            for pt in matches:
                top_left = pt
                bottom_right = (top_left[0] + w, top_left[1] + h)
                icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                #save_debug_image(icon_image, f"icon_attempt_{attempt}_{pt}")
                icon_rgb = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(icon_rgb)
                tensor = transform(pil_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    # 记录候选区域的置信度和预测结果
                    logging.info(f"候选区域 {pt}: 置信度 {confidence.item():.4f}, 预测类别 {predicted.item()}")
                    if predicted.item() == 1 and confidence.item() > model_threshold:
                        x = top_left[0] + w // 2
                        y = top_left[1] + h // 2
                        logging.info(f"找到 {icon_name} 坐标: ({x}, {y})")
                        try:
                            pyautogui.moveTo(x, y)
                            cv2.imshow(f"Detected {icon_name}", icon_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        except Exception as e:
                            logging.error(f"Failed to display image or move mouse: {e}")
                        return (x, y)
            logging.info(f"Attempt {attempt+1}: Icon {icon_name} not detected in matched regions")
        else:
            logging.info(f"Attempt {attempt+1}: No matches found for {icon_name}")
    
    return False


def evaluate_accuracy(icon_name):
    """
    在 traindata/{icon}-1 与 traindata/{icon}-0 上评估模型准确率，
    输出：准确率、精确率、召回率、F1、混淆矩阵。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IconCNN().to(device)
    model_path = os.path.join('../model_cnn', f"{icon_name}_classifier.pth")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found. Train first: python train.py {icon_name}")
        sys.exit(1)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    pos_dir = os.path.join('traindata', f'{icon_name}-1')
    neg_dir = os.path.join('traindata', f'{icon_name}-0')
    pos_paths = sorted(glob.glob(os.path.join(pos_dir, '*.png')))
    neg_paths = sorted(glob.glob(os.path.join(neg_dir, '*.png')))
    if not pos_paths:
        logging.error(f"No images in {pos_dir}")
        sys.exit(1)
    if not neg_paths:
        logging.error(f"No images in {neg_dir}. Need both -1 and -0 for accuracy evaluation.")
        sys.exit(1)

    all_paths = pos_paths + neg_paths
    true_labels = [1] * len(pos_paths) + [0] * len(neg_paths)

    pred_labels = []
    with torch.no_grad():
        for path in all_paths:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(probs, 1)
            pred_labels.append(pred.item())

    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    n = len(true_labels)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n===== 模型准确率验证 [{icon_name}] =====")
    print(f"  样本: 正 {len(pos_paths)} / 负 {len(neg_paths)}")
    print(f"  准确率 (Accuracy):  {accuracy:.2%}")
    print(f"  精确率 (Precision): {precision:.2%}  (预测为正的里真正是正的比例)")
    print(f"  召回率 (Recall):    {recall:.2%}  (真实正样本中被正确识别的比例)")
    print(f"  F1:                 {f1:.2%}")
    print(f"  混淆矩阵:  pred=0  pred=1")
    print(f"    true=0     {tn:4d}    {fp:4d}")
    print(f"    true=1     {fn:4d}    {tp:4d}")
    print("========================================\n")
    return accuracy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="验证图标 CNN 模型：实时找图 或 在 traindata 上算准确率")
    parser.add_argument("icon_name", nargs="?", help="图标名，如 jump5")
    parser.add_argument("--eval", action="store_true", help="在 traindata 上评估准确率（精确率/召回率/F1/混淆矩阵）")
    parser.add_argument("--max-attempts", type=int, default=3, help="实时验证时的截屏尝试次数")
    args = parser.parse_args()
    if not args.icon_name:
        parser.print_help()
        print("\nExample:  python verify.py jump5 --eval   # 评估准确率")
        print("          python verify.py jump5           # 实时屏幕找图验证")
        sys.exit(1)
    if args.eval:
        evaluate_accuracy(args.icon_name)
    else:
        verify_icon(args.icon_name, max_attempts=args.max_attempts)