# -*- coding: utf-8 -*-
"""
通用工具库（建议作为 eveauto/evalert 共用）
- 图标识别：模板匹配 + CNN 二次验证（.pth）
- OCR：CnOcr 查找/提取
- 截图/区域缩放/日志/纠错/地址读取
"""

import os
import sys
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import pyautogui
from cnocr import CnOcr

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pyttsx3

from pydub import AudioSegment
from pydub.playback import play

# =========================
# 打包需要加载路径
# =========================
def resource_path(rel_path: str) -> str:
    # PyInstaller onefile: sys._MEIPASS 存在
    if getattr(sys, "_MEIPASS", None):
        base = sys._MEIPASS
    # PyInstaller onedir: 用 exe 所在目录（多 exe 共用 dist 目录）
    elif getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    # 源码运行：用当前文件所在目录（或你项目根）
    else:
        base = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base, rel_path)

ICON_DIR = resource_path("icon")
MODEL_DIR = resource_path("model_cnn")

# =========================
# 日志 & 调试截图
# =========================

DEBUG_DIR = "debug_screenshots"
os.makedirs(DEBUG_DIR, exist_ok=True)

class NoCnocrFilter(logging.Filter):
    """禁用 cnocr/cnstd 的 use model 噪音日志"""
    def filter(self, record):
        msg = record.getMessage().lower()
        return "use model" not in msg

logging.getLogger("").addFilter(NoCnocrFilter())
logging.getLogger("cnocr").setLevel(logging.ERROR)
logging.getLogger("cnstd").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("task.log", mode="a", encoding="utf-8"), logging.StreamHandler()],
    force=True,
)

# =========================
# 控制台 WARNING 过滤（fsd0/fsd10 不输出「未找到」类 WARNING）
# =========================

class _SuppressNotFoundWarningFilter(logging.Filter):
    """过滤掉 WARNING 且消息含「未找到」的日志，不输出到控制台（仍写入文件）。"""
    def filter(self, record):
        if record.levelno != logging.WARNING:
            return True
        return "未找到" not in (record.getMessage() or "")


def suppress_not_found_warnings_console():
    """为当前进程的根 logger 的 StreamHandler 添加过滤，不向控制台输出「未找到」类 WARNING。"""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.addFilter(_SuppressNotFoundWarningFilter())
            break

# =========================
# 启动界面（终端友好输出）
# =========================

def print_startup(app_name: str, hints=None):
    """
    在终端输出带分隔线的友好启动界面。
    :param app_name: 程序名称，如 "FSD0", "FSD10", "GuardA", "GuardB"
    :param hints: 可选，提示文案列表，如 ["按 Ctrl+F12 可停止程序"]
    """
    width = 44
    line = "=" * width
    pad = "  "
    parts = ["", line, pad + f"{app_name} 已启动，正在运行"]
    if hints:
        for h in hints:
            parts.append(pad + h)
    parts.extend([line, ""])
    print("\n".join(parts), flush=True)


def log_message(level: str, message: str, screenshot: bool = False, script_name: str = "debug", suffix: str = ""):
    """
    兼容旧调用：log_message(..., screenshot=True/False)
    screenshot=True 时保存截图，并把路径写入日志
    """
    shot_path = None
    if screenshot:
        try:
            shot_path = save_screenshot(script_name=script_name, suffix=suffix)
        except Exception:
            shot_path = None

    if shot_path:
        message = f"{message} | screenshot={shot_path}"

    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    else:
        logging.error(message)

    # flush
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass

def save_screenshot(script_name: str, suffix: str = "") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DEBUG_DIR, f"{ts}_{script_name}{suffix}.png")
    pyautogui.screenshot().save(path)
    return path


# =========================
# 异常
# =========================
class IconNotFoundException(Exception):
    pass

class TextNotFoundException(Exception):
    pass


# =========================
# 分辨率缩放
# =========================
def adjust_region(region: Tuple[int, int, int, int], base_resolution=(1920, 1080)):
    """根据当前屏幕分辨率缩放 ROI（以 1920x1080 为基准）"""
    current_width, current_height = pyautogui.size()
    base_width, base_height = base_resolution
    x, y, w, h = region
    x = int(x * current_width / base_width)
    y = int(y * current_height / base_height)
    w = int(w * current_width / base_width)
    h = int(h * current_height / base_height)
    return (x, y, w, h)


# =========================
# 截图 & 预处理
# =========================
def capture_screen_area(region: Tuple[int, int, int, int], save_path: Optional[str] = None, do_adjust=True):
    """捕获指定区域 -> OpenCV BGR"""
    r = adjust_region(region) if do_adjust else region
    screenshot = pyautogui.screenshot(region=r)
    if save_path:
        screenshot.save(save_path)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """OCR 预处理：灰度 -> CLAHE -> 锐化 -> OTSU"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    contrast = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(contrast, -1, kernel)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# =========================
# 滚动（注意：会改变 UI）
# =========================
def scrollscreen():
    """水平转动屏幕（会移动鼠标/改变视角，请慎用）"""
    time.sleep(1)
    pyautogui.moveTo(250, 700)
    pyautogui.dragRel(-30, 0, 0.5, pyautogui.easeOutQuad)

def hscrollscreen():
    """垂直滑动（可选）"""
    pyautogui.moveTo(1600, 400)
    pyautogui.scroll(-2000)
    time.sleep(1)


# =========================
# OCR：CnOcr 单例缓存
# =========================
_OCR_INSTANCE: Optional[CnOcr] = None

def _get_ocr() -> CnOcr:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = CnOcr(rec_model_name="scene-densenet_lite_246-gru_base")
    return _OCR_INSTANCE

def find_txt_ocr(txt: str, max_attempts=5, region=None, allow_scroll=True) -> bool:
    """在屏幕区域内查找文本，找到则移动鼠标到文本中心并返回 True"""
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    ocr = _get_ocr()
    attempts = 0
    while attempts < max_attempts:
        r = adjust_region(region) if do_adjust else region
        img = pyautogui.screenshot(region=r)
        res = ocr.ocr(img)
        for line in res:
            if txt in line["text"]:
                x = r[0] + line["position"][0][0] + (line["position"][1][0] - line["position"][0][0]) // 2
                y = r[1] + line["position"][0][1] + (line["position"][2][1] - line["position"][0][1]) // 2
                pyautogui.moveTo(x, y)
                log_message("INFO", f"找到[{txt}] 坐标=({x},{y})")
                return True

        attempts += 1
        if allow_scroll:
            scrollscreen()
        time.sleep(0.3)

    log_message("WARNING", f"[{txt}] 未找到，尝试次数={max_attempts}")
    return False

def find_txt_ocr2(prefix_txt: str, max_attempts=5, region=None, allow_scroll=True) -> Optional[str]:
    """
    用于提取“固定前缀 + 内容”的场景：匹配 prefix_txt 后提取其后面的汉字
    例：prefix_txt='货物' -> 返回货物名
    """
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    ocr = _get_ocr()
    pattern = re.compile(f"{prefix_txt}([^()]+)")
    attempts = 0

    while attempts < max_attempts:
        img = capture_screen_area(region, do_adjust=do_adjust)
        res = ocr.ocr(img)

        for item in res:
            m = pattern.search(item["text"])
            if m:
                extracted = re.sub(r"[^\u4e00-\u9fff]", "", m.group(1))
                log_message("INFO", f"OCR提取: prefix={prefix_txt}, value={extracted}")
                return extracted

        attempts += 1
        if allow_scroll:
            scrollscreen()
        time.sleep(0.3)

    log_message("WARNING", f"OCR提取失败: prefix={prefix_txt}, attempts={max_attempts}")
    return None

def find_txt_ocr3(txt: str, max_attempts=1, region=None) -> int:
    """返回识别到的 txt 个数（evalert 的能力合入）"""
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    ocr = _get_ocr()
    attempts = 0
    while attempts < max_attempts:
        r = adjust_region(region) if do_adjust else region
        img = pyautogui.screenshot(region=r)
        res = ocr.ocr(img)
        count = sum(1 for line in res if txt in line["text"])
        if count > 0:
            log_message("INFO", f"识别到 {count} 个 [{txt}]")
        return count
    return 0


# =========================
# 地址读取（evalert 的 load_location_name 合入）
# =========================
def load_location_name(tag: str, file_path="addr.txt") -> Optional[str]:
    """
    从 addr.txt (json, utf-8-sig) 读取 location name
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.loads(f.read())
        return data.get(tag)
    except FileNotFoundError:
        log_message("ERROR", f"地址文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        log_message("ERROR", f"地址文件JSON解析失败: {e}")
    except UnicodeDecodeError:
        log_message("ERROR", "地址文件编码错误，建议 utf-8-sig")
    return None


# =========================
# OCR 纠错（合并两边规则）
# =========================
def correct_string(input_str: str) -> str:
    rules = [
        ("天", "大"),
        ("性", "牲"),
        ("拉", "垃"),
        ("级", "圾"),
        ("杀", "OP杀"),
        ("者门", "看门"),
    ]
    out = input_str
    for old, new in rules:
        out = re.sub(old, new, out)
    return out


# =========================
# CNN 图标识别（模板匹配 + 模型验证）
# =========================

@dataclass
class CnnConfig:

    # template_threshold是用来通过icon模版找到图片位置,可以低一些,减少漏检
    # cnn_threshold是用来验证找到的图片位置是否是目标icon,可以高一些,减少误触
    # 想减少漏检：template_threshold 稍微降一点（比如 0.80→0.78）
    # 想减少误触：cnn_threshold 稍微升一点（比如 0.80→0.85）
    # topk是用来限制验证的候选点数量,可以少一些,减少计算量
    # dedup_ratio是用来去重,可以小一些,减少误触
    # debug_save是用来保存候选crop到DEBUG_DIR,可以打开,方便调试

    template_threshold: float = 0.75  # 模板匹配阈值
    cnn_threshold: float = 0.80       # CNN pos 置信度阈值
    topk: int = 20                    # 只验证匹配分数Top-K候选点
    dedup_ratio: float = 0.5          # 去重半径比例：min(w,h)*dedup_ratio
    debug_save: bool = False          # 是否保存候选crop到DEBUG_DIR


class IconCNN(nn.Module):
    """必须与 evmodel/cnn/train.py 的网络结构一致"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
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
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 推理预处理：建议与 train.py 的 transforms 保持一致
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#_CNN_INPUT_SIZE = (64, 64)  #启用resize后,就识别不准确了. #
_TRANSFORM = transforms.Compose([
    # transforms.Resize(_CNN_INPUT_SIZE), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


_MODEL_CACHE: Dict[str, nn.Module] = {}

def _get_cnn_model(icon: str, model_dir=MODEL_DIR) -> nn.Module:    
    """同一个 icon 的 pth 只加载一次"""
    if icon in _MODEL_CACHE:
        return _MODEL_CACHE[icon]

    model_path = os.path.join(model_dir, f"{icon}_classifier.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = IconCNN().to(_DEVICE)
    state = torch.load(model_path, map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[icon] = model
    log_message("INFO", f"加载CNN模型: {model_path}")
    return model

def find_icon_cnn(
        icon: str,
        region=None,
        offset_x: int = 0,
        offset_y: int = 0,
        cfg: Optional[CnnConfig] = None,
        icon_dir: str = ICON_DIR,
        model_dir: str = MODEL_DIR,
        threshold: Optional[float] = None,
        cnn_threshold: Optional[float] = None,
        move: bool = True,
) -> bool:
    """
    单次查找：模板匹配找到候选 -> CNN 验证

    - 候选点按模板匹配分数降序排序
    - 半径去重：同一图标附近候选只保留最高分一个
    - 只验证 cfg.topk 个候选
    - move=True：通过验证后 moveTo 图标中心（不点击）

    返回 True/False

    调用示例

    - 只判断存在不移动鼠标
    ok = find_icon_cnn("jump4", region=region_full_right, move=False)
    
    - 更防漏检一点（候选更多、去重更宽）,配置cfg : 用 cfg 来统一控制阈值/TopK/去重/是否保存候选图
    cfg = CnnConfig(
    template_threshold=0.80,
    cnn_threshold=0.80,
    topk=30,
    dedup_ratio=0.6,
    debug_save=False
    )
    ok = find_icon_cnn("jump4", region=region_full_right, cfg=cfg, move=False)
    """
    if cfg is None:
        cfg = CnnConfig()

    # 兼容旧参数：threshold/cnn_threshold 覆盖 cfg
    if threshold is not None:
        cfg.template_threshold = threshold
    if cnn_threshold is not None:
        cfg.cnn_threshold = cnn_threshold

    # region 处理
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    # 读取模板
    template_path = os.path.join(icon_dir, f"{icon}.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像无法加载: {template_path}")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2]

    # 截图
    screen = capture_screen_area(region, do_adjust=do_adjust)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 记录 max_val：定位到底是模板阈值卡住还是CNN阈值卡住
    max_val = float(np.max(res)) if res.size else 0.0
    log_message("INFO", f"[{icon}] template max_val={max_val:.4f}, thr={cfg.template_threshold:.4f}")

    ys, xs = np.where(res >= cfg.template_threshold)
    if len(xs) == 0:
        return False

    # 收集候选点 + 分数，并按分数降序排序（防漏检关键）
    cands = [(int(x), int(y), float(res[y, x])) for x, y in zip(xs, ys)]
    cands.sort(key=lambda t: t[2], reverse=True)

    # 半径去重：避免同一图标附近重复候选
    min_dist = max(1, int(min(w, h) * float(cfg.dedup_ratio)))
    min_dist2 = min_dist * min_dist

    filtered = []
    for x, y, s in cands:
        keep = True
        for fx, fy, fs in filtered:
            if (x - fx) * (x - fx) + (y - fy) * (y - fy) <= min_dist2:
                keep = False
                break
        if keep:
            filtered.append((x, y, s))
        if len(filtered) >= int(cfg.topk):
            break

    # 加载CNN模型（缓存）
    model = _get_cnn_model(icon, model_dir=model_dir)

    # 遍历候选做CNN验证
    for idx, (x0, y0, score) in enumerate(filtered):
        crop = screen[y0:y0 + h, x0:x0 + w]
        if crop.size == 0:
            continue

        # debug 保存候选 crop（可选）
        if cfg.debug_save:
            try:
                debug_path = os.path.join(DEBUG_DIR, f"{icon}_match_{idx}_s{score:.3f}.png")
                cv2.imwrite(debug_path, crop)
            except Exception:
                pass

        # BGR -> RGB -> PIL -> Tensor
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)[0, 1].item()  # pos 概率

        log_message("INFO", f"[{icon}] cand#{idx} pt=({x0},{y0}) score={score:.4f} prob={prob:.4f} thr={cfg.cnn_threshold:.4f}")

        if prob >= cfg.cnn_threshold:
            r = adjust_region(region) if do_adjust else region
            mx = x0 + r[0] + w // 2 + offset_x
            my = y0 + r[1] + h // 2 + offset_y

            if move:
                pyautogui.moveTo(mx, my)

            return True

    return False

def safe_find_icon(
        icon: str,
        region=None,
        max_attempts: int = 1,
        action: str = "leftclick",
        offset_x: int = 0,
        offset_y: int = 0,
        cfg: Optional[CnnConfig] = None,
        allow_scroll: bool = False,
        threshold: Optional[float] = None,
        cnn_threshold: Optional[float] = None,
        move: bool = True,
) -> bool:
    """
    多次尝试封装：模板匹配 + CNN 验证
    - cfg（方式B）：统一控制 template_threshold/cnn_threshold/topk/dedup_ratio/debug_save
    - threshold/cnn_threshold：兼容旧参数，会覆盖 cfg 中的阈值
    - move：是否移动鼠标到目标（默认 True）
    - action:
        - "leftclick": 找到后左键点击
        - "rightclick": 找到后右键点击
        - "none" 或 None: 只定位（可 move），不点击
    - allow_scroll=True：找不到时会调用 scrollscreen()/scollscreen()（注意会改变 UI）

    返回 True/False

    调用示例:
    cfg = CnnConfig(template_threshold=0.80, cnn_threshold=0.80, topk=30, dedup_ratio=0.6)
    # 只判断存在，不移动鼠标、不点击
    ok = safe_find_icon("jump4", region=region_full_right, cfg=cfg, move=False, action="none", allow_scroll=False)
    # 需要点击
    ok = safe_find_icon("close1", region=right_panel, cfg=cfg, action="leftclick")
    """
    if cfg is None:
        cfg = CnnConfig()

    # 兼容旧参数：覆盖 cfg
    if threshold is not None:
        cfg.template_threshold = threshold
    if cnn_threshold is not None:
        cfg.cnn_threshold = cnn_threshold

    for attempt in range(max_attempts):
        found = find_icon_cnn(
            icon=icon,
            region=region,
            offset_x=offset_x,
            offset_y=offset_y,
            cfg=cfg,
            icon_dir=ICON_DIR,
            model_dir=MODEL_DIR,
            threshold=threshold,
            cnn_threshold=cnn_threshold,
            move=move,
        )

        if found:
            time.sleep(0.15)

            if action in ("leftclick", "click", "left"):
                pyautogui.leftClick()
            elif action in ("rightclick", "right"):
                pyautogui.rightClick()
            # action == "none"/None: 不点击

            log_message("INFO", f"找到[{icon}] attempt={attempt+1}/{max_attempts} action={action}")
            return True

        if allow_scroll:
            scrollscreen()
        else:
            time.sleep(1)

    log_message("WARNING", f"未找到[{icon}] attempts={max_attempts}")
    return False

# 屏幕区域配置
screen_regions = {
    'left_panel': (30, 30, 300, 800),  # 左侧面板
    'center_panel': (300, 30, 800, 800),  # 中间面板
    'center_panel2': (200, 30,1200, 800),  # 中间面板2
    'right_panel': (1000, 30, 700, 1000),  # 右侧面板
    'full_right_panel': (1380, 30, 540, 1000),
    'upper_right_panel': (1380, 30, 540, 260),
    'upper_left_panel': (0, 0, 400, 200),
    'mid_left_panel': (50, 150, 500, 600),
    'agent_panel1': (1450, 250, 500, 500),
    'agent_panel2': (1500, 400, 400, 500),
    'agent_panel3': (200, 100, 1400, 900),
    'cangku_panel3': (0, 0, 1700, 850),
    'need_goods_panel': (50, 50, 400, 500),
    'control_panel': (500, 800, 850,230)
}

# =========================
# 业务相关函数
# =========================
def rolljump(max_attempts=0):
    """循环跳跃星门"""
    region_full_right = screen_regions['full_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']
    attempts = 0
    while max_attempts == 0 or attempts < max_attempts:
        # 首先检查是否到达目的地
        if find_txt_ocr("跃迁至该处", max_attempts=1, region=mid_left_panel):
            log_message("INFO", "已到达目的地")
            return 0  # 程序停止
        
        if safe_find_icon("jump3", region_full_right, max_attempts=1):
            log_message("INFO", "找到jump3，退出rolljump")
            return True 
        else:
            safe_find_icon("jump1", region_full_right, max_attempts=1)
            safe_find_icon("jump2", region_full_right, max_attempts=1)          
            log_message("INFO", f"rolljump,循环跳跃星门:{attempts}")
        time.sleep(2)
        attempts += 1
    log_message("ERROR", f"rolljump达到最大尝试次数: {max_attempts}", screenshot=True)
    return False  # 返回特殊状态

def rolljump2(max_attempts=0):
    """新的循环跳跃星门方式"""
    region_full_right = screen_regions['full_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']
    control_panel = screen_regions['control_panel']
    attempts = 0
    
    while max_attempts == 0 or attempts < max_attempts:

        # 先找jump3
        if safe_find_icon("jump3", region_full_right, max_attempts=1,offset_x=0,offset_y=0):
            log_message("INFO", "找到jump3，退出rolljump2")
            return True
        
        # 找不到jump3，则找jump1
        log_message("INFO", "未找到jump3，尝试查找jump1")
        if safe_find_icon("jump1", region_full_right, max_attempts=1):
            log_message("INFO", "找到jump1")
            time.sleep(1)
            # 找warp1
            log_message("INFO", "查找warp1")
            if safe_find_icon("warp1", control_panel, max_attempts=3,threshold=0.9,cnn_threshold=0.65):
                log_message("INFO", "找到warp1，等待3秒后再次点击")
                pyautogui.doubleClick()
                time.sleep(3)
                pyautogui.doubleClick()  # 再次点击warp1
                time.sleep(1)
                # 再点击jump2
                log_message("INFO", "查找并点击jump2")
                if safe_find_icon("jump2", region_full_right, max_attempts=3,threshold=0.9,cnn_threshold=0.7):
                    log_message("INFO", "找到并点击jump2，等待10秒")
                    time.sleep(10)
            else:
                log_message("WARNING", "未找到warp1")
        else:
            log_message("WARNING", "未找到jump1")
        
        attempts += 1
        log_message("INFO", f"rolljump2循环次数: {attempts}")
        time.sleep(2)
    
    log_message("ERROR", f"rolljump2达到最大尝试次数: {max_attempts}", screenshot=True)
    return False

def find_and_close_icons(icon, region):
    if safe_find_icon(icon, region, max_attempts=1, threshold=0.86):
        time.sleep(1)
        log_message("INFO", f"关闭图标: {icon}")
        return True
    log_message("INFO", f"未找到关闭图标: {icon}")
    return False

def close_icons_main():
    find_and_close_icons("close1", region=None)

def play_sound_wav(file_path):
    """播放 WAV 文件。Windows 下用 winsound 避免 pydub 写临时文件时的权限错误。"""
    file_path = resource_path(file_path)
    if not os.path.isfile(file_path):
        logging.warning(f"play_sound_wav: 文件不存在 {file_path}")
        return
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        else:
            sound = AudioSegment.from_file(file_path, format="wav")
            play(sound)
    except Exception as e:
        logging.warning(f"play_sound_wav 失败: {e}")

def speak(text,rate=220):
    # 初始化语音引擎
    engine = pyttsx3.init()

    # 获取并设置语音
    voices = engine.getProperty('voices')
    huihui_voice = next((voice for voice in voices if 'Huihui' in voice.name), None)
    if huihui_voice:
        engine.setProperty('voice', huihui_voice.id)
    else:
        print("Microsoft Huihui Desktop voice not found, using default.")

    # 设置语速
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh-CN')
    engine.setProperty('volume', 4)
    engine.setProperty('rate', rate)

    # 文本转语音3
    engine.say(text)
    engine.runAndWait()


# =========================
# check_errors,run_with_timeout不确定是否使用
# =========================
def check_errors():
    """检查错误提示窗口"""
    if find_txt_ocr("错误", max_attempts=1, region=None) or find_txt_ocr("警告", max_attempts=1, region=None):
        log_message("WARNING", "检测到错误或警告提示，尝试关闭窗口", screenshot=True)
        return True
    return False

def run_with_timeout(timeout, action, log_prefix):
    """带超时的循环执行"""
    start_time = time.time()
    while time.time() < start_time + timeout:
        if action():
            return True
        time.sleep(1)
    log_message("ERROR", f"{log_prefix}超时", screenshot=True)
    return False
