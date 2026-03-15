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
import random
import math
import logging
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pyautogui
from cnocr import CnOcr
from human_control import HumanMouse

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

_SHARED_UI_LOG_FILE = os.environ.get("EVGUARD_UI_LOG_FILE", "").strip()
_SHARED_UI_LOG_PREFIX = os.environ.get("EVGUARD_UI_LOG_PREFIX", "").strip()
_SHARED_UI_LOG_LOCK = threading.RLock()
_LAST_MOUSE_TARGET_SIZE: Optional[Tuple[int, int]] = None
_HUMAN_MOUSE = HumanMouse(human_factor=0.9, default_duration=0.12, default_offset=5)


def random_mouse_click_interval_sec() -> float:
    """返回鼠标按下到抬起的随机间隔，单位秒。"""
    return random.uniform(0.08, 0.18)


def _remember_mouse_target_size(target_size: Optional[Tuple[int, int]]) -> None:
    global _LAST_MOUSE_TARGET_SIZE
    _LAST_MOUSE_TARGET_SIZE = target_size


def _adaptive_click_offset_px(target_size: Optional[Tuple[int, int]] = None) -> int:
    """按目标尺寸返回点击偏移上限像素。"""
    width, height = target_size or _LAST_MOUSE_TARGET_SIZE or (48, 24)
    is_small_target = max(width, height) <= 80 and (width * height) <= 5000
    return 3 if is_small_target else 5


def _move_duration_for_distance(x: int, y: int) -> float:
    sx, sy = pyautogui.position()
    distance = math.hypot(x - sx, y - sy)
    return min(0.16, max(0.05, distance / 1800.0))


def mouse_click(
    button: str = "left",
    clicks: int = 1,
    interval_sec: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """统一鼠标点击动作：按下 -> 随机等待 -> 抬起。"""
    if clicks <= 0:
        return

    offset_px = _adaptive_click_offset_px(target_size)
    press_duration = random_mouse_click_interval_sec()
    for idx in range(clicks):
        if button == "left":
            _HUMAN_MOUSE.click_left(offset=offset_px, press_duration=press_duration, move_duration=0.03)
        elif button == "right":
            _HUMAN_MOUSE.click_right(offset=offset_px, press_duration=press_duration, move_duration=0.03)
        elif button == "middle":
            _HUMAN_MOUSE.click_middle(offset=offset_px, press_duration=press_duration, move_duration=0.03)
        else:
            raise ValueError(f"Unsupported mouse button: {button}")
        if idx < clicks - 1:
            time.sleep(interval_sec if interval_sec is not None else random_mouse_click_interval_sec())


def human_move_to(x: int, y: int, target_size: Optional[Tuple[int, int]] = None) -> None:
    """使用 human_control.py 中的 HumanMouse 执行人类化移动。"""
    _remember_mouse_target_size(target_size)
    _HUMAN_MOUSE.move_to(x, y, duration=_move_duration_for_distance(x, y))


def mouse_drag_rel(dx: int, dy: int, duration: float = 0.5, button: str = "left") -> None:
    """使用 HumanMouse 执行相对拖动。"""
    start_x, start_y = pyautogui.position()
    target_x = start_x + dx
    target_y = start_y + dy

    if button == "left":
        _HUMAN_MOUSE.press_left()
    elif button == "right":
        _HUMAN_MOUSE.press_right()
    elif button == "middle":
        _HUMAN_MOUSE.press_middle()
    else:
        raise ValueError(f"Unsupported mouse button: {button}")

    time.sleep(random.uniform(0.08, 0.18))
    _HUMAN_MOUSE.move_to(target_x, target_y, duration=duration, curvature=35)

    if button == "left":
        _HUMAN_MOUSE.release_left()
    elif button == "right":
        _HUMAN_MOUSE.release_right()
    else:
        _HUMAN_MOUSE.release_middle()

    time.sleep(random.uniform(0.12, 0.3))


def _append_shared_ui_log_line(line: str):
    if not _SHARED_UI_LOG_FILE:
        return
    text = line.rstrip("\r")
    if not text:
        return
    with _SHARED_UI_LOG_LOCK:
        with open(_SHARED_UI_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{_SHARED_UI_LOG_PREFIX}{text}\n")


class _SharedUILogStream:
    def __init__(self, wrapped_stream):
        self.wrapped_stream = wrapped_stream
        self._buffer = ""
        self.encoding = getattr(wrapped_stream, "encoding", "utf-8")

    def write(self, data):
        if self.wrapped_stream:
            self.wrapped_stream.write(data)
        text = str(data)
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            _append_shared_ui_log_line(line)
        return len(text)

    def flush(self):
        if self.wrapped_stream:
            self.wrapped_stream.flush()
        if self._buffer:
            _append_shared_ui_log_line(self._buffer)
            self._buffer = ""

    def isatty(self):
        if self.wrapped_stream:
            return self.wrapped_stream.isatty()
        return False


def _install_shared_ui_log_streams():
    if not _SHARED_UI_LOG_FILE:
        return
    if not isinstance(sys.stdout, _SharedUILogStream):
        sys.stdout = _SharedUILogStream(sys.stdout)
    if not isinstance(sys.stderr, _SharedUILogStream):
        sys.stderr = _SharedUILogStream(sys.stderr)


_install_shared_ui_log_streams()


def is_process_running(process_name: str) -> bool:
    if not process_name:
        return False

    target = process_name.lower()
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            output = (result.stdout or "").lower()
            return target in output
        except Exception:
            return False

    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "comm="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return any(line.strip().lower().endswith(target) for line in result.stdout.splitlines())
    except Exception:
        return False


def is_game_exefile_running() -> bool:
    return is_process_running("exefile.exe")

class NoCnocrFilter(logging.Filter):
    """禁用 cnocr/cnstd 的 use model 噪音日志"""
    def filter(self, record):
        msg = record.getMessage().lower()
        return "use model" not in msg

logging.getLogger("").addFilter(NoCnocrFilter())
logging.getLogger("cnocr").setLevel(logging.ERROR)
logging.getLogger("cnstd").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("task.log", mode="a", encoding="utf-8"), logging.StreamHandler()],
    force=True,
)
# INFO 只写 task.log，不输出到终端；终端仅显示 WARNING 及以上
root = logging.getLogger()
for h in root.handlers:
    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
        h.setLevel(logging.WARNING)
        break

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
    width = 38
    line = "=" * width
    pad = "  "
    parts = ["", line, pad + f"{app_name} 已启动，开始加载..."]
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

    # 当作为子进程由 start.exe 启动时，通过环境变量显式要求把日志同步到 stdout，
    # 这样 start.exe 的 stdout 读取线程可以在控制台界面显示这些内容。
    try:
        if os.environ.get("EVGUARD_CHILD_LOG_TO_STDOUT") == "1":
            print(message, flush=True)
    except Exception:
        pass

    # flush 所有 handler，确保 task.log 等文件即时落盘
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
    human_move_to(250, 700)
    mouse_drag_rel(-30, 0, duration=0.5, button="left")

def hscrollscreen():
    """垂直滑动（可选）"""
    human_move_to(1600, 400)
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

def find_txt_ocr(txt: str, max_attempts=5, region=None, allow_scroll=False) -> bool:
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
                target_width = line["position"][1][0] - line["position"][0][0]
                target_height = line["position"][2][1] - line["position"][0][1]
                x = r[0] + line["position"][0][0] + target_width // 2
                y = r[1] + line["position"][0][1] + target_height // 2
                human_move_to(x, y, target_size=(target_width, target_height))
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

    template_threshold: float = 0.8  # 模板匹配阈值
    cnn_threshold: float = 0.9       # CNN pos 置信度阈值
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
    log_message("INFO", f"加载CNN模型...")
    return model


def _find_icon_candidates_one_shot(
    icon: str,
    region: Tuple[int, int, int, int],
    screen: np.ndarray,
    cfg: CnnConfig,
    do_adjust: bool,
    icon_dir: str,
    model_dir: str,
) -> Tuple[List[Dict], float]:
    """
    单次截图内：模板匹配 -> 去重 TopK -> CNN 验证，返回所有通过验证的候选。
    返回: (details_list, template_max_val)
    details_list 每项: {'icon_name', 'match_val', 'prob', 'position': (x,y) 屏幕坐标}
    """
    template_path = os.path.join(icon_dir, f"{icon}.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像无法加载: {template_path}")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2]
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    max_val = float(np.max(res)) if res.size else 0.0
    ys, xs = np.where(res >= cfg.template_threshold)
    if len(xs) == 0:
        return [], max_val

    cands = [(int(x), int(y), float(res[y, x])) for x, y in zip(xs, ys)]
    cands.sort(key=lambda t: t[2], reverse=True)

    min_dist = max(1, int(min(w, h) * float(cfg.dedup_ratio)))
    min_dist2 = min_dist * min_dist
    filtered = []
    for x, y, s in cands:
        keep = True
        for fx, fy, _ in filtered:
            if (x - fx) ** 2 + (y - fy) ** 2 <= min_dist2:
                keep = False
                break
        if keep:
            filtered.append((x, y, s))
        if len(filtered) >= int(cfg.topk):
            break

    model = _get_cnn_model(icon, model_dir=model_dir)
    r = adjust_region(region) if do_adjust else region
    details_list: List[Dict] = []

    for idx, (x0, y0, match_val) in enumerate(filtered):
        crop = screen[y0 : y0 + h, x0 : x0 + w]
        if crop.size == 0 or crop.shape[0] != h or crop.shape[1] != w:
            continue
        if cfg.debug_save:
            try:
                debug_path = os.path.join(DEBUG_DIR, f"{icon}_match_{idx}_s{match_val:.3f}.png")
                cv2.imwrite(debug_path, crop)
            except Exception:
                pass
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x_t = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            out = model(x_t)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        if prob >= cfg.cnn_threshold:
            px = x0 + r[0] + w // 2
            py = y0 + r[1] + h // 2
            details_list.append({
                "icon_name": icon,
                "match_val": match_val,
                "prob": prob,
                "position": (px, py),
                "target_size": (w, h),
            })

    return details_list, max_val


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

    screen = capture_screen_area(region, do_adjust=do_adjust)
    details_list, max_val = _find_icon_candidates_one_shot(
        icon, region, screen, cfg, do_adjust, icon_dir, model_dir
    )

    if len(details_list) == 0:
        log_message("INFO", f"[{icon}] template max_val={max_val:.4f}, thr={cfg.template_threshold:.4f}")
        return False

    d0 = details_list[0]
    log_message("INFO", f"[{icon}] cand#0 score={d0['match_val']:.4f} prob={d0['prob']:.4f} thr={cfg.cnn_threshold:.4f}")
    mx = d0["position"][0] + offset_x
    my = d0["position"][1] + offset_y
    if move:
        human_move_to(mx, my, target_size=d0.get("target_size"))
    return True


def find_icon_count_cnn(
    icon_name: str,
    max_attempts: int = 1,
    region=None,
    screen=None,
    cfg: Optional[CnnConfig] = None,
    threshold: Optional[float] = None,
    cnn_threshold: Optional[float] = None,
    icon_dir: str = ICON_DIR,
    model_dir: str = MODEL_DIR,
) -> Tuple[int, List[Dict]]:
    """
    统计检测到的图标数量（同一类型多个实例）：模板匹配 + CNN 验证，与 find_icon_cnn 共用同一套逻辑。
    返回: (count, details_list)
    details 每项: {'icon_name', 'match_val', 'prob', 'position': (x,y) 屏幕坐标}
    """
    if cfg is None:
        cfg = CnnConfig()
    if threshold is not None:
        cfg.template_threshold = threshold
    if cnn_threshold is not None:
        cfg.cnn_threshold = cnn_threshold

    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    for attempt in range(max_attempts):
        if screen is None:
            screen = capture_screen_area(region, do_adjust=do_adjust)
        details_list, max_val = _find_icon_candidates_one_shot(
            icon_name, region, screen, cfg, do_adjust, icon_dir, model_dir
        )
        if len(details_list) > 0:
            parts = [f"#{i+1} match={d['match_val']:.3f} prob={d['prob']:.3f}" for i, d in enumerate(details_list)]
            log_message(
                "INFO",
                f"[{icon_name}] 检测到 {len(details_list)} 个 "
                f"template_thr={cfg.template_threshold:.2f} "
                f"cnn_thr={cfg.cnn_threshold:.2f} | " + " | ".join(parts),
            )
            return len(details_list), details_list
        screen = None  # 下一轮重新截屏
    return 0, []


def find_icon_detailed_cnn(
    icon_name: str,
    max_attempts: int = 1,
    region=None,
    screen=None,
    cfg: Optional[CnnConfig] = None,
) -> Tuple[bool, Dict]:
    """兼容旧逻辑：返回 (found, first_detail)，detail 含 icon_name/match_val/prob/position 或 found=False"""
    cnt, details = find_icon_count_cnn(icon_name, max_attempts=max_attempts, region=region, screen=screen, cfg=cfg)
    if cnt > 0:
        return True, details[0]
    return False, {"icon_name": icon_name, "match_val": 0, "prob": 0.0, "position": None, "found": False}


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
                mouse_click(button="left")
            elif action in ("rightclick", "right"):
                mouse_click(button="right")
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
            if safe_find_icon("warp1", control_panel, max_attempts=3):
                log_message("INFO", "找到warp1，等待3秒后再次点击")
                mouse_click(clicks=2)
                time.sleep(3)
                mouse_click(clicks=2)  # 再次点击warp1
                time.sleep(1)
                # 再点击jump2
                log_message("INFO", "查找并点击jump2")
                if safe_find_icon("jump2", region_full_right, max_attempts=3,threshold=0.9,cnn_threshold=0.85):
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
