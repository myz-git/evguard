import sys
import argparse
import numpy as np
import pyautogui
import time
import random
import cv2
import os
import pynput
from PIL import Image
from datetime import datetime

from pynput import keyboard
from license_utils import ensure_license_or_exit



# 复用公共 utils（CNN + OCR + 截图 + 区域配置）
from utils import (
    log_message,
    speak,
    play_sound_wav,
    mouse_click,
    human_move_to,
    print_startup,
    capture_screen_area,
    adjust_region,
    find_txt_ocr,
    find_txt_ocr3,
    locate_txt_ocr,
    get_shared_ocr,
    screen_regions,
    ICON_DIR,
    CnnConfig,
    find_icon_count_cnn,
    find_icon_detailed_cnn,
    is_game_exefile_running,
    save_image_cv,
    log_screen_region_layout,
)

ensure_license_or_exit()

# 预警模式:
# A=低安(只报警不规避)
# B=高安(触发后规避)
# C=低安主动防御(触发后规避)
MODE_A_LOWSEC = 'A'
MODE_B_HIGHSEC = 'B'
MODE_C_LOWSEC_ACTIVE = 'C'
# 正常扫描间隔（秒）
TARGET_INTERVAL_MIN_SEC = 3.0
TARGET_INTERVAL_MAX_SEC = 5.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(BASE_DIR, "evguard.cfg")
DEFAULT_RUNTIME_CFG = {
    "location_hotkey": "l",
    "dscan_enabled": False,
    "dscan_hotkey": "middle",
    "deep_safe_a": "PIN999",
    "deep_safe_b": "PIN888",
    "alarm_volume": "mid",
    "alarm_sound": "static/soundmid.wav",
    "general_template_threshold": 0.86,
    "general_cnn_threshold": 0.9,
    "zhongli_template_threshold": 0.86,
    "zhongli_cnn_threshold": 0.9,
    "suspect_template_threshold": 0.86,
    "suspect_cnn_threshold": 0.95,
    "didui_template_threshold": 0.9,
    "didui_cnn_threshold": 0.95,
}


class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass


def load_runtime_cfg():
    cfg = dict(DEFAULT_RUNTIME_CFG)
    if not os.path.exists(CFG_FILE):
        return cfg
    try:
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                cfg[key.strip().lower()] = value.strip()
    except Exception:
        return cfg

    def _get_float(name):
        default = float(DEFAULT_RUNTIME_CFG[name])
        try:
            return float(cfg.get(name, default))
        except (TypeError, ValueError):
            return default

    cfg["location_hotkey"] = str(cfg.get("location_hotkey", DEFAULT_RUNTIME_CFG["location_hotkey"])).strip().lower() or DEFAULT_RUNTIME_CFG["location_hotkey"]
    cfg["dscan_enabled"] = str(cfg.get("dscan_enabled", DEFAULT_RUNTIME_CFG["dscan_enabled"])).strip().lower() in {"1", "true", "yes", "on"}
    cfg["dscan_hotkey"] = str(cfg.get("dscan_hotkey", cfg.get("scankey", DEFAULT_RUNTIME_CFG["dscan_hotkey"]))).strip().lower() or DEFAULT_RUNTIME_CFG["dscan_hotkey"]
    cfg["deep_safe_a"] = str(cfg.get("deep_safe_a", DEFAULT_RUNTIME_CFG["deep_safe_a"])).strip() or DEFAULT_RUNTIME_CFG["deep_safe_a"]
    cfg["deep_safe_b"] = str(cfg.get("deep_safe_b", DEFAULT_RUNTIME_CFG["deep_safe_b"])).strip() or DEFAULT_RUNTIME_CFG["deep_safe_b"]
    cfg["general_template_threshold"] = _get_float("general_template_threshold")
    cfg["general_cnn_threshold"] = _get_float("general_cnn_threshold")
    cfg["zhongli_template_threshold"] = _get_float("zhongli_template_threshold")
    cfg["zhongli_cnn_threshold"] = _get_float("zhongli_cnn_threshold")
    cfg["suspect_template_threshold"] = _get_float("suspect_template_threshold")
    cfg["suspect_cnn_threshold"] = _get_float("suspect_cnn_threshold")
    cfg["didui_template_threshold"] = _get_float("didui_template_threshold")
    cfg["didui_cnn_threshold"] = _get_float("didui_cnn_threshold")
    volume = str(cfg.get("alarm_volume", DEFAULT_RUNTIME_CFG["alarm_volume"])).strip().lower()
    sound_map = {
        "high": "static/soundhigh.wav",
        "mid": "static/soundmid.wav",
        "low": "static/soundlow.wav",
    }
    cfg["alarm_volume"] = volume if volume in sound_map else DEFAULT_RUNTIME_CFG["alarm_volume"]
    cfg["alarm_sound"] = sound_map[cfg["alarm_volume"]]
    return cfg


def trigger_dscan(hotkey):
    key = str(hotkey or "").strip().lower()
    if key in {"left", "middle", "right"}:
        mouse_click(button=key)
        return
    if key:
        pyautogui.press(key)


def emergency_evasion(reason):
    """
    紧急规避函数
    reason: 触发紧急规避的原因（字符串）
    """
    ctr = pynput.keyboard.Controller()
    with ctr.pressed(pynput.keyboard.Key.ctrl, 's'):
        print(f"[警报] {reason}")
        speak("执行自动导航")
        time.sleep(0.1)
        pass
    sys.exit(1)

# 全局变量，用于控制程序是否继续运行
running = True

# 记录 Ctrl 键是否被按下
ctrl_pressed = False

# 调试模式：是否保存误识别截图
DEBUG_MODE = False
DEBUG_SAVE_DIR = "debug_icons"

#一般模板匹配阈值
GENERAL_TEMPLATE_THRESHOLD = 0.86
#中立模板匹配阈值
# ZHONGLI_TEMPLATE_THRESHOLD = 0.70
#罪犯/嫌犯模板匹配阈值
SUSPECT_TEMPLATE_THRESHOLD = 0.86
#敌对模板匹配阈值
DIDUI_TEMPLATE_THRESHOLD = 0.9

ICON_CONFIRM_DELAY_SEC = 0.3

_IMPORT_RUNTIME_CFG = load_runtime_cfg()
GENERAL_TEMPLATE_THRESHOLD = _IMPORT_RUNTIME_CFG["general_template_threshold"]
GENERAL_CNN_THRESHOLD = _IMPORT_RUNTIME_CFG["general_cnn_threshold"]
ZHONGLI_TEMPLATE_THRESHOLD = _IMPORT_RUNTIME_CFG["zhongli_template_threshold"]
ZHONGLI_CNN_THRESHOLD = _IMPORT_RUNTIME_CFG["zhongli_cnn_threshold"]
SUSPECT_TEMPLATE_THRESHOLD = _IMPORT_RUNTIME_CFG["suspect_template_threshold"]
SUSPECT_CNN_THRESHOLD = _IMPORT_RUNTIME_CFG["suspect_cnn_threshold"]
DIDUI_TEMPLATE_THRESHOLD = _IMPORT_RUNTIME_CFG["didui_template_threshold"]
DIDUI_CNN_THRESHOLD = _IMPORT_RUNTIME_CFG["didui_cnn_threshold"]

ICON_DETECTION_SPECS = [
    {
        "icon": "zhongli",
        "icons": ["zhongli", "zhonglis"],
        "label": "中立",
        "modes": {MODE_A_LOWSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": ZHONGLI_TEMPLATE_THRESHOLD,
        "cnn_threshold": ZHONGLI_CNN_THRESHOLD,
    },
    {
        "icon": "zuifan",
        "label": "罪犯",
        "modes": {MODE_A_LOWSEC, MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": SUSPECT_TEMPLATE_THRESHOLD,
        "cnn_threshold": SUSPECT_CNN_THRESHOLD,
    },
    {
        "icon": "jisha",
        "label": "击杀",
        "modes": {MODE_A_LOWSEC, MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": GENERAL_TEMPLATE_THRESHOLD,
        "cnn_threshold": GENERAL_CNN_THRESHOLD,
    },
    {
        "icon": "xianfan",
        "label": "嫌犯",
        "modes": {MODE_A_LOWSEC, MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": SUSPECT_TEMPLATE_THRESHOLD,
        "cnn_threshold": SUSPECT_CNN_THRESHOLD,
    },
    {
        "icon": "didui",
        "label": "敌对",
        "modes": {MODE_A_LOWSEC, MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": DIDUI_TEMPLATE_THRESHOLD,
        "cnn_threshold": DIDUI_CNN_THRESHOLD,
    },
    {
        "icon": "buliang",
        "label": "不良",
        "modes": set(),
        "threshold": GENERAL_TEMPLATE_THRESHOLD,
        "cnn_threshold": GENERAL_CNN_THRESHOLD,
    },
    {
        "icon": "zaogao",
        "icons": ["zaogao", "zaogaos"],
        "label": "糟糕",
        "modes": {MODE_A_LOWSEC, MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": GENERAL_TEMPLATE_THRESHOLD,
        "cnn_threshold": GENERAL_CNN_THRESHOLD,
    },
    {
        "icon": "wuwei",
        "label": "鏃犵晱",
        "modes": {MODE_A_LOWSEC, MODE_C_LOWSEC_ACTIVE},
        "threshold": GENERAL_TEMPLATE_THRESHOLD,
        "cnn_threshold": GENERAL_CNN_THRESHOLD,
    },
]


def _detect_icons_once(mode, region, screen, cfg):
    hits = []
    for spec in ICON_DETECTION_SPECS:
        if mode not in spec["modes"]:
            continue
        icon_cfg = CnnConfig(
            template_threshold=cfg.template_threshold,
            cnn_threshold=cfg.cnn_threshold,
            topk=cfg.topk,
            dedup_ratio=cfg.dedup_ratio,
            debug_save=cfg.debug_save,
        )
        count = 0
        details_list = []
        for icon_name in spec.get("icons", [spec["icon"]]):
            icon_count, icon_details = find_icon_count_cnn(
                icon_name,
                max_attempts=1,
                region=region,
                screen=screen,
                cfg=icon_cfg,
                threshold=spec.get("threshold"),
                cnn_threshold=spec.get("cnn_threshold"),
            )
            count += icon_count
            details_list.extend(icon_details)
        if count > 0:
            hits.append({"spec": spec, "count": count, "details": details_list})
    return hits


def _confirm_icon_hits(mode, region, cfg, first_hits):
    if not first_hits:
        return []

    for hit in first_hits:
        best_match = max((detail.get("match_val", 0.0) for detail in hit["details"]), default=0.0)
        print(f"[检测] {hit['spec']['label']} 命中待确认 1/2 match={best_match:.3f}")

    time.sleep(ICON_CONFIRM_DELAY_SEC)
    confirm_screen = capture_screen_area(region)
    confirmed_hits = []
    first_icons = {hit["spec"]["icon"] for hit in first_hits}
    confirm_hits = _detect_icons_once(mode, region, confirm_screen, cfg)
    confirm_map = {hit["spec"]["icon"]: hit for hit in confirm_hits if hit["spec"]["icon"] in first_icons}

    for hit in first_hits:
        icon_name = hit["spec"]["icon"]
        confirmed = confirm_map.get(icon_name)
        if not confirmed:
            print(f"[检测] {hit['spec']['label']} 0.5秒复检未通过")
            continue
        best_match = max((detail.get("match_val", 0.0) for detail in confirmed["details"]), default=0.0)
        print(f"[检测] {hit['spec']['label']} 0.5秒复检通过 match={best_match:.3f}")
        confirmed_hits.append(confirmed)

    return confirmed_hits

def on_press(key):
    global running, ctrl_pressed
    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            ctrl_pressed = True
        if key == keyboard.Key.f12 and ctrl_pressed:
            running = False
            print("Ctrl+F12 pressed, stopping the program.")
            return False  # Stop the listener
    except AttributeError:
        pass

def on_release(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False

ctrl_pressed = False

def save_debug_image(icon_name, icon_image, match_val, model_prediction, is_correct):
    """保存调试截图"""
    if not os.path.exists(DEBUG_SAVE_DIR):
        os.makedirs(DEBUG_SAVE_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    status = "correct" if is_correct else "false_positive"
    filename = f"{icon_name}_{status}_match{match_val:.3f}_model{model_prediction}_{timestamp}.png"
    filepath = os.path.join(DEBUG_SAVE_DIR, filename)
    save_image_cv(filepath, icon_image)
    print(f"[调试] 已保存截图: {filepath}")

def find_txt_ocr3_debug(txt, max_attempts=1, region=None):
    """
    带调试功能的OCR文字识别
    返回: (found, all_ocr_texts)
    """
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    all_ocr_texts = []
    ocr = get_shared_ocr()

    while attempts < max_attempts:
        screen_image = pyautogui.screenshot(region=region)
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像
        
        # 收集所有识别到的文字
        current_texts = []
        for line in res:
            text = line['text']
            current_texts.append(text)
            all_ocr_texts.append(text)
            
            # 检查是否包含目标文字
            if txt in text:
                target_width = line['position'][1][0] - line['position'][0][0]
                target_height = line['position'][2][1] - line['position'][0][1]
                x = region[0] + line['position'][0][0] + target_width // 2
                y = region[1] + line['position'][0][1] + target_height // 2
                human_move_to(x, y, target_size=(target_width, target_height))
                # print(f"[OCR] 找到文字 {txt} 在位置 ({x}, {y})")
                return True, ' | '.join(all_ocr_texts)
        
        attempts += 1
        time.sleep(0.5)
    
    # 返回所有识别到的文字，用 | 分隔
    return False, ' | '.join(all_ocr_texts) if all_ocr_texts else ''

def main(mode=MODE_A_LOWSEC):
    """
    mode: 'A' 低安预警 - 只报警不规避，多监控「中立声望」
          'B' 高安预警 - 触发后进行规避
          'C' 低安主动防御 - 任意图标触发后进行规避
    """
    global running


    if mode == MODE_A_LOWSEC:
        app_name = "GuardA"
        mode_desc = "低安被动预警(无雷达检测,不自动规避)"
    elif mode == MODE_B_HIGHSEC:
        app_name = "GuardB"
        mode_desc = "高安主动预警(雷达主动检测,可自动规避)"
    elif mode == MODE_C_LOWSEC_ACTIVE:
        app_name = "GuardC"
        mode_desc = "低安主动防御(雷达主动检测,任意目标触发即规避)"

    print_startup(app_name, ["按 Ctrl+F12 可停止程序", f"模式: {mode_desc}"])
    runtime_cfg = load_runtime_cfg()
    
    

    # 开始监听键盘事件
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    if not is_game_exefile_running():
        print("游戏未启动, 请先启动游戏")
        return 1
    print("游戏检测中...请稍后...")
    log_screen_region_layout(f"{app_name}区域布局", ["left_panel", "center_panel", "center_panel2", "right_panel"])
    played_start_sound = False
    while running:
        if not played_start_sound:
            play_sound_wav("static/Login_Connecting.wav")
            played_start_sound = True
        loop_start = time.time()
        # CNN 模式：无需加载 joblib/scaler，直接使用 icon/ + model_cnn/

        # 设置需要捕获的屏幕区域
        fx, fy = pyautogui.size()
        left_panel = screen_regions['left_panel']
        center_panel = screen_regions['center_panel']
        center_panel2 = screen_regions['center_panel2']
        right_panel = screen_regions['right_panel']


        try:
            # A 模式：不做任何键鼠操作；B 模式可触发雷达扫描（鼠标中键）
            play_sound_wav("static/faction.wav")
            # if mode in {MODE_B_HIGHSEC, MODE_C_LOWSEC_ACTIVE}:
            # if mode in {MODE_B_HIGHSEC}:
            if mode == MODE_B_HIGHSEC and runtime_cfg.get("dscan_enabled", DEFAULT_RUNTIME_CFG["dscan_enabled"]):
                # 通过 evguard.cfg 配置 D-SCAN 触发方式。
                trigger_dscan(runtime_cfg.get("dscan_hotkey"))
                print("雷达扫描中...")
                
            else:
                print("扫描中...")


            total_icon_count = 0
            icon_details_summary = []

            # 每轮只截屏一次，供所有图标检测共用，缩短 A 模式间隔（否则 A 多一次截屏+匹配约 5s）
            screen = capture_screen_area(right_panel)

            # 默认阈值统一在 utils.CnnConfig 中配置，此处用默认即可；特殊图标（如敌对）在调用时传 cnn_threshold 等覆盖
            cfg_icon = CnnConfig()
            first_icon_hits = _detect_icons_once(mode, right_panel, screen, cfg_icon)
            confirmed_icon_hits = _confirm_icon_hits(mode, right_panel, cfg_icon, first_icon_hits)
            for hit in confirmed_icon_hits:
                total_icon_count += hit["count"]
                for detail in hit["details"]:
                    icon_details_summary.append((hit["spec"]["label"], detail))
                    if hit["spec"]["icon"] == "didui":
                        print(
                            f"[检测] 敌对声望 - 匹配值: {detail['match_val']:.3f}, "
                            f"模型预测: {detail.get('prob', 0)}, "
                            f"位置: {detail['position']}"
                        )

            icon_found = total_icon_count > 0

            # 文字识别：「促进」。仅 B 模式检测，A/C 模式不检测促进
            if mode == MODE_B_HIGHSEC:
                suspect_txt_count = find_txt_ocr3("促进", 1, right_panel)
                suspect_txt_found = suspect_txt_count >= 2
            else:
                suspect_txt_count = 0
                suspect_txt_found = False
            if mode in {MODE_A_LOWSEC, MODE_C_LOWSEC_ACTIVE}:
                wuwei_txt_count = find_txt_ocr3("无畏", 1, right_panel)
                wuwei_txt_found = wuwei_txt_count >= 1
            else:
                wuwei_txt_count = 0
                wuwei_txt_found = False

            txt_count = suspect_txt_count + wuwei_txt_count
            txt_found = suspect_txt_found or wuwei_txt_found
            total_danger_count = total_icon_count + txt_count

            # 任一触发则声音告警
            if icon_found or txt_found:
                play_sound_wav(runtime_cfg.get("alarm_sound", DEFAULT_RUNTIME_CFG["alarm_sound"]))
                if icon_found:
                    icon_summary = {}
                    for name, _ in icon_details_summary:
                        icon_summary[name] = icon_summary.get(name, 0) + 1
                    parts = [f"{c}个 {n}单位" for n, c in icon_summary.items()]
                    print(f"[警报] 发现: {'、'.join(parts)}")
                if suspect_txt_found:
                    print(f"[警报] 发现 {suspect_txt_count} 个可疑舰船")
                if wuwei_txt_found:
                    print(f"[警报] 识别到 {wuwei_txt_count} 处无畏文字")

            # B模式：达到 2 个及以上危险项时紧急规避
            if mode == MODE_B_HIGHSEC and total_danger_count >= 2:
                danger_items = []
                if total_icon_count > 0:
                    icon_summary = {}
                    for name, _ in icon_details_summary:
                        icon_summary[name] = icon_summary.get(name, 0) + 1
                    icon_str = ', '.join([f"{name}{count}个" if count > 1 else name for name, count in icon_summary.items()])
                    danger_items.append(f"图标{icon_str}(共{total_icon_count}个)")
                if suspect_txt_count > 0:
                    danger_items.append(f"可疑舰船{suspect_txt_count}个")
                reason = f"发现{'、'.join(danger_items)}"
                #emergency_evasion(reason)
                speak("紧急规避,3秒后启动")
                print("紧急规避,3秒后启动")
                time.sleep(3)
                emergency_evade_pin999(
                    center_panel2,
                    open_key=runtime_cfg.get("location_hotkey", DEFAULT_RUNTIME_CFG["location_hotkey"]),
                    pin_text=runtime_cfg.get("deep_safe_a", DEFAULT_RUNTIME_CFG["deep_safe_a"]),
                    backup_pin_text=runtime_cfg.get("deep_safe_b", DEFAULT_RUNTIME_CFG["deep_safe_b"]),
                )
            elif mode == MODE_C_LOWSEC_ACTIVE and (total_icon_count >= 1 or wuwei_txt_found):
                danger_items = []
                if total_icon_count > 0:
                    icon_summary = {}
                    for name, _ in icon_details_summary:
                        icon_summary[name] = icon_summary.get(name, 0) + 1
                    icon_str = ', '.join([f"{name}{count}个" if count > 1 else name for name, count in icon_summary.items()])
                    danger_items.append(f"图标{icon_str}(共{total_icon_count}个)")
                if wuwei_txt_found:
                    danger_items.append(f"无畏文字{wuwei_txt_count}处")
                reason = f"发现{'、'.join(danger_items)}"
                speak("紧急规避,3秒后启动")
                print("紧急规避,3秒后启动")
                time.sleep(3)
                emergency_evade_pin999(
                    center_panel2,
                    open_key=runtime_cfg.get("location_hotkey", DEFAULT_RUNTIME_CFG["location_hotkey"]),
                    pin_text=runtime_cfg.get("deep_safe_a", DEFAULT_RUNTIME_CFG["deep_safe_a"]),
                    backup_pin_text=runtime_cfg.get("deep_safe_b", DEFAULT_RUNTIME_CFG["deep_safe_b"]),
                )

            if icon_found or txt_found:
                time.sleep(2)

        except IconNotFoundException as e:
            print(e)

        elapsed = time.time() - loop_start
        target_interval_sec = random.uniform(TARGET_INTERVAL_MIN_SEC, TARGET_INTERVAL_MAX_SEC)
        time.sleep(max(0, target_interval_sec - elapsed))

    listener.join()
    play_sound_wav("static/Notification_Ping.wav")
    return 0


def emergency_evade_pin999(
    place_region=None,
    open_key='l',
    pin_text='PIN999',
    backup_pin_text='PIN888',
    action_text='带领舰队',
    fallback_action_text='跃迁至',
):
    """
    紧急规避（新）：
      1) 按 l 打开“地点”面板
      2) 找到 PIN999（可能多个，取第一个），鼠标移上去，右键
      3) 在右键菜单优先找“带领舰队”（可能多个，取第一个）并点击；
         若失败，再找“跳跃至”（取第一个）并点击；
         若仍失败，回退 emergency_evasion()（Ctrl+S）并退出

    依赖：
      - speak(text)
      - emergency_evasion(reason)  # 内部应执行 Ctrl+S 并退出
      - find_txt_ocr(text, 1, region)  # 找到后会把鼠标移动到文字上，并返回 True/False
      - pyautogui
      - time, sys
    """


    # 1) 先聚焦屏幕中部，再打开地点面板
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    human_move_to(center_x, center_y, target_size=(160, 120))
    mouse_click(button='left')
    time.sleep(0.08)

    # 2) 打开地点面板
    pyautogui.press(open_key)
    time.sleep(0.2)

    # 3) 只扫必要区域，避免重复查找和全屏误扫。
    def _contains_region(outer, inner):
        if not outer or not inner:
            return False
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih

    search_regions = []
    for region in (
        place_region,
        screen_regions.get('center_panel2'),
        screen_regions.get('center_panel'),
    ):
        if not region or region in search_regions:
            continue
        if any(_contains_region(existing, region) for existing in search_regions):
            continue
        search_regions.append(region)

    ok = False
    matched_pin = None
    matched_region = None
    matched_location = None

    def _try_pin_regions(regions):
        nonlocal ok, matched_pin, matched_region, matched_location
        for region in regions:
            location = locate_txt_ocr(pin_text, 1, region, allow_scroll=False, log_miss=False)
            if location:
                ok = True
                matched_pin = pin_text
                matched_region = region
                matched_location = location
                return True
            location = locate_txt_ocr(backup_pin_text, 1, region, allow_scroll=False, log_miss=False) if backup_pin_text else None
            if location:
                ok = True
                matched_pin = backup_pin_text
                matched_region = region
                matched_location = location
                return True
        return False

    _try_pin_regions(search_regions)

    if not ok:
        log_message("INFO", f"安全点查找失败: pin_a={pin_text}, pin_b={backup_pin_text}, place_region={place_region}, search_regions={search_regions}")
        speak("未找到安全点")
        emergency_evasion('未找到安全点')
        return

    log_message("INFO", f"安全点命中: pin={matched_pin}, region={matched_region}")
    human_move_to(matched_location["x"], matched_location["y"], target_size=matched_location["target_size"])
    time.sleep(0.1)

    mouse_click(button='right')
    time.sleep(0.2)

    # 3) 在右键菜单区域优先找“带领舰队”，失败则找“跳跃至”
    mx, my = pyautogui.position()
    menu_regions = [
        (max(mx - 40, 0), max(my - 10, 0), 520, 560),   # 常规：鼠标右侧/下方
        (max(mx - 520, 0), max(my - 10, 0), 520, 560),  # 兜底：菜单翻到左侧
    ]

    def _find_and_click_first(menu_text: str) -> bool:
        for r in menu_regions:
            location = locate_txt_ocr(menu_text, 1, r, allow_scroll=False, log_miss=False)
            if location:
                human_move_to(location["x"], location["y"], target_size=location["target_size"])
                time.sleep(0.2)
                mouse_click(button='left')
                time.sleep(0.2)
                return True
        return False

    if _find_and_click_first(action_text):
        print("紧急规避已启动,程序运行结束")
        speak("紧急规避已启动,程序运行结束")
        play_sound_wav("static/Notification_Ping.wav")
        sys.exit(1)

    if _find_and_click_first(fallback_action_text):
        speak("紧急规避已启动,程序运行结束")
        print("紧急规避已启动,程序运行结束")
        play_sound_wav("static/Notification_Ping.wav")
        sys.exit(1)

    # 都失败：回退
    emergency_evasion('规避失败,未找到跳跃目标,执行自动导航')





def run(mode):
    """运行入口：mode='A' or 'B'"""
    return main(mode)
