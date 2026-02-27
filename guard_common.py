import sys
import argparse
import numpy as np
import pyautogui
import time
import cv2
import os
import pynput
import torch
from PIL import Image
from datetime import datetime

from pynput import keyboard



# 复用公共 utils（CNN + OCR + 截图 + 区域配置）
from utils import (
    log_message,
    speak,
    play_sound_wav,
    print_startup,
    capture_screen_area,
    adjust_region,
    find_txt_ocr,
    find_txt_ocr3,
    screen_regions,
    ICON_DIR,
    CnnConfig,
    _get_cnn_model,
    _TRANSFORM,
    _DEVICE,
)

########################################################
# license 授权验证
from license_verify import check_license_or_trial, get_request_code

st = check_license_or_trial()
if not st.ok:
    print(st.message)
    print("RequestCode:", get_request_code())
    raise SystemExit(2)

if st.ok and st.mode == "licensed":
    exp_str = st.exp.date().isoformat() if st.exp else "未知"
    print(f"license已授权,到期时间 {exp_str}")
elif st.ok and st.mode == "trial":
    print(f"未授权,30天试用")
    exp_str = st.exp.date().isoformat() if st.exp else "未知"
    if st.days_left <= 5:
        print(f"试用模式,到期时间 {exp_str}（剩余 {st.days_left} 天）")
else:
    # 失败逻辑
    print(st.message)
########################################################

# 预警模式: A=低安(只报警不规避), B=高安(触发后规避)
MODE_A_LOWSEC = 'A'
MODE_B_HIGHSEC = 'B'
# 目标扫描间隔（秒），B 补足到此间隔，A 通过隔轮做 OCR 尽量接近
TARGET_INTERVAL_SEC = 5.0


class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass


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
# 警告阈值：匹配值超过此值但未通过验证时显示警告（0.7表示70%相似度）
WARNING_THRESHOLD = 0.75

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

def _softmax_pos_prob(logits: torch.Tensor) -> float:
    """logits shape (1,2) -> pos prob"""
    return torch.softmax(logits, dim=1)[0, 1].item()


def find_icon_count_cnn(
    icon_name: str,
    max_attempts: int = 1,
    region=None,
    screen=None,
    cfg: CnnConfig | None = None,
):
    """
    统计检测到的图标数量（同一类型多个实例）
    - 先用模板匹配定位候选
    - 再用 CNN 模型验证候选是否为目标
    返回: (count, details_list)
    details: {icon_name, match_val, prob, position(x,y)}
    """
    if cfg is None:
        cfg = CnnConfig()

    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        do_adjust = False
    else:
        do_adjust = True

    # 模板
    template_path = os.path.join(ICON_DIR, f"{icon_name}-1.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像无法加载: {template_path}")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2]

    attempts = 0
    while attempts < max_attempts:
        if screen is None:
            screen = capture_screen_area(region, do_adjust=do_adjust)

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        ys, xs = np.where(res >= cfg.template_threshold)
        if len(xs) == 0:
            attempts += 1
            continue

        # 候选点 + 分数（降序）
        cands = [(int(x), int(y), float(res[y, x])) for x, y in zip(xs, ys)]
        cands.sort(key=lambda t: t[2], reverse=True)

        # 去重 + TopK
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

        model = _get_cnn_model(icon_name)
        r = adjust_region(region) if do_adjust else region

        found_count = 0
        details_list = []

        for x0, y0, match_val in filtered:
            crop = screen[y0:y0 + h, x0:x0 + w]
            if crop.size == 0 or crop.shape[0] != h or crop.shape[1] != w:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)

            with torch.no_grad():
                logits = model(x)
                prob = _softmax_pos_prob(logits)

            if prob >= cfg.cnn_threshold:
                px = x0 + r[0] + w // 2
                py = y0 + r[1] + h // 2
                details_list.append({
                    'icon_name': icon_name,
                    'match_val': match_val,
                    'prob': prob,
                    'position': (px, py),
                })
                found_count += 1

        if found_count > 0:
            return found_count, details_list

        attempts += 1

    return 0, []


def find_icon_detailed_cnn(icon_name: str, max_attempts=1, region=None, screen=None, cfg: CnnConfig | None = None):
    """兼容旧逻辑：返回 (found, first_detail)"""
    cnt, details = find_icon_count_cnn(icon_name, max_attempts=max_attempts, region=region, screen=screen, cfg=cfg)
    if cnt > 0:
        return True, details[0]
    return False, {'icon_name': icon_name, 'match_val': 0, 'prob': 0.0, 'position': None, 'found': False}

ctrl_pressed = False

def save_debug_image(icon_name, icon_image, match_val, model_prediction, is_correct):
    """保存调试截图"""
    if not os.path.exists(DEBUG_SAVE_DIR):
        os.makedirs(DEBUG_SAVE_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    status = "correct" if is_correct else "false_positive"
    filename = f"{icon_name}_{status}_match{match_val:.3f}_model{model_prediction}_{timestamp}.png"
    filepath = os.path.join(DEBUG_SAVE_DIR, filename)
    cv2.imwrite(filepath, icon_image)
    print(f"[调试] 已保存截图: {filepath}")

def find_txt_ocr3_debug(txt, max_attempts=1, region=None):
    """
    带调试功能的OCR文字识别
    返回: (found, all_ocr_texts)
    """
    from cnocr import CnOcr
    
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    all_ocr_texts = []
    
    while attempts < max_attempts:        
        # 初始化OCR工具
        ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
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
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
                pyautogui.moveTo(x, y)
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
    """
    global running


    if mode == MODE_A_LOWSEC:
        app_name = "GuardA"
        mode_desc = "低安被动预警(无雷达检测,不自动规避)"
    elif mode == MODE_B_HIGHSEC:
        app_name = "GuardB"
        mode_desc = "高安主动预警(雷达主动检测,可自动规避)"

    print_startup(app_name, ["按 Ctrl+F12 可停止程序", f"模式: {mode_desc}"])
    play_sound_wav("static/Login_Connecting.wav")


    # 开始监听键盘事件
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while running:
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
            if mode == MODE_B_HIGHSEC:
                # 使用鼠标中键触发雷达扫描：按住0.1秒再松开
                pyautogui.mouseDown(button='middle')
                print("雷达扫描中...")
                time.sleep(0.1)
                pyautogui.mouseUp(button='middle')

            total_icon_count = 0
            icon_details_summary = []

            # 每轮只截屏一次，供所有图标检测共用，缩短 A 模式间隔（否则 A 多一次截屏+匹配约 5s）
            screen = capture_screen_area(right_panel)

            cfg_icon = CnnConfig(template_threshold=0.75, cnn_threshold=0.80, topk=30, dedup_ratio=0.5, debug_save=False)


            # A模式：检测「中立声望」
            if mode == MODE_A_LOWSEC:
                zhongli_count, zhongli_details_list = find_icon_count_cnn("zhongli", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
                if zhongli_count > 0:
                    total_icon_count += zhongli_count
                    for detail in zhongli_details_list:
                        icon_details_summary.append(("中立", detail))
                        # print(f"[检测] 中立声望 - 匹配值: {detail['match_val']:.3f}, "
                        #       f"模型预测: {detail.get('prob', 0)}, "
                        #       f"位置: {detail['position']}")

            # 检测「罪犯声望」
            zuifan_count, zuifan_details_list = find_icon_count_cnn("zuifan", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
            if zuifan_count > 0:
                total_icon_count += zuifan_count
                for detail in zuifan_details_list:
                    icon_details_summary.append(("罪犯", detail))
                    # print(f"[检测] 罪犯声望 - 匹配值: {detail['match_val']:.3f}, "
                    #       f"模型预测: {detail.get('prob', 0)}, "
                    #       f"位置: {detail['position']}")

            # 检测「击杀权限」
            jisha_count, jisha_details_list = find_icon_count_cnn("jisha", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
            if jisha_count > 0:
                total_icon_count += jisha_count
                for detail in jisha_details_list:
                    icon_details_summary.append(("击杀", detail))
                    # print(f"[检测] 击杀权限 - 匹配值: {detail['match_val']:.3f}, "
                    #       f"模型预测: {detail.get('prob', 0)}, "
                    #       f"位置: {detail['position']}")

            # 检测「嫌犯声望」
            xianfan_count, xianfan_details_list = find_icon_count_cnn("xianfan", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
            if xianfan_count > 0:
                total_icon_count += xianfan_count
                for detail in xianfan_details_list:
                    icon_details_summary.append(("嫌犯", detail))
                    # print(f"[检测] 嫌犯声望 - 匹配值: {detail['match_val']:.3f}, "
                    #       f"模型预测: {detail.get('prob', 0)}, "
                    #       f"位置: {detail['position']}")


            # 检测「敌对」
            didui_count, didui_details_list = find_icon_count_cnn("didui", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
            if didui_count > 0:
                total_icon_count += didui_count
                for detail in didui_details_list:
                    icon_details_summary.append(("敌对", detail))

            # 检测「不良」
            buliang_count, buliang_details_list = find_icon_count_cnn("buliang", max_attempts=1, region=right_panel, screen=screen, cfg=cfg_icon)
            if buliang_count > 0:
                total_icon_count += buliang_count
                for detail in buliang_details_list:
                    icon_details_summary.append(("不良", detail))

            icon_found = total_icon_count > 0

            # 文字识别：「促进」。仅 B 模式检测，A 模式不检测促进
            if mode == MODE_A_LOWSEC:
                txt_count = 0
                txt_found = False
            else:
                txt_count = find_txt_ocr3("促进", 1, right_panel)
                txt_found = txt_count >= 2

            total_danger_count = total_icon_count + txt_count

            # 任一触发则声音告警
            if icon_found or txt_found:
                play_sound_wav('static/sound.wav')
                if icon_found:
                    icon_summary = {}
                    for name, _ in icon_details_summary:
                        icon_summary[name] = icon_summary.get(name, 0) + 1
                    parts = [f"{c}个 {n}单位" for n, c in icon_summary.items()]
                    print(f"[警报] 发现: {'、'.join(parts)}")
                if txt_found:
                    print(f"[警报] 发现 {txt_count} 个可疑舰船")

            # B模式：达到 2 个及以上危险项时紧急规避
            if mode == MODE_B_HIGHSEC and total_danger_count >= 2:
                danger_items = []
                if total_icon_count > 0:
                    icon_summary = {}
                    for name, _ in icon_details_summary:
                        icon_summary[name] = icon_summary.get(name, 0) + 1
                    icon_str = ', '.join([f"{name}{count}个" if count > 1 else name for name, count in icon_summary.items()])
                    danger_items.append(f"图标{icon_str}(共{total_icon_count}个)")
                if txt_count > 0:
                    danger_items.append(f"可疑舰船{txt_count}个")
                reason = f"发现{'、'.join(danger_items)}"
                #emergency_evasion(reason)
                emergency_evade_pin999(center_panel2)

            if icon_found or txt_found:
                time.sleep(2)

        except IconNotFoundException as e:
            print(e)

        # 统一补足到约 TARGET_INTERVAL_SEC 秒一轮（B 约 2.4s 补到 5s，A 不检促进约 5s）
        elapsed = time.time() - loop_start
        time.sleep(max(0, TARGET_INTERVAL_SEC - elapsed))

    listener.join()
    play_sound_wav("static/Notification_Ping.wav")
    return 0


def emergency_evade_pin999(
    place_region=None,
    open_key='l',
    pin_text='PIN999',
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
    speak("紧急规避,发现可疑舰船")

    # 1) 打开地点面板
    pyautogui.press(open_key)
    time.sleep(0.2)

    # 2) 找 PIN999（第一个）并右键
    ok = find_txt_ocr(pin_text, 1, place_region,allow_scroll=False) if place_region else find_txt_ocr(pin_text, 1,allow_scroll=False)
    if not ok:
        speak("未找到安全点")
        emergency_evasion('未找到安全点')
        return

    pyautogui.click(button='right')
    time.sleep(0.2)

    # 3) 在右键菜单区域优先找“带领舰队”，失败则找“跳跃至”
    mx, my = pyautogui.position()
    menu_regions = [
        (max(mx - 40, 0), max(my - 10, 0), 520, 560),   # 常规：鼠标右侧/下方
        (max(mx - 520, 0), max(my - 10, 0), 520, 560),  # 兜底：菜单翻到左侧
    ]

    def _find_and_click_first(menu_text: str) -> bool:
        for r in menu_regions:
            okx = find_txt_ocr(menu_text, 1, r,allow_scroll=False)  # 找到会 moveTo
            if okx:
                pyautogui.click(button='left')
                time.sleep(0.15)
                return True
        return False

    if _find_and_click_first(action_text):
        speak("紧急规避已启动,程序终止")
        sys.exit(1)

    if _find_and_click_first(fallback_action_text):
        speak("紧急规避已启动,程序终止")
        sys.exit(1)

    # 都失败：回退
    emergency_evasion('规避失败,未找到跳跃目标,执行自动导航')





def run(mode):
    """运行入口：mode='A' or 'B'"""
    return main(mode)