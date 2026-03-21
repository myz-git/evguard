import time
import random
from pynput import keyboard
import sys
import logging
from license_utils import ensure_license_or_exit
from utils import play_sound_wav, log_message, print_startup, safe_find_icon, safe_find_any_icon, hscrollscreen, rolljump, screen_regions, find_txt_ocr, dump_ocr_texts, speak, suppress_not_found_warnings_console, is_game_exefile_running


ensure_license_or_exit()

# fsd0 执行时不向控制台输出「未找到」类 WARNING（仍写 task.log）
suppress_not_found_warnings_console()

########################################################
# 监听键盘事件

# 全局变量，用于控制程序是否继续运行
running = True

def stop_program():
    global running
    running = False
    print("Ctrl+F12 pressed, stopping the program.")

def start_hotkey_listener():
    hotkeys = keyboard.GlobalHotKeys({
        '<ctrl>+<f12>': stop_program
    })
    hotkeys.start()
    return hotkeys
########################################################

def main():
    """主函数，整合导航和自动驾驶"""
    log_message("INFO", "FSD自动导航 运行开始", screenshot=False)
    # speak("自动导航启动")
    play_sound_wav("static/Login_Connecting.wav")

    # 开始监听键盘事件
    global running
    hotkeys = start_hotkey_listener()

    # 屏幕区域配置
    region_full_right = screen_regions['full_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']

    ctr = keyboard.Controller()
    state = "set_destination"
    find_gate_attempts = 0
    max_find_gate_attempts = 5

    if not is_game_exefile_running():
        print("游戏未启动, 请先启动游戏")
        return 1
    print("游戏检测中...请稍后...")
    # 只在循环第一次执行时播放一次启动音效
    played_start_sound = False    
    while running:
        if not played_start_sound:
            # play_sound_wav("static/started.wav")
            played_start_sound = True
        if state == "set_destination":
            print("finding remote destination")
            if safe_find_icon("zhongdian2", mid_left_panel, max_attempts=1):
                log_message("INFO", "终点设置成功，切换到check_local状态", screenshot=False)
                logging.info("终点设置成功，切换到check_local状态")
                state = "check_local"
            else:
                # log_message("ERROR", "未找到终点", screenshot=False)
                state = "find_gate"
                # return 1
        
        elif state == "check_local":
            print("finding local destination")
            if safe_find_icon("tingkao1", mid_left_panel, max_attempts=1):
                log_message("INFO", "找到tingkao1，切换到check_dock状态", screenshot=False)
                logging.info("找到tingkao1，切换到check_dock状态")
                state = "check_dock"
            else:
                log_message("INFO", "未找到tingkao1，切换到find_gate状态", screenshot=True)
                logging.info("未找到tingkao1，切换到find_gate状态")
                state = "find_gate"

        elif state == "find_gate":
            print("finding jump gate")
            gate_icons = [
                ("jump0", 2),
                ("jump8", 1),
                ("jump7", 1),
                ("jump6", 1),
                ("jump5", 1),
                ("jump4", 1),
            ]

            if any(
                safe_find_icon(name, region_full_right, max_attempts=attempts)
                for name, attempts in gate_icons
            ):
                log_message("INFO", "找到跳跃门，切换到warp状态", screenshot=False)
                logging.info("找到跳跃门，切换到warp状态")
                print("starting warp")
                state = "warp"
                find_gate_attempts = 0
            else:
                hscrollscreen()
                find_gate_attempts += 1
                if find_gate_attempts >= max_find_gate_attempts:
                    log_message("ERROR", f"find_gate尝试{max_find_gate_attempts}次失败", screenshot=True)
                    logging.error(f"find_gate尝试{max_find_gate_attempts}次失败")
                    return 1
                log_message("INFO", f"find_gate尝试第{find_gate_attempts}次", screenshot=False)

        elif state == "warp":
            if safe_find_any_icon(["jump3", "jump3s"], region_full_right, max_attempts=1):
                log_message("INFO", "找到jump3，切换到check_dock状态", screenshot=False)
                state = "check_dock"
            else:
                if rolljump():
                    log_message("INFO", "rolljump成功，切换到check_dock状态", screenshot=False)
                    logging.info("rolljump成功，切换到check_dock状态")
                    # pyautogui.click()
                    # speak("rolljump成功，准备切换到check_dock状态,等待30秒")
                    time.sleep(20)
                    state = "check_dock"
                elif find_txt_ocr("跃迁至该处",max_attempts=1,region=mid_left_panel): 
                    logging.info("已到达目的地,程序停止")
                    print("Arrived at destination")
                    return 0
                else:
                    log_message("INFO", "rolljump继续尝试", screenshot=False)

        elif state == "check_dock":
            # if safe_find_icon("out1", region_full_right, max_attempts=30,threshold=0.7, cnn_threshold=0.60, action=None):
            while running:
                # speak("已切换到check_dock状态,等待完成停靠")
                if find_txt_ocr("离站",max_attempts=1,region=region_full_right):
                    log_message("INFO", "空间站已停靠，运行结束", screenshot=False)
                    logging.info("空间站已停靠，运行结束")
                    print("已到达目的地，运行结束")
                    time.sleep(2)
                    return 0

        time.sleep(random.uniform(0.5, 2.0))

if __name__ == "__main__":
    print_startup("FSD0", ["按 Ctrl+F12 可停止程序"])
    try:
        exit_code = main()
        # speak("自动导航停止")
        play_sound_wav("static/Notification_Ping.wav")
        sys.exit(exit_code)
    except Exception as e:
        log_message("ERROR", f"全局异常: {e}", screenshot=True)
        play_sound_wav("static/Notification_Ping.wav")
        # speak("自动导航停止")
        
        sys.exit(1)
