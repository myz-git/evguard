"""
start_tk.py

纯 tkinter 版本的 EVGuard 控制台（Windows 9x 经典复古风格）。
- 不依赖 customtkinter
- 左侧 A 区: Fsd0, Fsd1, GuardA, GuardB，每项有 Start/Stop/状态
- 底部: 状态刷新、全部退出
- 右侧 B 区: 控制台 (Text) + 自动滚动复选框、清空日志、查看详细日志
- 后台服务用线程模拟（可替换为 subprocess 启动实际脚本）
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import queue
import os

# -------------------------
# 配色（Windows 9x 风格）
# -------------------------
RETRO_BG = "#C0C0C0"        # 窗口背景
RETRO_PANEL = "#E0E0E0"     # 面板
RETRO_BORDER = "#808080"    # 分隔线
RETRO_BTN_FACE = "#D4D0C8"  # 按钮面
RETRO_BTN_HL = "#000080"    # 按钮高亮 (深蓝)
RETRO_TEXT = "#000000"
RETRO_ERR = "#FF0000"

FONT_TITLE = ("MS Sans Serif", 10, "bold")
FONT_NORMAL = ("MS Sans Serif", 10)
MONO_FONT = ("Courier New", 10)

# -------------------------
# Service 模拟器
# -------------------------
class ServiceController:
    def __init__(self, name, log_fn):
        self.name = name
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        self.log = log_fn

    def start(self):
        if self._running:
            self.log(f"{self.name} already running.")
            return False
        self._stop_event.clear()
        t = threading.Thread(target=self._run, daemon=True)
        self._thread = t
        self._running = True
        t.start()
        self.log(f"{self.name} started.")
        return True

    def stop(self):
        if not self._running:
            self.log(f"{self.name} already stopped.")
            return False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False
        self.log(f"{self.name} stopped.")
        return True

    def _run(self):
        counter = 0
        try:
            while not self._stop_event.is_set():
                counter += 1
                self.log(f"{self.name} heartbeat {counter}")
                time.sleep(2.0)
        except Exception as e:
            self.log(f"{self.name} error: {e}")
        finally:
            self._running = False

    def is_running(self):
        return self._running

# -------------------------
# 主 GUI
# -------------------------
class EvguardTkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EVGuard 控制台（Retro Win9x 风格）")
        self.geometry("1000x600")
        self.configure(bg=RETRO_BG)

        # 日志队列（线程安全）
        self.log_queue = queue.Queue()
        self.detailed_logs = []

        # 服务控制器存放
        self.controllers = {}

        # UI 构建
        self._build_ui()

        # 启动定时器处理日志和状态刷新
        self.auto_scroll_var.set(1)
        self.after(150, self._poll_loop)

        # 启动时记录一条日志
        self._enqueue_log("EVGuard GUI (tkinter) started.")

    def _build_ui(self):
        # 外层边框（模拟复古边）
        outer = tk.Frame(self, bg=RETRO_BORDER, bd=0)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        main = tk.Frame(outer, bg=RETRO_PANEL, bd=1, relief="flat")
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # 左右分栏
        left = tk.Frame(main, bg=RETRO_PANEL, width=320)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(main, bg=RETRO_PANEL)
        right.pack(side="right", fill="both", expand=True, padx=(4,8), pady=8)

        # 左侧 - 标题
        lbl_title = tk.Label(left, text="功能列表", bg=RETRO_PANEL, fg=RETRO_TEXT, font=FONT_TITLE, anchor="w")
        lbl_title.pack(fill="x", padx=6, pady=(6,4))

        # 各服务行
        services = ["Fsd0", "Fsd1", "GuardA", "GuardB"]
        for svc in services:
            self._create_service_row(left, svc)
            sep = tk.Frame(left, height=1, bg=RETRO_BORDER)
            sep.pack(fill="x", padx=6, pady=6)

        # 底部按钮
        bottom = tk.Frame(left, bg=RETRO_PANEL)
        bottom.pack(fill="x", padx=6, pady=6)

        btn_refresh = tk.Button(bottom, text="状态刷新", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, relief="raised",
                                font=FONT_NORMAL, command=self._refresh_all_statuses)
        btn_refresh.pack(side="left", expand=True, fill="x", padx=(0,6))

        btn_exit = tk.Button(bottom, text="全部退出", bg="#FF8080", fg=RETRO_TEXT, relief="raised",
                             font=FONT_NORMAL, command=self._on_exit)
        btn_exit.pack(side="right", expand=True, fill="x", padx=(6,0))

        # 右侧 - 控制台顶栏
        ctrlbar = tk.Frame(right, bg=RETRO_PANEL)
        ctrlbar.pack(fill="x", padx=6, pady=(6,4))

        lbl_console = tk.Label(ctrlbar, text="控制台", bg=RETRO_PANEL, fg=RETRO_TEXT, font=FONT_TITLE)
        lbl_console.pack(side="left", padx=(4,8))

        self.auto_scroll_var = tk.IntVar(value=1)
        chk_auto = tk.Checkbutton(ctrlbar, text="自动滚动", variable=self.auto_scroll_var, bg=RETRO_PANEL,
                                  fg=RETRO_TEXT, font=FONT_NORMAL, activebackground=RETRO_PANEL, bd=0)
        chk_auto.pack(side="left", padx=(4,8))

        btn_clear = tk.Button(ctrlbar, text="清空日志", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, relief="raised",
                              font=FONT_NORMAL, command=self._clear_console)
        btn_clear.pack(side="left", padx=(8,4))

        btn_detail = tk.Button(ctrlbar, text="查看详细日志", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, relief="raised",
                               font=FONT_NORMAL, command=self._show_detailed_log)
        btn_detail.pack(side="left", padx=(4,8))

        # 控制台文本区（带滚动条）
        text_frame = tk.Frame(right, bg="white", bd=1, relief="sunken")
        text_frame.pack(fill="both", expand=True, padx=6, pady=(4,6))

        self.console_text = scrolledtext.ScrolledText(text_frame, wrap="none", bg="white", fg=RETRO_TEXT,
                                                      font=MONO_FONT, state="disabled")
        self.console_text.pack(fill="both", expand=True)

    def _create_service_row(self, parent, name):
        frame = tk.Frame(parent, bg=RETRO_PANEL)
        frame.pack(fill="x", padx=6, pady=2)

        lbl = tk.Label(frame, text=name, bg=RETRO_PANEL, fg=RETRO_TEXT, font=FONT_NORMAL, width=10, anchor="w")
        lbl.grid(row=0, column=0, padx=(2,6), pady=6, sticky="w")

        btn_start = tk.Button(frame, text="Start", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, relief="raised",
                              width=8, command=lambda n=name: self._on_start(n))
        btn_start.grid(row=0, column=1, padx=6, pady=6)

        btn_stop = tk.Button(frame, text="Stop", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, relief="raised",
                             width=8, command=lambda n=name: self._on_stop(n))
        btn_stop.grid(row=0, column=2, padx=6, pady=6)

        status_lbl = tk.Label(frame, text="状态: Stopped", bg=RETRO_PANEL, fg=RETRO_TEXT, font=FONT_NORMAL, width=18, anchor="w")
        status_lbl.grid(row=0, column=3, padx=(12,2), pady=6, sticky="w")

        # store controller info
        self.controllers[name] = {
            "controller": ServiceController(name, log_fn=self._enqueue_log),
            "status_label": status_lbl,
            "start_btn": btn_start,
            "stop_btn": btn_stop
        }

    # -------------------------
    # 日志方法
    # -------------------------
    def _enqueue_log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_queue.put(line)
        self.detailed_logs.append(line)

    def _poll_loop(self):
        # 处理队列中的日志
        updated = False
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_console(line + "\n")
            updated = True

        # 刷新状态标签
        self._refresh_status_labels()

        # 继续调度
        self.after(150, self._poll_loop)

    def _append_console(self, text):
        self.console_text.configure(state="normal")
        self.console_text.insert("end", text)
        if self.auto_scroll_var.get():
            self.console_text.see("end")
        self.console_text.configure(state="disabled")

    def _clear_console(self):
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")
        self._enqueue_log("Console cleared by user.")

    def _show_detailed_log(self):
        top = tk.Toplevel(self)
        top.title("详细日志")
        top.geometry("700x400")
        top.configure(bg=RETRO_PANEL)

        txt = scrolledtext.ScrolledText(top, wrap="none", bg="white", fg=RETRO_TEXT, font=MONO_FONT)
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(self.detailed_logs))
        txt.configure(state="disabled")

        btn_frame = tk.Frame(top, bg=RETRO_PANEL)
        btn_frame.pack(fill="x", padx=8, pady=6)

        def save_file():
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("\n".join(self.detailed_logs))
                    self._enqueue_log(f"Detailed log saved to {path}")
                    messagebox.showinfo("保存成功", f"已保存到 {path}")
                except Exception as e:
                    messagebox.showerror("保存失败", str(e))
                    self._enqueue_log(f"Failed to save detailed log: {e}")

        save_btn = tk.Button(btn_frame, text="保存到文件", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, command=save_file)
        save_btn.pack(side="left", padx=6)

        close_btn = tk.Button(btn_frame, text="关闭", bg=RETRO_BTN_FACE, fg=RETRO_TEXT, command=top.destroy)
        close_btn.pack(side="right", padx=6)

    # -------------------------
    # 服务控制回调
    # -------------------------
    def _on_start(self, name):
        entry = self.controllers.get(name)
        if not entry:
            return
        ctl: ServiceController = entry["controller"]
        if ctl.start():
            entry["status_label"].configure(text="状态: Running")
        else:
            entry["status_label"].configure(text="状态: Running" if ctl.is_running() else "状态: Stopped")

    def _on_stop(self, name):
        entry = self.controllers.get(name)
        if not entry:
            return
        ctl: ServiceController = entry["controller"]
        if ctl.stop():
            entry["status_label"].configure(text="状态: Stopped")
        else:
            entry["status_label"].configure(text="状态: Running" if ctl.is_running() else "状态: Stopped")

    def _refresh_all_statuses(self):
        for name, entry in self.controllers.items():
            ctl = entry["controller"]
            entry["status_label"].configure(text="状态: Running" if ctl.is_running() else "状态: Stopped")
        self._enqueue_log("Statuses refreshed by user.")

    def _refresh_status_labels(self):
        for name, entry in self.controllers.items():
            ctl = entry["controller"]
            desired = "状态: Running" if ctl.is_running() else "状态: Stopped"
            if entry["status_label"].cget("text") != desired:
                entry["status_label"].configure(text=desired)

    def _on_exit(self):
        if messagebox.askyesno("退出确认", "确定要停止所有服务并退出吗？"):
            self._enqueue_log("Exiting: stopping all services.")
            # 停止所有服务
            for name, entry in list(self.controllers.items()):
                try:
                    entry["controller"].stop()
                except Exception as e:
                    self._enqueue_log(f"Error stopping {name}: {e}")
            self._enqueue_log("All services requested to stop. Exiting.")
            self.after(300, self.destroy)

# -------------------------
# 运行入口
# -------------------------
def main():
    app = EvguardTkApp()
    app.mainloop()

if __name__ == "__main__":
    main()