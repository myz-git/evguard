import json
import os
import re
import subprocess
import threading
import time
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from license_utils import get_license_info_text, get_machine_code


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# exe 所在目录（打包后）或 dist/FsdGuard 目录（源码运行）
if getattr(sys, "frozen", False):
    EXEC_BASE_DIR = os.path.dirname(sys.executable)
    CONFIG_BASE_DIR = EXEC_BASE_DIR
else:
    EXEC_BASE_DIR = os.path.join(BASE_DIR, "dist", "FsdGuard")
    CONFIG_BASE_DIR = BASE_DIR


def resource_path(rel_path: str) -> str:
    """
    运行在 PyInstaller 打包后的 exe 或源码环境下时，返回静态资源的绝对路径。
    - 一体包(onefile)：使用 sys._MEIPASS
    - 目录模式(onedir)：使用 exe 所在目录
    - 源码运行：使用当前文件所在目录
    """
    if getattr(sys, "_MEIPASS", None):
        base = sys._MEIPASS
    elif getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = BASE_DIR
    return os.path.join(base, rel_path)


def play_sound_wav(file_path: str):
    wav_path = resource_path(file_path)
    if not os.path.isfile(wav_path):
        return
    try:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass

# 日志 / 命令配置 / cfg 等运行时文件，放在 CONFIG_BASE_DIR
LOG_FILE = os.path.join(CONFIG_BASE_DIR, "evguard_log.txt")
COMMANDS_FILE = os.path.join(CONFIG_BASE_DIR, "commands.json")
PROCESS_ORDER = ["Fsd0", "Fsd10", "GuardA", "GuardB", "GuardC"]
DISPLAY_NAMES = {
    "Fsd0": "FSD0",
    "Fsd10": "FSD10",
    "GuardA": "GUARDA",
    "GuardB": "GUARDB",
    "GuardC": "GUARDC",
}
PROCESS_DESCRIPTIONS = {
    "Fsd0": "0M自动导航",
    "Fsd10": "微曲加速导航",
    "GuardA": "低安被动预警",
    "GuardB": "高安主动防御",
    "GuardC": "低安主动防御",
}
DEFAULT_COMMANDS = {
    "Fsd0": os.path.join(EXEC_BASE_DIR, "FSD0.exe"),
    "Fsd10": os.path.join(EXEC_BASE_DIR, "FSD10.exe"),
    "GuardA": os.path.join(EXEC_BASE_DIR, "GuardA.exe"),
    "GuardB": os.path.join(EXEC_BASE_DIR, "GuardB.exe"),
    "GuardC": os.path.join(EXEC_BASE_DIR, "GuardC.exe"),
}
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
if getattr(sys, "frozen", False):
    CFG_FILE = resource_path("cfg.txt")
else:
    CFG_FILE = os.path.join(BASE_DIR, "cfg.txt")


COLORS = {
    "bg": "#1b2026",
    "surface": "#242b33",
    "surface_alt": "#2f3842",
    "surface_active": "#34404d",
    "line": "#4a5563",
    "line_active": "#d1d5db",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "brand": "#f59e0b",
    "brand_dark": "#fbbf24",
    "danger": "#ef4444",
    "ok": "#22c55e",
    "warn": "#f59e0b",
}

BUTTON_THEME = {
    "primary": {"bg": "#f59e0b", "fg": "#111827", "hover": "#fbbf24", "border": "#d97706"},
    "danger": {"bg": "#b91c1c", "fg": "#ffffff", "hover": "#dc2626", "border": "#7f1d1d"},
    "soft": {"bg": "#3a4654", "fg": "#e5e7eb", "hover": "#4b5a6b", "border": "#64748b"},
}


def load_commands():
    # 打包后的 start.exe：始终调用当前目录下的其它 exe，不读写 commands.json
    if getattr(sys, "frozen", False):
        return dict(DEFAULT_COMMANDS)

    # 源码运行时才允许通过 commands.json 自定义路径
    commands = dict(DEFAULT_COMMANDS)
    if os.path.exists(COMMANDS_FILE):
        try:
            with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for name in PROCESS_ORDER:
                    value = raw.get(name)
                    if isinstance(value, str) and value.strip():
                        commands[name] = value.strip()
        except Exception:
            pass
    else:
        with open(COMMANDS_FILE, "w", encoding="utf-8") as f:
            json.dump(commands, f, ensure_ascii=False, indent=2)
    return commands


def load_cfg():
    config = {}
    if not os.path.exists(CFG_FILE):
        return config

    try:
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                config[key.strip().lower()] = value.strip()
    except Exception:
        return {}

    return config


class ProcessManager:
    def __init__(self, commands, log_file=LOG_FILE):
        self.commands = commands
        self.log_file = log_file
        self.lock = threading.RLock()
        self.processes = {name: None for name in PROCESS_ORDER}
        self.statuses = {name: "Stopped" for name in PROCESS_ORDER}
        self.output_threads = {}
        self.clear_logs()

    def get_status(self, name):
        with self.lock:
            return self.statuses.get(name, "Unknown")

    def get_pid(self, name):
        with self.lock:
            proc = self.processes.get(name)
            if proc and proc.poll() is None:
                return proc.pid
            return None

    def _decode_line(self, line_bytes):
        for encoding in ("utf-8", "gbk"):
            try:
                return line_bytes.decode(encoding).rstrip("\r\n")
            except UnicodeDecodeError:
                continue
        return line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")

    def _finalize_exited_process(self, name, proc, source):
        with self.lock:
            current = self.processes.get(name)
            if current is not proc:
                return
            return_code = proc.poll()
            if return_code is None:
                return
            self.processes[name] = None
            self.statuses[name] = "Stopped"
            self.output_threads.pop(name, None)
        self.log(f"{DISPLAY_NAMES.get(name, name)} 已退出({source})，返回码: {return_code}")

    def start_process(self, name):
        command = self.commands.get(name, "")
        command_path = command.strip() if isinstance(command, str) else ""
        with self.lock:
            proc = self.processes.get(name)
            if proc and proc.poll() is None:
                return False, f"{DISPLAY_NAMES.get(name, name)} 已在运行中。"
            for other_name, other_proc in self.processes.items():
                if other_name != name and other_proc and other_proc.poll() is None:
                    return False, (
                        f"当前仅允许同时运行一个功能。"
                        f"请先停止 {DISPLAY_NAMES.get(other_name, other_name)}。"
                    )
        if not command_path:
            return False, f"{DISPLAY_NAMES.get(name, name)} 未配置可执行文件路径。"
        if not os.path.exists(command_path):
            return False, f"{DISPLAY_NAMES.get(name, name)} 可执行文件不存在: {command_path}"

        launch_command = [command_path]
        if command_path.lower().endswith(".py"):
            launch_command = [sys.executable, command_path]

        try:
            # 传入环境变量，让子进程在 log_message 时同步一份到 stdout，便于这里抓取并显示到控制台界面
            env = dict(os.environ)
            env["EVGUARD_CHILD_LOG_TO_STDOUT"] = "1"
            env["EVGUARD_UI_LOG_FILE"] = self.log_file
            env["EVGUARD_UI_LOG_PREFIX"] = f"[{DISPLAY_NAMES.get(name, name)}] "
            proc = subprocess.Popen(
                launch_command,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env=env,
            )
        except Exception as e:
            self.log(f"启动 {DISPLAY_NAMES.get(name, name)} 失败: {e}")
            return False, f"启动失败: {e}"

        with self.lock:
            self.processes[name] = proc
            self.statuses[name] = "Running"

        self.log(f"{DISPLAY_NAMES.get(name, name)} 启动中")
        return True, f"{DISPLAY_NAMES.get(name, name)} 启动成功。"

    def _read_output_thread(self, name, proc):
        pipe = proc.stdout
        if pipe is None:
            self._finalize_exited_process(name, proc, "no-stdout")
            return

        try:
            for line_bytes in iter(pipe.readline, b""):
                if not line_bytes:
                    continue
                line = ANSI_ESCAPE.sub("", self._decode_line(line_bytes)).strip()
                if line:
                    self.log(f"[{DISPLAY_NAMES.get(name, name)}] {line}", add_timestamp=False)
        except Exception as e:
            self.log(f"读取 {name} 输出异常: {e}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass
            self._finalize_exited_process(name, proc, "stdout-thread")

    def stop_process(self, name):
        with self.lock:
            proc = self.processes.get(name)
            if not proc or proc.poll() is not None:
                self.processes[name] = None
                self.statuses[name] = "Stopped"
                self.output_threads.pop(name, None)
                return False, f"{DISPLAY_NAMES.get(name, name)} 当前未运行。"

        try:
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.log(f"{DISPLAY_NAMES.get(name, name)} 正常停止超时，尝试强制结束")
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(proc.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    proc.kill()
                proc.wait(timeout=3)
        except Exception as e:
            self.log(f"停止 {DISPLAY_NAMES.get(name, name)} 失败: {e}")
            return False, f"停止失败: {e}"

        with self.lock:
            self.statuses[name] = "Stopped"
            self.processes[name] = None
            self.output_threads.pop(name, None)
        self.log(f"{DISPLAY_NAMES.get(name, name)} 已停止")
        return True, f"{DISPLAY_NAMES.get(name, name)} 已停止。"

    def refresh_statuses(self):
        with self.lock:
            items = list(self.processes.items())
        for name, proc in items:
            if proc is not None:
                self._finalize_exited_process(name, proc, "refresh")

    def stop_all(self):
        for name in PROCESS_ORDER:
            self.stop_process(name)

    def clear_logs(self):
        with self.lock:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, message, add_timestamp=True):
        with self.lock:
            if add_timestamp:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                full_message = f"{timestamp} - {message}"
            else:
                full_message = message
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(full_message + "\n")

    def get_logs(self):
        with self.lock:
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""

    def get_logs_since(self, offset):
        with self.lock:
            try:
                size = os.path.getsize(self.log_file)
                if offset > size:
                    offset = 0
                with open(self.log_file, "r", encoding="utf-8") as f:
                    f.seek(offset)
                    data = f.read()
                    new_offset = f.tell()
                return data, new_offset
            except Exception:
                return "", offset


class EvGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EvGuard")
        self.root.geometry("840x640")
        self.root.minsize(720, 520)
        self.root.configure(bg=COLORS["bg"])
        self._apply_app_icon()
        self._apply_dark_titlebar()
        self._setup_ttk_styles()

        self.manager = ProcessManager(load_commands())
        self.config = load_cfg()
        self.log_offset = 0
        self.running = True
        self.pulse_on = False
        self.all_logs = ""

        self.process_ui = {}
        self.log_filter_var = tk.StringVar(value="全部")
        self.auto_scroll_var = tk.BooleanVar(value=True)
        self.status_message_var = tk.StringVar(value="就绪 | 系统稳定")
        self.running_count_var = tk.StringVar(value="0")
        self.stopped_count_var = tk.StringVar(value=str(len(PROCESS_ORDER)))
        self.updated_at_var = tk.StringVar(value="--:--:--")
        self.starting_states = {}
        self.spinner_step = 0

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._schedule_log_refresh()
        self._animate_status_lights()

    def _setup_ttk_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Console.Vertical.TScrollbar",
            troughcolor="#0f172a",
            background="#1f2937",
            bordercolor="#334155",
            arrowcolor="#94a3b8",
            darkcolor="#1f2937",
            lightcolor="#1f2937",
            relief="flat",
            gripcount=0,
        )
        style.map(
            "Console.Vertical.TScrollbar",
            background=[("active", "#334155"), ("pressed", "#475569")],
            arrowcolor=[("active", "#cbd5e1")],
        )

    def _apply_dark_titlebar(self):
        if os.name != "nt":
            return
        try:
            import ctypes

            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
            hwnd = self.root.winfo_id()
            value = ctypes.c_int(1)
            dwmapi = ctypes.windll.dwmapi
            sizeof_value = ctypes.sizeof(value)
            result = dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value),
                sizeof_value,
            )
            if result != 0:
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE_OLD,
                    ctypes.byref(value),
                    sizeof_value,
                )
        except Exception:
            pass

    def _apply_app_icon(self):
        ico_path = os.path.join(BASE_DIR, "icon", "eva.ico")
        try:
            self.root.iconbitmap(ico_path)
        except Exception:
            pass

    def _build_layout(self):
        shell = tk.Frame(self.root, bg=COLORS["bg"])
        shell.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_top_bar(shell)

        main = tk.Frame(shell, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, pady=(10, 0))

        split = tk.PanedWindow(
            main,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief="raised",
            showhandle=False,
            bg=COLORS["bg"],
            bd=0,
        )
        split.pack(fill="both", expand=True)

        left_host = tk.Frame(split, bg=COLORS["bg"], width=390)
        right_host = tk.Frame(split, bg=COLORS["bg"])
        split.add(left_host, minsize=320)
        split.add(right_host, minsize=240)

        self._build_process_panel(left_host)
        self._build_log_panel(right_host)
        self.root.after(0, lambda: split.sash_place(0, 380, 0))

    def _build_top_bar(self, parent):
        top = tk.Frame(parent, bg=COLORS["surface"], highlightbackground=COLORS["line"], highlightthickness=1)
        top.pack(fill="x")

        banner_wrap = tk.Frame(top, bg="black", height=44)
        banner_wrap.pack(fill="x")
        banner_wrap.pack_propagate(False)
        self._build_banner(banner_wrap)

        meta = tk.Frame(top, bg=COLORS["surface_alt"])
        meta.pack(fill="x", padx=12, pady=10)

        title_box = tk.Frame(meta, bg=COLORS["surface_alt"])
        title_box.pack(side="left", fill="x", expand=True)
        tk.Label(title_box, text="EvGuard 控制台", bg=COLORS["surface_alt"], fg=COLORS["text"], font=("Bahnschrift", 15, "bold")).pack(side="left")
        self._build_stat_chip(title_box, "RUN", self.running_count_var, "#062814", COLORS["ok"])
        self._build_stat_chip(title_box, "STOP", self.stopped_count_var, "#2b1313", COLORS["danger"])
        self._build_stat_chip(title_box, "SYNC", self.updated_at_var, "#172554", "#60a5fa")

        exit_btn = tk.Button(meta, text="EMERGENCY EXIT", command=self.exit_all)
        self._style_button(exit_btn, "danger")
        exit_btn.pack(side="right", padx=(8, 0))

        refresh_btn = tk.Button(meta, text="REFRESH", command=self.refresh)
        self._style_button(refresh_btn, "soft")
        refresh_btn.pack(side="right", padx=(8, 0))

        stripe = tk.Frame(top, bg="#0f141b", height=2)
        stripe.pack(fill="x")

    def _build_stat_chip(self, parent, title, value_var, bg, fg):
        chip = tk.Frame(parent, bg=bg, highlightbackground=COLORS["line"], highlightthickness=1)
        chip.pack(side="left", padx=(0, 8), ipadx=3, ipady=1)
        tk.Label(chip, text=f"{title} ", bg=bg, fg=COLORS["muted"], font=("Consolas", 8, "bold")).pack(side="left", padx=(6, 0))
        tk.Label(chip, textvariable=value_var, bg=bg, fg=fg, font=("Consolas", 9, "bold")).pack(side="left", padx=(0, 6))

    def _build_banner(self, parent):
        try:
            banner_img = tk.PhotoImage(file=resource_path("static/eve-ban.png"))
            label = tk.Label(parent, image=banner_img, bg="black", borderwidth=0, highlightthickness=0)
            label.image = banner_img
            label.pack(side="left", padx=6)
        except Exception as e:
            tk.Label(parent, text=f"Banner 加载失败: {e}", bg="black", fg="#fca5a5", font=("Consolas", 9)).pack(anchor="w", padx=8)

    def _build_process_panel(self, parent):
        left = tk.Frame(parent, bg=COLORS["bg"])
        left.pack(fill="both", expand=True, padx=(0, 10))

        self.card_container = tk.Frame(left, bg=COLORS["bg"])
        self.card_container.pack(fill="both", expand=True)

        for name in PROCESS_ORDER:
            card = tk.Frame(self.card_container, bg=COLORS["surface"], highlightbackground=COLORS["line"], highlightthickness=2, relief="groove", bd=1)
            card.pack(fill="x", pady=5)

            head = tk.Frame(card, bg=COLORS["surface"])
            head.pack(fill="x", padx=10, pady=(8, 4))

            status_dot = tk.Canvas(head, width=10, height=10, bg=COLORS["surface"], highlightthickness=0)
            status_dot.pack(side="left", padx=(0, 6))
            status_dot.create_oval(1, 1, 9, 9, fill="#4b5563", outline="")
            title_label = tk.Label(
                head,
                text=DISPLAY_NAMES.get(name, name),
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Segoe UI", 10, "bold"),
            )
            title_label.pack(side="left")
            status_pill = tk.Label(head, text="Stopped", bg="#fee2e2", fg=COLORS["danger"], padx=8, pady=2, font=("Segoe UI", 8, "bold"))
            status_pill.pack(side="right")

            info = tk.Frame(card, bg=COLORS["surface"])
            info.pack(fill="x", padx=10, pady=(0, 8))
            desc_label = tk.Label(
                info,
                text=PROCESS_DESCRIPTIONS.get(name, ""),
                bg=COLORS["surface"],
                fg="#c5cedb",
                font=("Microsoft YaHei UI", 10),
                anchor="w",
            )
            desc_label.pack(side="left", fill="x", expand=True)

            btns = tk.Frame(info, bg=COLORS["surface"])
            btns.pack(side="right")
            start_btn = tk.Button(btns, text=">> START", width=9, command=lambda n=name: self.start(n))
            self._style_button(start_btn, "primary")
            start_btn.pack(side="left", padx=(0, 6))
            stop_btn = tk.Button(btns, text="|| STOP", width=9, command=lambda n=name: self.stop(n))
            self._style_button(stop_btn, "danger")
            stop_btn.pack(side="left")

            self.process_ui[name] = {
                "card": card,
                "head": head,
                "info": info,
                "status_dot": status_dot,
                "status_pill": status_pill,
                "title_label": title_label,
                "desc_label": desc_label,
                "btns": btns,
                "start_btn": start_btn,
                "stop_btn": stop_btn,
            }
            self._update_one_status(name)

        self.status_strip = tk.Label(
            left,
            textvariable=self.status_message_var,
            anchor="w",
            bg="#111827",
            fg=COLORS["brand_dark"],
            padx=12,
            pady=6,
        )
        self.loading_bar = tk.Canvas(
            left,
            height=8,
            bg="#0b1220",
            highlightthickness=1,
            highlightbackground="#223048",
            bd=0,
        )
        self.loading_bar.pack(fill="x", pady=(6, 0))
        self.status_strip.pack(fill="x", pady=(8, 0))

    def _build_log_panel(self, parent):
        right = tk.Frame(parent, bg=COLORS["surface"], highlightbackground=COLORS["line"], highlightthickness=1)
        right.pack(fill="both", expand=True)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        tool = tk.Frame(right, bg=COLORS["surface_alt"])
        tool.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        tk.Label(tool, text="日志", bg=COLORS["surface_alt"], fg=COLORS["text"], font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 8))
        tk.Label(tool, text="过滤", bg=COLORS["surface_alt"], fg=COLORS["muted"]).pack(side="left")
        filter_options = ["全部"] + [DISPLAY_NAMES.get(name, name) for name in PROCESS_ORDER]
        filter_menu = tk.OptionMenu(tool, self.log_filter_var, "全部", *filter_options, command=lambda *_: self._render_logs())
        filter_menu.config(bg=COLORS["surface"], fg=COLORS["text"], activebackground=COLORS["surface_alt"], relief="flat", highlightthickness=1, highlightbackground=COLORS["line"])
        filter_menu.pack(side="left", padx=(4, 8))
        filter_menu["menu"].config(bg=COLORS["surface"], fg=COLORS["text"])

        about_btn = tk.Button(tool, text="About", command=self.show_about)
        self._style_button(about_btn, "soft")
        about_btn.pack(side="right")

        clear_btn = tk.Button(tool, text="Clear", command=self.clear_logs)
        self._style_button(clear_btn, "soft")
        clear_btn.pack(side="right", padx=(8, 0))

        tk.Checkbutton(
            tool,
            text="自动滚动",
            variable=self.auto_scroll_var,
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
        ).pack(side="right", padx=(0, 8))

        console_wrap = tk.Frame(right, bg=COLORS["surface"])
        console_wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        console_wrap.grid_rowconfigure(0, weight=1)
        console_wrap.grid_columnconfigure(0, weight=1)

        self.console_scroll = ttk.Scrollbar(
            console_wrap,
            orient="vertical",
            style="Console.Vertical.TScrollbar",
        )
        self.console_scroll.grid(row=0, column=1, sticky="ns")

        self.console = tk.Text(
            console_wrap,
            font=("Microsoft YaHei UI", 8),
            bg="#0b1220",
            fg="#dbe7f5",
            insertbackground="#dbe7f5",
            wrap=tk.WORD,
            relief="flat",
            padx=10,
            pady=9,
            spacing1=1,
            spacing2=1,
            spacing3=2,
            yscrollcommand=self.console_scroll.set,
        )
        self.console.grid(row=0, column=0, sticky="nsew")
        self.console_scroll.config(command=self.console.yview)
        self.console.tag_config("warn", foreground="#fbbf24")
        self.console.tag_config("error", foreground="#fca5a5")
        self.console.tag_config("ok", foreground="#93c5fd")

    def _set_status_message(self, message, level="info"):
        self.status_message_var.set(message)
        if level == "error":
            bg, fg = "#2b1313", COLORS["danger"]
        elif level == "warn":
            bg, fg = "#2b2113", COLORS["warn"]
        else:
            bg, fg = "#111827", COLORS["brand_dark"]
        self.status_strip.config(bg=bg, fg=fg)

    def _style_button(self, button, kind):
        theme = BUTTON_THEME[kind]
        button.configure(
            bg=theme["bg"],
            fg=theme["fg"],
            activebackground=theme["hover"],
            activeforeground=theme["fg"],
            disabledforeground="#9ca3af",
            relief="raised",
            bd=2,
            overrelief="ridge",
            cursor="hand2",
            font=("Bahnschrift", 10, "bold"),
            padx=13,
            pady=2,
            highlightthickness=1,
            highlightbackground=theme["border"],
            highlightcolor=theme["border"],
        )
        button.bind("<Enter>", lambda e, b=button, t=theme: self._on_button_hover(b, t["hover"]))
        button.bind("<Leave>", lambda e, b=button, t=theme: self._on_button_hover(b, t["bg"]))
        button.bind("<ButtonPress-1>", lambda e, b=button: self._on_button_press(b))
        button.bind("<ButtonRelease-1>", lambda e, b=button: self._on_button_release(b))

    def _on_button_hover(self, button, color):
        if str(button.cget("state")) != "disabled":
            button.configure(bg=color)

    def _on_button_press(self, button):
        if str(button.cget("state")) != "disabled":
            button.configure(relief="sunken", padx=12, pady=1)

    def _on_button_release(self, button):
        if str(button.cget("state")) != "disabled":
            button.configure(relief="raised", padx=13, pady=2)

    def start(self, name):
        self._set_status_message(f"{DISPLAY_NAMES.get(name, name)} 启动中，正在加载模块…")
        ok, message = self.manager.start_process(name)
        if ok:
            play_sound_wav("static/started.wav")
            self.starting_states[name] = {"started_at": time.time(), "seen_output": False}
        else:
            self.starting_states.pop(name, None)
        self._update_one_status(name)
        self._update_summary()
        if ok:
            self._set_status_message(f"{DISPLAY_NAMES.get(name, name)} 已启动，模块加载中…")
        else:
            self._set_status_message(message, "error")
            messagebox.showwarning("提示", message)

    def stop(self, name):
        ok, message = self.manager.stop_process(name)
        self.starting_states.pop(name, None)
        self._update_one_status(name)
        self._update_summary()
        if ok:
            play_sound_wav("static/Notification_Ping.wav")
            self._set_status_message(message)
        else:
            self._set_status_message(message, "warn")
            messagebox.showinfo("提示", message)

    def refresh(self):
        self.manager.refresh_statuses()
        for name in PROCESS_ORDER:
            self._update_one_status(name)
        self._update_summary()
        self._set_status_message("状态已刷新")

    def exit_all(self):
        self.running = False
        self.manager.stop_all()
        self.refresh()
        self.root.quit()

    def clear_logs(self):
        self.manager.clear_logs()
        self.log_offset = 0
        self.all_logs = ""
        self.console.delete("1.0", tk.END)
        self._set_status_message("日志已清空")

    def show_about(self):
        version = self.config.get("version", "未知")
        machine_code = get_machine_code()
        try:
            license_info = get_license_info_text()
        except Exception as e:
            license_info = f"读取失败: {e}"
        dialog = tk.Toplevel(self.root)
        dialog.title("About")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(bg=COLORS["surface"])
        dialog.grab_set()

        body = tk.Frame(dialog, bg=COLORS["surface"], padx=16, pady=16)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text=f"版本信息: {version}",
            anchor="w",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill="x")

        tk.Label(
            body,
            text=f"授权信息: {license_info}",
            justify="left",
            anchor="w",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
            wraplength=420,
        ).pack(fill="x", pady=(10, 0))

        tk.Label(
            body,
            text="设备序列号:",
            anchor="w",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", pady=(10, 4))

        code_row = tk.Frame(body, bg=COLORS["surface"])
        code_row.pack(fill="x")

        code_var = tk.StringVar(value=machine_code)
        code_entry = tk.Entry(
            code_row,
            textvariable=code_var,
            relief="flat",
            readonlybackground="#0b1220",
            bg="#0b1220",
            fg="#dbe7f5",
            insertbackground="#dbe7f5",
            font=("Consolas", 9),
        )
        code_entry.pack(side="left", fill="x", expand=True)
        code_entry.configure(state="readonly")

        copy_btn = tk.Button(code_row, text="Copy", command=lambda: self._copy_machine_code(machine_code))
        self._style_button(copy_btn, "soft")
        copy_btn.pack(side="left", padx=(8, 0))

        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - width) // 2)
        y = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - height) // 2)
        dialog.geometry(f"+{x}+{y}")
        code_entry.focus_set()
        code_entry.selection_range(0, tk.END)

    def _copy_machine_code(self, machine_code):
        self.root.clipboard_clear()
        self.root.clipboard_append(machine_code)
        self._set_status_message("设备序列号已复制")

    def _update_summary(self):
        running = sum(1 for name in PROCESS_ORDER if self.manager.get_status(name) == "Running")
        stopped = len(PROCESS_ORDER) - running
        self.running_count_var.set(str(running))
        self.stopped_count_var.set(str(stopped))
        self.updated_at_var.set(time.strftime("%H:%M:%S"))

    def _update_one_status(self, name):
        ui = self.process_ui.get(name)
        if not ui:
            return

        status = self.manager.get_status(name)
        start_meta = self.starting_states.get(name)

        if status == "Running":
            self._apply_card_state(ui, active=True)
            if start_meta:
                elapsed = int(time.time() - start_meta["started_at"])
                ui["status_pill"].config(text=f"STARTING · {elapsed}s", bg="#13233f", fg="#93c5fd")
                ui["status_dot"].itemconfig(1, fill="#3b82f6")
            else:
                ui["status_pill"].config(text="RUNNING", bg="#082515", fg=COLORS["ok"])
                ui["status_dot"].itemconfig(1, fill=COLORS["ok"])
            ui["start_btn"].config(state="disabled")
            ui["stop_btn"].config(state="normal")
        else:
            self.starting_states.pop(name, None)
            self._apply_card_state(ui, active=False)
            ui["status_pill"].config(text="STOPPED", bg="#2b1313", fg=COLORS["danger"])
            ui["status_dot"].itemconfig(1, fill=COLORS["danger"])
            ui["start_btn"].config(state="normal")
            ui["stop_btn"].config(state="disabled")

    def _apply_card_state(self, ui, active):
        if active:
            card_bg = COLORS["surface_active"]
            card_line = COLORS["line_active"]
            text_fg = "#f8fafc"
            desc_fg = "#dbe7f5"
        else:
            card_bg = COLORS["surface"]
            card_line = COLORS["line"]
            text_fg = COLORS["text"]
            desc_fg = "#c5cedb"

        ui["card"].config(bg=card_bg, highlightbackground=card_line)
        ui["head"].config(bg=card_bg)
        ui["info"].config(bg=card_bg)
        ui["btns"].config(bg=card_bg)
        ui["title_label"].config(bg=card_bg, fg=text_fg)
        ui["desc_label"].config(bg=card_bg, fg=desc_fg)
        ui["status_dot"].config(bg=card_bg)

    def _cleanup_starting_states(self):
        now = time.time()
        to_remove = []
        for name, meta in self.starting_states.items():
            if self.manager.get_status(name) != "Running":
                to_remove.append((name, "stopped"))
                continue
            if meta.get("seen_output"):
                to_remove.append((name, "ready"))
                continue
            if now - meta["started_at"] > 40:
                to_remove.append((name, "timeout"))

        for name, reason in to_remove:
            self.starting_states.pop(name, None)
            if reason == "ready":
                self._set_status_message(f"{DISPLAY_NAMES.get(name, name)} 已进入运行状态")
            elif reason == "timeout":
                self._set_status_message(f"{DISPLAY_NAMES.get(name, name)} 运行中，暂无日志输出", "warn")

    def _animate_status_lights(self):
        if not self.running:
            return
        self._cleanup_starting_states()
        self.pulse_on = not self.pulse_on
        self.spinner_step = (self.spinner_step + 1) % 4
        on_color = "#4ade80"
        dim_color = "#166534"
        start_on = "#60a5fa"
        start_dim = "#1d4ed8"
        for name, ui in self.process_ui.items():
            if self.manager.get_status(name) == "Running":
                if name in self.starting_states:
                    ui["status_dot"].itemconfig(1, fill=(start_on if self.pulse_on else start_dim))
                else:
                    ui["status_dot"].itemconfig(1, fill=(on_color if self.pulse_on else dim_color))

        if self.starting_states:
            name = next(iter(self.starting_states.keys()))
            elapsed = int(time.time() - self.starting_states[name]["started_at"])
            dots = "." * self.spinner_step
            self.status_strip.config(bg="#172554", fg="#93c5fd")
            self.status_message_var.set(
                f"{DISPLAY_NAMES.get(name, name)} 启动中{dots} 正在加载模块，请稍候 ({elapsed}s)"
            )
            self.loading_bar.delete("all")
            width = max(60, self.loading_bar.winfo_width())
            seg = max(16, width // 5)
            travel = width + seg
            phase = (int(time.time() * 1000 / 120) % travel) - seg
            self.loading_bar.create_rectangle(0, 0, width, 8, fill="#0b1220", outline="")
            self.loading_bar.create_rectangle(phase, 1, phase + seg, 7, fill="#60a5fa", outline="")
            self.loading_bar.create_rectangle(phase - (seg + 24), 2, phase - 24, 6, fill="#2563eb", outline="")
        else:
            self.loading_bar.delete("all")
        self.root.after(380, self._animate_status_lights)

    def _render_logs(self):
        selected = self.log_filter_var.get()

        lines = self.all_logs.splitlines()
        if selected != "全部":
            lines = [line for line in lines if f"[{selected}]" in line]

        self.console.delete("1.0", tk.END)
        for line in lines:
            tag = None
            low = line.lower()
            if "error" in low or "失败" in line or "异常" in line:
                tag = "error"
            elif "warn" in low or "超时" in line:
                tag = "warn"
            elif "启动成功" in line or "已启动" in line:
                tag = "ok"
            if tag:
                self.console.insert(tk.END, line + "\n", tag)
            else:
                self.console.insert(tk.END, line + "\n")

        if self.auto_scroll_var.get():
            self.console.see(tk.END)

    def _schedule_log_refresh(self):
        if not self.running:
            return

        self.manager.refresh_statuses()
        for name in PROCESS_ORDER:
            self._update_one_status(name)
        self._update_summary()

        data, self.log_offset = self.manager.get_logs_since(self.log_offset)
        if data:
            for line in data.splitlines():
                for name, disp in DISPLAY_NAMES.items():
                    if f"[{disp}]" in line and name in self.starting_states:
                        self.starting_states[name]["seen_output"] = True
            self.all_logs += data
            self._render_logs()

        self.root.after(500, self._schedule_log_refresh)

    def on_closing(self):
        self.running = False
        self.manager.stop_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EvGuardApp(root)
    root.mainloop()
