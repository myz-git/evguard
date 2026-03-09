import json
import os
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "evguard_log.txt")
COMMANDS_FILE = os.path.join(BASE_DIR, "commands.json")
PROCESS_ORDER = ["Fsd0", "Fsd1", "GuardA", "GuardB"]
DEFAULT_COMMANDS = {
    "Fsd0": os.path.join(BASE_DIR, "dist", "FsdGuard", "FSD0.exe"),
    "Fsd1": os.path.join(BASE_DIR, "dist", "FsdGuard", "FSD10.exe"),
    "GuardA": os.path.join(BASE_DIR, "dist", "FsdGuard", "GuardA.exe"),
    "GuardB": os.path.join(BASE_DIR, "dist", "FsdGuard", "GuardB.exe"),
}
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def load_commands():
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
            # 配置文件损坏时，继续使用默认配置。
            pass
    else:
        with open(COMMANDS_FILE, "w", encoding="utf-8") as f:
            json.dump(commands, f, ensure_ascii=False, indent=2)
    return commands


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
        self.log(f"{name} 已退出({source})，返回码: {return_code}")

    def start_process(self, name):
        command = self.commands.get(name, "")
        with self.lock:
            proc = self.processes.get(name)
            if proc and proc.poll() is None:
                return False, f"{name} 已在运行中。"
        if not command:
            return False, f"{name} 未配置可执行文件路径。"
        if not os.path.exists(command):
            return False, f"{name} 可执行文件不存在: {command}"

        try:
            proc = subprocess.Popen(
                [command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as e:
            self.log(f"启动 {name} 失败: {e}")
            return False, f"启动失败: {e}"

        with self.lock:
            self.processes[name] = proc
            self.statuses[name] = "Running"

            t = threading.Thread(
                target=self._read_output_thread,
                args=(name, proc),
                daemon=True,
            )
            t.start()
            self.output_threads[name] = t

        self.log(f"{name} 启动中，PID: {proc.pid}")
        return True, f"{name} 启动成功。"

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
                    self.log(f"[{name}] {line}", add_timestamp=False)
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
                return False, f"{name} 当前未运行。"

        try:
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.log(f"{name} 正常停止超时，尝试强制结束 (PID: {proc.pid})")
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
            self.log(f"停止 {name} 失败: {e}")
            return False, f"停止失败: {e}"

        with self.lock:
            self.statuses[name] = "Stopped"
            self.processes[name] = None
            self.output_threads.pop(name, None)
        self.log(f"{name} 已停止，PID: {proc.pid}")
        return True, f"{name} 已停止。"

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
        self.root.title("EvGuard Scheduler")
        self.root.geometry("860x580")
        self.root.configure(bg="#C0C0C0")
        self.root.resizable(True, True)

        self.manager = ProcessManager(load_commands())
        self.log_offset = 0
        self.running = True

        banner_frame = tk.Frame(self.root, width=860, height=44, bg="black")
        banner_frame.pack(side="top", fill="x", padx=1, pady=(0, 0))
        self._build_banner(banner_frame)

        self.main_frame = tk.Frame(self.root, bg="#C0C0C0")
        self.main_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.main_frame.grid_columnconfigure(0, weight=1, minsize=340)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._schedule_log_refresh()

    def _build_banner(self, banner_frame):
        try:
            banner_img = tk.PhotoImage(file="static/eve-ban.png")
            inner_banner = tk.Frame(banner_frame, bg="black")
            inner_banner.pack(expand=True, fill="both")
            banner_label = tk.Label(
                inner_banner,
                image=banner_img,
                bg="black",
                borderwidth=0,
                highlightthickness=0,
            )
            banner_label.image = banner_img
            banner_label.pack(side="left", padx=4)
        except Exception as e:
            tk.Label(
                banner_frame,
                text=f"Banner 加载失败: {e}",
                bg="black",
                fg="red",
                font=("MS Sans Serif", 10),
            ).pack(expand=True, fill="both")

    def _build_left_panel(self):
        self.left_frame = tk.LabelFrame(
            self.main_frame,
            text=" 功能列表 ",
            bg="#C0C0C0",
            fg="#000080",
            font=("MS Sans Serif", 10, "bold"),
            padx=6,
            pady=6,
        )
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.process_labels = {}
        for name in PROCESS_ORDER:
            row_frame = tk.Frame(self.left_frame, bg="#C0C0C0")
            row_frame.pack(fill="x", pady=6)

            name_label = tk.Label(
                row_frame,
                text=f" {name} ",
                bg="#76391d",
                fg="white",
                font=("MS Sans Serif", 10, "bold"),
                width=5,
                padx=8,
                pady=2,
                relief="raised",
                bd=1,
            )
            name_label.pack(side="left", padx=(6, 12))

            status = self.manager.get_status(name)
            status_label = tk.Label(
                row_frame,
                text=f"状态: {status}",
                bg="#C0C0C0",
                fg="#006400" if status == "Running" else "#800000",
                font=("MS Sans Serif", 8),
                anchor="w",
            )
            status_label.pack(side="left", fill="x", expand=True, padx=10)
            self.process_labels[name] = status_label

            btn_frame = tk.Frame(row_frame, bg="#C0C0C0")
            btn_frame.pack(side="right", padx=8)
            tk.Button(
                btn_frame,
                text="Start",
                width=8,
                font=("MS Sans Serif", 8),
                command=lambda n=name: self.start(n),
            ).pack(side="left", padx=(0, 4))
            tk.Button(
                btn_frame,
                text="Stop",
                width=8,
                font=("MS Sans Serif", 8),
                command=lambda n=name: self.stop(n),
            ).pack(side="left", padx=(4, 0))

            tk.Frame(self.left_frame, height=2, bg="#808080").pack(fill="x", pady=4)

        bottom_frame = tk.Frame(self.left_frame, bg="#C0C0C0")
        bottom_frame.pack(fill="x", pady=8)
        tk.Button(
            bottom_frame,
            text="刷新状态",
            width=12,
            font=("MS Sans Serif", 8),
            command=self.refresh,
        ).pack(side="left", padx=8)
        tk.Button(
            bottom_frame,
            text="全部退出",
            width=12,
            font=("MS Sans Serif", 8),
            command=self.exit_all,
        ).pack(side="left", padx=8)

    def _build_right_panel(self):
        self.right_frame = tk.LabelFrame(
            self.main_frame,
            text=" 控制台 ",
            bg="#C0C0C0",
            fg="#000080",
            font=("MS Sans Serif", 10, "bold"),
            padx=6,
            pady=6,
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        header = tk.Frame(self.right_frame, bg="#C0C0C0")
        header.pack(fill="x", pady=(0, 4))
        tk.Label(
            header,
            text="控制台输出",
            bg="#C0C0C0",
            fg="#000080",
            font=("MS Sans Serif", 10, "bold"),
        ).pack(side="left", padx=4)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            header,
            text="自动滚动",
            variable=self.auto_scroll_var,
            bg="#C0C0C0",
            font=("MS Sans Serif", 8),
        ).pack(side="left", padx=12)
        tk.Button(
            header,
            text="清空日志",
            width=10,
            font=("MS Sans Serif", 8),
            command=self.clear_logs,
        ).pack(side="left", padx=4)
        tk.Button(
            header,
            text="查看详细日志",
            width=14,
            font=("MS Sans Serif", 8),
            command=self.view_logs,
        ).pack(side="left", padx=4)

        self.console = scrolledtext.ScrolledText(
            self.right_frame,
            font=("Courier New", 9),
            bg="#FFFFFF",
            fg="#000000",
            wrap=tk.WORD,
            relief="sunken",
            bd=2,
        )
        self.console.pack(fill="both", expand=True)

    def start(self, name):
        ok, message = self.manager.start_process(name)
        self._update_one_status(name)
        if not ok:
            messagebox.showwarning("提示", message)

    def stop(self, name):
        ok, message = self.manager.stop_process(name)
        self._update_one_status(name)
        if not ok:
            messagebox.showinfo("提示", message)

    def refresh(self):
        self.manager.refresh_statuses()
        for name in PROCESS_ORDER:
            self._update_one_status(name)

    def exit_all(self):
        self.running = False
        self.manager.stop_all()
        self.refresh()
        self.root.quit()

    def clear_logs(self):
        self.manager.clear_logs()
        self.log_offset = 0
        self.console.delete("1.0", tk.END)

    def view_logs(self):
        content = self.manager.get_logs()
        if not content.strip():
            messagebox.showinfo("日志", "当前无日志内容。")
            return

        top = tk.Toplevel(self.root)
        top.title("详细日志 - EvGuard")
        top.geometry("720x520")
        top.configure(bg="#C0C0C0")

        text = scrolledtext.ScrolledText(top, font=("Courier New", 9), wrap=tk.WORD)
        text.insert("1.0", content)
        text.pack(fill="both", expand=True, padx=8, pady=8)

    def _update_one_status(self, name):
        status = self.manager.get_status(name)
        color = "#006400" if status == "Running" else "#800000"
        self.process_labels[name].config(text=f"状态: {status}", fg=color)

    def _schedule_log_refresh(self):
        if not self.running:
            return
        self.manager.refresh_statuses()
        for name in PROCESS_ORDER:
            self._update_one_status(name)

        data, self.log_offset = self.manager.get_logs_since(self.log_offset)
        if data:
            self.console.insert(tk.END, data)
            if self.auto_scroll_var.get():
                self.console.see(tk.END)

        self.root.after(500, self._schedule_log_refresh)

    def on_closing(self):
        self.running = False
        self.manager.stop_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EvGuardApp(root)
    root.mainloop()
