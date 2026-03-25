import os
import time
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Optional, Tuple

import pyautogui
from PIL import Image, ImageTk

from utils import _SCREEN_REGION_SPECS, resource_path

PREVIEW_RESOLUTIONS = {
    "1920*1080": {
        "size": (1920, 1080),
        "cfg_tag": "1920x1080",
        "background": "1920_1080.png",
    },
    "3440*1440": {
        "size": (3440, 1440),
        "cfg_tag": "3440x1440",
        "background": "3440_1440.png",
    },
}
DEFAULT_PREVIEW_RESOLUTION = "1920*1080"
BASE_RESOLUTION = PREVIEW_RESOLUTIONS[DEFAULT_PREVIEW_RESOLUTION]["size"]


CALIBRATED_REGION_NAMES = [
    "right_panel",
    "full_right_panel",
    "center_panel2",
    "mid_left_panel",
]

REGION_DISPLAY_NAMES = {
    "full_right_panel": "FSD自动导航识别区",
    "mid_left_panel": "FSD导航终点区",
    "right_panel": "GUARD白名监控区",
    "center_panel2": "GUARD紧急规避区",
}

REGION_DESCRIPTIONS = {
    "right_panel": "Guard 扫描右侧危险图标和文字的区域。",
    "full_right_panel": "FSD 扫描右侧跳门、离站等按钮的区域。",
    "center_panel2": "紧急规避时查找 PIN999 / PIN888 的中部区域。",
    "mid_left_panel": "FSD 查找终点、停靠和目的地提示的左中区域。",
}

CANVAS_MAX_WIDTH = 1280
CANVAS_MAX_HEIGHT = 760

BG_APP = "#0f141b"
BG_PANEL = "#161d27"
BG_CARD = "#1a2330"
BG_LIST = "#121821"
FG_PRIMARY = "#edf3fb"
FG_SECONDARY = "#9fb0c7"
FG_MUTED = "#7c8aa3"
ACCENT = "#39ff14"
ACCENT_SOFT = "#46c8ff"
BORDER = "#2a3546"
WARN = "#ffd84d"


def parse_region_rect(value: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        parts = [int(part.strip()) for part in str(value).split(",")]
    except Exception:
        return None
    if len(parts) != 4:
        return None
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def cfg_file_path() -> str:
    return resource_path("evguard.cfg")


def _cfg_tag_to_size(cfg_tag: str) -> Optional[Tuple[int, int]]:
    for spec in PREVIEW_RESOLUTIONS.values():
        if spec["cfg_tag"] == cfg_tag:
            return spec["size"]
    return None


def _size_to_resolution_label(size: Tuple[int, int]) -> Optional[str]:
    for label, spec in PREVIEW_RESOLUTIONS.items():
        if spec["size"] == size:
            return label
    return None


def _scale_by_height(value: float, scale_y: float) -> int:
    return int(round(value * scale_y))


def _clamp_region(rect: Tuple[int, int, int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    screen_width, screen_height = size
    x, y, w, h = rect
    x = max(0, min(int(x), max(0, screen_width - 1)))
    y = max(0, min(int(y), max(0, screen_height - 1)))
    w = max(1, min(int(w), max(1, screen_width - x)))
    h = max(1, min(int(h), max(1, screen_height - y)))
    return (x, y, w, h)


def _convert_base_rect_to_image(
    region_name: str,
    rect: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    base_width, base_height = BASE_RESOLUTION
    image_width, image_height = image_size
    if image_width == base_width and image_height == base_height:
        return tuple(map(int, rect))

    anchor = _SCREEN_REGION_SPECS.get(region_name, {}).get("anchor", "scale")
    scale_y = image_height / float(base_height)
    x, y, w, h = rect
    scaled_w = _scale_by_height(w, scale_y)
    scaled_h = _scale_by_height(h, scale_y)
    scaled_y = _scale_by_height(y, scale_y)

    if anchor == "right":
        right_margin = base_width - (x + w)
        scaled_x = image_width - scaled_w - _scale_by_height(right_margin, scale_y)
    elif anchor == "center":
        base_center_x = x + (w / 2.0)
        center_offset = base_center_x - (base_width / 2.0)
        scaled_center_x = (image_width / 2.0) + (center_offset * scale_y)
        scaled_x = int(round(scaled_center_x - (scaled_w / 2.0)))
    else:
        scaled_x = _scale_by_height(x, scale_y)

    return _clamp_region((scaled_x, scaled_y, scaled_w, scaled_h), image_size)


def _convert_image_rect_to_base(
    region_name: str,
    rect: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    base_width, base_height = BASE_RESOLUTION
    image_width, image_height = image_size
    if image_width == base_width and image_height == base_height:
        return tuple(map(int, rect))

    anchor = _SCREEN_REGION_SPECS.get(region_name, {}).get("anchor", "scale")
    scale_y = image_height / float(base_height)
    x, y, w, h = rect
    base_w = max(1, int(round(w / scale_y)))
    base_h = max(1, int(round(h / scale_y)))
    base_y = int(round(y / scale_y))

    if anchor == "right":
        right_margin = image_width - (x + w)
        base_right_margin = int(round(right_margin / scale_y))
        base_x = base_width - base_w - base_right_margin
    elif anchor == "center":
        image_center_x = x + (w / 2.0)
        center_offset = (image_center_x - (image_width / 2.0)) / scale_y
        base_center_x = (base_width / 2.0) + center_offset
        base_x = int(round(base_center_x - (base_w / 2.0)))
    else:
        base_x = int(round(x / scale_y))

    return _clamp_region((base_x, base_y, base_w, base_h), BASE_RESOLUTION)


def _default_regions_for_size(size: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
    return {
        name: _convert_base_rect_to_image(name, tuple(_SCREEN_REGION_SPECS[name]["rect"]), size)
        for name in CALIBRATED_REGION_NAMES
        if name in _SCREEN_REGION_SPECS
    }


def _parse_cfg_region_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = key.strip().lower()
    if not normalized.startswith("region_"):
        return (None, None)

    body = normalized[len("region_"):]
    for spec in PREVIEW_RESOLUTIONS.values():
        prefix = f"{spec['cfg_tag']}_"
        if body.startswith(prefix):
            region_name = body[len(prefix):]
            return (spec["cfg_tag"], region_name)

    return (None, body)


def load_cfg_regions() -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
    regions = {
        spec["cfg_tag"]: _default_regions_for_size(spec["size"])
        for spec in PREVIEW_RESOLUTIONS.values()
    }
    path = cfg_file_path()
    if not os.path.exists(path):
        return regions

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                cfg_tag, region_name = _parse_cfg_region_key(key)
                if not region_name:
                    continue
                rect = parse_region_rect(value)
                if not rect:
                    continue
                if cfg_tag and cfg_tag in regions and region_name in regions[cfg_tag]:
                    regions[cfg_tag][region_name] = rect
                elif cfg_tag is None and region_name in regions[PREVIEW_RESOLUTIONS[DEFAULT_PREVIEW_RESOLUTION]["cfg_tag"]]:
                    regions[PREVIEW_RESOLUTIONS[DEFAULT_PREVIEW_RESOLUTION]["cfg_tag"]][region_name] = rect
    except Exception:
        pass
    return regions


def save_cfg_regions(regions_by_resolution: Dict[str, Dict[str, Tuple[int, int, int, int]]]) -> str:
    path = cfg_file_path()
    original_lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            original_lines = f.read().splitlines()

    pending = {}
    for label, spec in PREVIEW_RESOLUTIONS.items():
        cfg_tag = spec["cfg_tag"]
        regions = regions_by_resolution.get(cfg_tag, {})
        for name in CALIBRATED_REGION_NAMES:
            rect = regions.get(name)
            if rect:
                pending[f"region_{cfg_tag}_{name}"] = f"{rect[0]},{rect[1]},{rect[2]},{rect[3]}"
    output_lines = []
    seen_keys = set()

    for raw_line in original_lines:
        stripped = raw_line.strip()
        if "=" not in stripped or stripped.startswith("#"):
            output_lines.append(raw_line)
            continue
        key, _ = stripped.split("=", 1)
        normalized_key = key.strip().lower()
        cfg_tag, region_name = _parse_cfg_region_key(normalized_key)
        if normalized_key in pending:
            output_lines.append(f"{key.strip()}={pending[normalized_key]}")
            seen_keys.add(normalized_key)
        elif cfg_tag is not None and region_name in CALIBRATED_REGION_NAMES:
            continue
        elif cfg_tag is None and region_name in CALIBRATED_REGION_NAMES:
            continue
        else:
            output_lines.append(raw_line)

    if output_lines and output_lines[-1].strip():
        output_lines.append("")

    if pending:
        if "region_1920x1080_right_panel" not in seen_keys and not any(
            line.strip().startswith("# Region overrides") for line in output_lines
        ):
            output_lines.append("# Region overrides")
        for key, value in pending.items():
            if key in seen_keys:
                continue
            output_lines.append(f"{key}={value}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines).rstrip() + "\n")

    return path


class RegionCalibratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("区域校准工具")
        self.root.geometry("1640x940")
        self.root.configure(bg=BG_APP)
        self.root.minsize(1460, 820)

        self.default_regions_by_resolution = {
            spec["cfg_tag"]: _default_regions_for_size(spec["size"])
            for spec in PREVIEW_RESOLUTIONS.values()
        }
        self.regions_by_resolution = load_cfg_regions()
        self.active_resolution_label = DEFAULT_PREVIEW_RESOLUTION
        self.active_resolution_tag = PREVIEW_RESOLUTIONS[self.active_resolution_label]["cfg_tag"]

        self.selected_region_name = CALIBRATED_REGION_NAMES[0]
        self.screenshot_image: Optional[Image.Image] = None
        self.tk_image = None
        self.display_scale = 1.0
        self.drag_start = None
        self.drag_current = None
        self.drag_mode = None
        self.drag_origin_image = None
        self.drag_original_rect = None

        self.var_x = tk.StringVar()
        self.var_y = tk.StringVar()
        self.var_w = tk.StringVar()
        self.var_h = tk.StringVar()
        self.status_var = tk.StringVar(value=f"cfg: {cfg_file_path()}")
        self.region_desc_var = tk.StringVar()
        self.preview_resolution_var = tk.StringVar(value=self.active_resolution_label)
        self.canvas_title_var = tk.StringVar(value="未加载截图")

        self._build_ui()
        self._sync_active_resolution()
        self._load_selected_region_to_form()
        self.load_background_for_active_resolution()

    def _sync_active_resolution(self) -> None:
        self.active_resolution_label = self.preview_resolution_var.get()
        self.active_resolution_tag = PREVIEW_RESOLUTIONS[self.active_resolution_label]["cfg_tag"]

    def _current_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        return self.regions_by_resolution[self.active_resolution_tag]

    def _current_defaults(self) -> Dict[str, Tuple[int, int, int, int]]:
        return self.default_regions_by_resolution[self.active_resolution_tag]

    def _create_button(self, parent, text, command, primary=False, width=14):
        bg = "#2449d8" if primary else "#243041"
        active = "#3059f6" if primary else "#324257"
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=bg,
            fg=FG_PRIMARY,
            activebackground=active,
            activeforeground=FG_PRIMARY,
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
            cursor="hand2",
        )

    def _build_card(self, parent, title: str):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        header = tk.Frame(card, bg=BG_CARD)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text=title,
            bg=BG_CARD,
            fg=FG_PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w")
        return card

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=70)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        title_wrap = tk.Frame(header, bg=BG_PANEL)
        title_wrap.pack(side="left", padx=20)
        tk.Label(
            title_wrap,
            text="扫描区域校准",
            bg=BG_PANEL,
            fg=FG_PRIMARY,
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_wrap,
            text="调整后保存到 evguard.cfg，主程序下次启动自动生效。",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        toolbar = tk.Frame(header, bg=BG_PANEL)
        toolbar.pack(side="right", padx=18)
        self._create_button(toolbar, "截图当前屏幕", self.capture_screen, primary=True).pack(side="left", padx=4)
        self._create_button(toolbar, "载入 evguard.cfg", self.reload_from_cfg).pack(side="left", padx=4)
        self._create_button(toolbar, "保存到 evguard.cfg", self.save_to_cfg).pack(side="left", padx=4)
        self._create_button(toolbar, "恢复默认", self.restore_defaults).pack(side="left", padx=4)
        self._create_button(toolbar, "退出", self.root.destroy, width=10).pack(side="left", padx=4)

        body = tk.Frame(self.root, bg=BG_APP)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        sidebar = tk.Frame(body, bg=BG_APP, width=320)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        region_card = self._build_card(sidebar, "区域")
        region_card.pack(fill="x")
        self.region_listbox = tk.Listbox(
            region_card,
            exportselection=False,
            width=28,
            height=max(4, len(CALIBRATED_REGION_NAMES)),
            bg=BG_LIST,
            fg=FG_PRIMARY,
            selectbackground="#2449d8",
            selectforeground=FG_PRIMARY,
            relief="flat",
            highlightthickness=0,
            activestyle="none",
            font=("Microsoft YaHei UI", 10),
        )
        for name in CALIBRATED_REGION_NAMES:
            self.region_listbox.insert("end", REGION_DISPLAY_NAMES.get(name, name))
        self.region_listbox.selection_set(0)
        self.region_listbox.bind("<<ListboxSelect>>", self.on_select_region)
        self.region_listbox.pack(fill="x", padx=14, pady=(0, 10))

        detail_content = tk.Frame(region_card, bg=BG_CARD)
        detail_content.pack(fill="x", padx=14, pady=(0, 14))

        tk.Frame(detail_content, bg=BORDER, height=1).pack(fill="x", pady=(0, 12))

        tk.Label(
            detail_content,
            text="当前区域",
            bg=BG_CARD,
            fg=FG_SECONDARY,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.current_region_label = tk.Label(
            detail_content,
            text="",
            anchor="w",
            bg=BG_CARD,
            fg=FG_PRIMARY,
            font=("Consolas", 13, "bold"),
        )
        self.current_region_label.pack(fill="x")

        tk.Label(
            detail_content,
            textvariable=self.region_desc_var,
            justify="left",
            anchor="w",
            wraplength=300,
            bg=BG_CARD,
            fg=FG_SECONDARY,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", pady=(8, 10))

        form_grid = tk.Frame(detail_content, bg=BG_CARD)
        form_grid.pack(fill="x")
        for idx, (label, var) in enumerate((("x", self.var_x), ("y", self.var_y), ("w", self.var_w), ("h", self.var_h))):
            row = idx // 2
            col = (idx % 2) * 2
            tk.Label(
                form_grid,
                text=label.upper(),
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Consolas", 10, "bold"),
            ).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=5)
            entry = tk.Entry(
                form_grid,
                textvariable=var,
                width=11,
                bg=BG_LIST,
                fg=FG_PRIMARY,
                insertbackground=FG_PRIMARY,
                relief="flat",
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=ACCENT_SOFT,
                font=("Consolas", 11),
            )
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=5)
        form_grid.grid_columnconfigure(1, weight=1)
        form_grid.grid_columnconfigure(3, weight=1)

        action_bar = tk.Frame(detail_content, bg=BG_CARD)
        action_bar.pack(fill="x", pady=(12, 0))
        self._create_button(action_bar, "应用输入", self.apply_form, primary=True, width=11).pack(side="left")
        self._create_button(action_bar, "恢复默认", self.restore_selected_default, width=11).pack(side="left", padx=(8, 0))

        help_card = self._build_card(sidebar, "操作说明")
        help_card.pack(fill="x", pady=(12, 0))
        help_text = (
            "1. 加载背景图或截图当前游戏界面。\n"
            "2. 左侧选中一个区域。\n"
            "3. 画布里拖拽可重新框选。\n"
            "4. 拖住框内部可整体移动。\n"
            "5. 保存后写入 evguard.cfg。"
        )
        tk.Label(
            help_card,
            text=help_text,
            justify="left",
            anchor="w",
            wraplength=300,
            bg=BG_CARD,
            fg=FG_SECONDARY,
            font=("Microsoft YaHei UI", 10),
        ).pack(fill="x", padx=14, pady=(0, 14))

        canvas_wrap = tk.Frame(body, bg=BG_APP)
        canvas_wrap.pack(side="left", fill="both", expand=True)

        canvas_header = tk.Frame(canvas_wrap, bg=BG_APP)
        canvas_header.pack(fill="x", pady=(0, 8))
        tk.Label(
            canvas_header,
            text="预览画布",
            bg=BG_APP,
            fg=FG_PRIMARY,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(side="left")
        resolution_menu = tk.OptionMenu(
            canvas_header,
            self.preview_resolution_var,
            *PREVIEW_RESOLUTIONS.keys(),
            command=self.on_preview_resolution_change,
        )
        resolution_menu.config(
            bg=BG_LIST,
            fg=FG_PRIMARY,
            activebackground="#243041",
            activeforeground=FG_PRIMARY,
            highlightthickness=1,
            highlightbackground=BORDER,
            relief="flat",
            font=("Consolas", 10, "bold"),
            width=12,
        )
        resolution_menu["menu"].config(
            bg=BG_LIST,
            fg=FG_PRIMARY,
            activebackground="#2449d8",
            activeforeground=FG_PRIMARY,
            font=("Consolas", 10),
        )
        resolution_menu.pack(side="left", padx=(12, 0))
        tk.Label(
            canvas_header,
            textvariable=self.canvas_title_var,
            bg=BG_APP,
            fg=FG_MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(side="left", padx=(12, 0))

        canvas_card = tk.Frame(canvas_wrap, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        canvas_card.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_card,
            bg="#0c1118",
            width=CANVAS_MAX_WIDTH,
            height=CANVAS_MAX_HEIGHT,
            cursor="crosshair",
            relief="flat",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True, padx=12, pady=12)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Microsoft YaHei UI", 10),
            padx=16,
            pady=10,
        )
        status_bar.pack(fill="x", side="bottom")

    def _render_loaded_image(self):
        if self.screenshot_image is None:
            self.tk_image = None
            self.canvas_title_var.set("未加载截图")
            self.redraw_canvas()
            return

        img_w, img_h = self.screenshot_image.size
        self.display_scale = min(CANVAS_MAX_WIDTH / img_w, CANVAS_MAX_HEIGHT / img_h, 1.0)
        display_size = (max(1, int(img_w * self.display_scale)), max(1, int(img_h * self.display_scale)))
        display_image = self.screenshot_image.resize(display_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.config(width=display_size[0], height=display_size[1])
        self.canvas_title_var.set(f"{img_w} x {img_h}")
        self.redraw_canvas()

    def capture_screen(self):
        self.status_var.set("3 秒后截图，请切回游戏界面...")
        self.root.update_idletasks()
        self.root.update()
        self.root.iconify()
        for remaining in (3, 2, 1):
            print(f"区域校准截图倒计时: {remaining}")
            time.sleep(1)
        self.screenshot_image = pyautogui.screenshot().convert("RGB")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._render_loaded_image()
        img_w, img_h = self.screenshot_image.size
        self.status_var.set(f"已截图: {img_w}x{img_h}")

    def load_background_for_active_resolution(self):
        for filename in DEFAULT_BACKGROUND_CANDIDATES:
            path = resource_path(filename)
            if not os.path.exists(path):
                continue
            try:
                self.screenshot_image = Image.open(path).convert("RGB")
            except Exception:
                continue
            self._render_loaded_image()
            img_w, img_h = self.screenshot_image.size
            self.status_var.set(f"已加载背景图: {filename} ({img_w}x{img_h})")
            return

    def reload_from_cfg(self):
        self.regions = dict(self.default_regions)
        self.regions.update(load_cfg_regions())
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"已载入: {cfg_file_path()}")

    def restore_defaults(self):
        self.regions = dict(self.default_regions)
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set("已恢复所有默认区域")

    def restore_selected_default(self):
        self.regions[self.selected_region_name] = self.default_regions[self.selected_region_name]
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"已恢复默认: {self.selected_region_name}")

    def save_to_cfg(self):
        path = save_cfg_regions(self.regions, coord_mode=REGION_COORD_MODE_BASE)
        self.status_var.set(f"已保存: {path}")
        messagebox.showinfo("已保存", f"校准结果已写入：\n{path}")

    def on_select_region(self, _event=None):
        selection = self.region_listbox.curselection()
        if not selection:
            return
        self.selected_region_name = CALIBRATED_REGION_NAMES[selection[0]]
        self._load_selected_region_to_form()
        self.redraw_canvas()

    def _load_selected_region_to_form(self):
        rect = self._region_to_image_space(self.selected_region_name, self.regions[self.selected_region_name])
        self.current_region_label.config(
            text=REGION_DISPLAY_NAMES.get(self.selected_region_name, self.selected_region_name)
        )
        self.region_desc_var.set(REGION_DESCRIPTIONS.get(self.selected_region_name, ""))
        self.var_x.set(str(rect[0]))
        self.var_y.set(str(rect[1]))
        self.var_w.set(str(rect[2]))
        self.var_h.set(str(rect[3]))

    def apply_form(self):
        try:
            rect = (
                int(self.var_x.get()),
                int(self.var_y.get()),
                int(self.var_w.get()),
                int(self.var_h.get()),
            )
        except ValueError:
            messagebox.showerror("输入错误", "x / y / w / h 必须是整数。")
            return

        if rect[2] <= 0 or rect[3] <= 0:
            messagebox.showerror("输入错误", "w / h 必须大于 0。")
            return

        self.regions[self.selected_region_name] = self._region_from_image_space(self.selected_region_name, rect)
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"已更新: {self.selected_region_name}={rect}")

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.tk_image is not None:
            self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        else:
            self.canvas.create_text(
                CANVAS_MAX_WIDTH / 2,
                CANVAS_MAX_HEIGHT / 2,
                text="未加载截图\n点击顶部“截图当前屏幕”或使用默认背景图",
                fill=FG_MUTED,
                font=("Microsoft YaHei UI", 14),
                justify="center",
            )

        for name in CALIBRATED_REGION_NAMES:
            rect = self.regions.get(name)
            if not rect:
                continue
            selected = name == self.selected_region_name
            self._draw_rect(
                self._region_to_image_space(name, rect),
                color=ACCENT if selected else ACCENT_SOFT,
                width=4 if selected else 1,
                label=name if selected else None,
            )

        if self.drag_start and self.drag_current:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=WARN, width=2, dash=(6, 3))

    def _draw_rect(self, rect, color: str, width: int, label: Optional[str] = None):
        x, y, w, h = rect
        sx0 = x * self.display_scale
        sy0 = y * self.display_scale
        sx1 = (x + w) * self.display_scale
        sy1 = (y + h) * self.display_scale
        self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline=color, width=width)
        if label:
            self.canvas.create_rectangle(sx0, max(0, sy0 - 26), sx0 + 160, sy0, fill="#0b1118", outline=color, width=1)
            self.canvas.create_text(
                sx0 + 8,
                max(0, sy0 - 13),
                text=label,
                anchor="w",
                fill=color,
                font=("Consolas", 11, "bold"),
            )

    def _canvas_to_image_xy(self, canvas_x, canvas_y):
        if self.display_scale <= 0:
            return (0, 0)
        img_x = int(canvas_x / self.display_scale)
        img_y = int(canvas_y / self.display_scale)
        if self.screenshot_image:
            img_w, img_h = self.screenshot_image.size
            img_x = max(0, min(img_x, img_w - 1))
            img_y = max(0, min(img_y, img_h - 1))
        return (img_x, img_y)

    def on_canvas_press(self, event):
        image_point = self._canvas_to_image_xy(event.x, event.y)
        selected_rect = self.regions.get(self.selected_region_name)
        selected_rect_image = (
            self._region_to_image_space(self.selected_region_name, selected_rect) if selected_rect else None
        )
        if selected_rect_image and self._point_in_rect(image_point, selected_rect_image):
            self.drag_mode = "move"
            self.drag_origin_image = image_point
            self.drag_original_rect = selected_rect_image
            self.drag_start = None
            self.drag_current = None
        else:
            self.drag_mode = "draw"
            self.drag_start = (event.x, event.y)
            self.drag_current = (event.x, event.y)
        self.redraw_canvas()

    def on_canvas_drag(self, event):
        if self.drag_mode == "move":
            if not self.drag_origin_image or not self.drag_original_rect or not self.screenshot_image:
                return
            img_x, img_y = self._canvas_to_image_xy(event.x, event.y)
            dx = img_x - self.drag_origin_image[0]
            dy = img_y - self.drag_origin_image[1]
            ox, oy, ow, oh = self.drag_original_rect
            max_x = max(0, self.screenshot_image.size[0] - ow)
            max_y = max(0, self.screenshot_image.size[1] - oh)
            new_rect = (
                max(0, min(ox + dx, max_x)),
                max(0, min(oy + dy, max_y)),
                ow,
                oh,
            )
            self.regions[self.selected_region_name] = self._region_from_image_space(self.selected_region_name, new_rect)
            self._load_selected_region_to_form()
            self.redraw_canvas()
            return

        if self.drag_mode == "draw" and self.drag_start:
            self.drag_current = (event.x, event.y)
            self.redraw_canvas()

    def on_canvas_release(self, event):
        if self.drag_mode == "move":
            self.drag_mode = None
            self.drag_origin_image = None
            self.drag_original_rect = None
            self.status_var.set(f"已拖动: {self.selected_region_name}={self.regions[self.selected_region_name]}")
            self.redraw_canvas()
            return

        if self.drag_mode != "draw" or not self.drag_start:
            self.drag_mode = None
            return

        self.drag_current = (event.x, event.y)
        x0, y0 = self._canvas_to_image_xy(*self.drag_start)
        x1, y1 = self._canvas_to_image_xy(*self.drag_current)
        self.drag_start = None
        self.drag_current = None
        self.drag_mode = None

        left = min(x0, x1)
        top = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if width < 5 or height < 5:
            self.redraw_canvas()
            return

        self.regions[self.selected_region_name] = self._region_from_image_space(
            self.selected_region_name,
            (left, top, width, height),
        )
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"已框选: {self.selected_region_name}={(left, top, width, height)}")

    def capture_screen(self):
        self.status_var.set("3 绉掑悗鎴浘锛岃鍒囧洖娓告垙鐣岄潰...")
        self.root.update_idletasks()
        self.root.update()
        self.root.iconify()
        for remaining in (3, 2, 1):
            print(f"鍖哄煙鏍″噯鎴浘鍊掕鏃? {remaining}")
            time.sleep(1)
        self.screenshot_image = pyautogui.screenshot().convert("RGB")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        match_label = _size_to_resolution_label(self.screenshot_image.size)
        if match_label:
            self.preview_resolution_var.set(match_label)
            self._sync_active_resolution()
        self._render_loaded_image()
        img_w, img_h = self.screenshot_image.size
        self.status_var.set(f"宸叉埅鍥? {img_w}x{img_h}")

    def load_background_for_active_resolution(self):
        filename = PREVIEW_RESOLUTIONS[self.active_resolution_label]["background"]
        path = resource_path(filename)
        if not os.path.exists(path):
            self.screenshot_image = None
            self._render_loaded_image()
            self.status_var.set(f"Background missing: {filename}")
            return
        try:
            self.screenshot_image = Image.open(path).convert("RGB")
        except Exception:
            self.screenshot_image = None
            self._render_loaded_image()
            self.status_var.set(f"Background load failed: {filename}")
            return
        self._render_loaded_image()
        img_w, img_h = self.screenshot_image.size
        self.status_var.set(f"Loaded background: {filename} ({img_w}x{img_h})")

    def on_preview_resolution_change(self, _value=None):
        self._sync_active_resolution()
        self.load_background_for_active_resolution()
        self._load_selected_region_to_form()
        self.redraw_canvas()

    def reload_from_cfg(self):
        self.regions_by_resolution = load_cfg_regions()
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"Loaded cfg: {cfg_file_path()}")

    def restore_defaults(self):
        self.regions_by_resolution = {
            cfg_tag: dict(defaults)
            for cfg_tag, defaults in self.default_regions_by_resolution.items()
        }
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set("Restored all defaults")

    def restore_selected_default(self):
        self._current_regions()[self.selected_region_name] = self._current_defaults()[self.selected_region_name]
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(
            f"Restored default: {self.active_resolution_label} {self.selected_region_name}"
        )

    def save_to_cfg(self):
        path = save_cfg_regions(self.regions_by_resolution)
        self.status_var.set(f"Saved: {path}")
        messagebox.showinfo("Saved", f"Calibration saved to:\n{path}")

    def on_select_region(self, _event=None):
        selection = self.region_listbox.curselection()
        if not selection:
            return
        self.selected_region_name = CALIBRATED_REGION_NAMES[selection[0]]
        self._load_selected_region_to_form()
        self.redraw_canvas()

    def _load_selected_region_to_form(self):
        rect = self._current_regions()[self.selected_region_name]
        self.current_region_label.config(text=self.selected_region_name)
        self.region_desc_var.set(REGION_DESCRIPTIONS.get(self.selected_region_name, ""))
        self.var_x.set(str(rect[0]))
        self.var_y.set(str(rect[1]))
        self.var_w.set(str(rect[2]))
        self.var_h.set(str(rect[3]))

    def apply_form(self):
        try:
            rect = (
                int(self.var_x.get()),
                int(self.var_y.get()),
                int(self.var_w.get()),
                int(self.var_h.get()),
            )
        except ValueError:
            messagebox.showerror("Input Error", "x / y / w / h must be integers.")
            return

        if rect[2] <= 0 or rect[3] <= 0:
            messagebox.showerror("Input Error", "w / h must be greater than 0.")
            return

        if self.screenshot_image is not None:
            rect = _clamp_region(rect, self.screenshot_image.size)
        else:
            rect = _clamp_region(rect, PREVIEW_RESOLUTIONS[self.active_resolution_label]["size"])

        self._current_regions()[self.selected_region_name] = rect
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"Updated: {self.active_resolution_label} {self.selected_region_name}={rect}")

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.tk_image is not None:
            self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        else:
            self.canvas.create_text(
                CANVAS_MAX_WIDTH / 2,
                CANVAS_MAX_HEIGHT / 2,
                text="No preview image loaded.\nUse the resolution list or capture the current screen.",
                fill=FG_MUTED,
                font=("Microsoft YaHei UI", 14),
                justify="center",
            )

        for name in CALIBRATED_REGION_NAMES:
            rect = self._current_regions().get(name)
            if not rect:
                continue
            selected = name == self.selected_region_name
            self._draw_rect(
                rect,
                color=ACCENT if selected else ACCENT_SOFT,
                width=4 if selected else 1,
                label=name if selected else None,
            )

        if self.drag_start and self.drag_current:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=WARN, width=2, dash=(6, 3))

    def on_canvas_press(self, event):
        image_point = self._canvas_to_image_xy(event.x, event.y)
        selected_rect = self._current_regions().get(self.selected_region_name)
        if selected_rect and self._point_in_rect(image_point, selected_rect):
            self.drag_mode = "move"
            self.drag_origin_image = image_point
            self.drag_original_rect = selected_rect
            self.drag_start = None
            self.drag_current = None
        else:
            self.drag_mode = "draw"
            self.drag_start = (event.x, event.y)
            self.drag_current = (event.x, event.y)
        self.redraw_canvas()

    def on_canvas_drag(self, event):
        if self.drag_mode == "move":
            if not self.drag_origin_image or not self.drag_original_rect or not self.screenshot_image:
                return
            img_x, img_y = self._canvas_to_image_xy(event.x, event.y)
            dx = img_x - self.drag_origin_image[0]
            dy = img_y - self.drag_origin_image[1]
            ox, oy, ow, oh = self.drag_original_rect
            max_x = max(0, self.screenshot_image.size[0] - ow)
            max_y = max(0, self.screenshot_image.size[1] - oh)
            new_rect = (
                max(0, min(ox + dx, max_x)),
                max(0, min(oy + dy, max_y)),
                ow,
                oh,
            )
            self._current_regions()[self.selected_region_name] = new_rect
            self._load_selected_region_to_form()
            self.redraw_canvas()
            return

        if self.drag_mode == "draw" and self.drag_start:
            self.drag_current = (event.x, event.y)
            self.redraw_canvas()

    def on_canvas_release(self, event):
        if self.drag_mode == "move":
            self.drag_mode = None
            self.drag_origin_image = None
            self.drag_original_rect = None
            self.status_var.set(
                f"Moved: {self.active_resolution_label} {self.selected_region_name}={self._current_regions()[self.selected_region_name]}"
            )
            self.redraw_canvas()
            return

        if self.drag_mode != "draw" or not self.drag_start:
            self.drag_mode = None
            return

        self.drag_current = (event.x, event.y)
        x0, y0 = self._canvas_to_image_xy(*self.drag_start)
        x1, y1 = self._canvas_to_image_xy(*self.drag_current)
        self.drag_start = None
        self.drag_current = None
        self.drag_mode = None

        left = min(x0, x1)
        top = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if width < 5 or height < 5:
            self.redraw_canvas()
            return

        rect = (left, top, width, height)
        self._current_regions()[self.selected_region_name] = rect
        self._load_selected_region_to_form()
        self.redraw_canvas()
        self.status_var.set(f"Selected: {self.active_resolution_label} {self.selected_region_name}={rect}")

    @staticmethod
    def _point_in_rect(point, rect):
        px, py = point
        x, y, w, h = rect
        return x <= px <= x + w and y <= py <= y + h


def main():
    root = tk.Tk()
    RegionCalibratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
