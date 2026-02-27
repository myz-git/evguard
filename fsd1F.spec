# -*- mode: python ; coding: utf-8 -*-


import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# ✅ PyInstaller 提供的 spec 路径（比 __file__ 更可靠）
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, "icon"), "icon"),
    (os.path.join(project_dir, "model_cnn"), "model_cnn"),
]


# 这些库经常需要 hiddenimports（尤其 cnocr/torch/torchvision/pynput/pyautogui）
hiddenimports = []
hiddenimports += collect_submodules("cnocr")
hiddenimports += collect_submodules("cnstd")     # cnocr 依赖，建议加
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("pynput")
hiddenimports += collect_submodules("pyautogui")
hiddenimports += collect_submodules("rapidocr")

# cnocr/torch 可能还有数据文件（比如模型配置等），尽量收集
datas += collect_data_files("cnocr", include_py_files=False)
datas += collect_data_files("cnstd", include_py_files=False)
datas += collect_data_files("rapidocr", include_py_files=False)


# Windows 下 pyautogui 常用依赖（一般 pip 会装，但这里保险）
hiddenimports += ["PIL", "PIL.Image"]

# 为了减少体积：排除 torchvision 的 detection/optical_flow/segmentation 等大模块
# 注意：如果你未来真的用到这些模块，再删掉 excludes
excludes = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]

# ========== jump1 EXE ==========
a1 = Analysis(
    [os.path.join(project_dir, "jump1.py")],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz1 = PYZ(a1.pure, a1.zipped_data, cipher=block_cipher)

exe1 = EXE(
    pyz1,
    a1.scripts,
    a1.binaries,
    a1.zipfiles,
    a1.datas,
    [],
    name="FSD0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # 有 UPX 才会压缩，没装也能打包
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,      # 你要隐藏控制台就改 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ========== jump2 EXE ==========
a2 = Analysis(
    [os.path.join(project_dir, "jump2.py")],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz2 = PYZ(a2.pure, a2.zipped_data, cipher=block_cipher)

exe2 = EXE(
    pyz2,
    a2.scripts,
    a2.binaries,
    a2.zipfiles,
    a2.datas,
    [],
    name="FSD10",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)