# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, "icon"), "icon"),
    (os.path.join(project_dir, "static"), "static"),
    (os.path.join(project_dir, "model_cnn"), "model_cnn"),
]

hiddenimports = []
hiddenimports += collect_submodules("cnocr")
hiddenimports += collect_submodules("cnstd")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("pynput")
hiddenimports += collect_submodules("pyautogui")
hiddenimports += collect_submodules("rapidocr")
hiddenimports += ["PIL", "PIL.Image"]

datas += collect_data_files("cnocr", include_py_files=False)
datas += collect_data_files("cnstd", include_py_files=False)
datas += collect_data_files("rapidocr", include_py_files=False)

excludes = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]

# ---------- jump1 ----------
a1 = Analysis(
    [os.path.join(project_dir, "fsd0.py")],
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
    [],                # 关键：不把 binaries 塞进 exe
    [],                # 关键：不把 zipfiles 塞进 exe
    [],                # 关键：不把 datas 塞进 exe
    [],
    name="FSD0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,   # ✅ 关键：让依赖进入 COLLECT
)

# ---------- jump2 ----------
a2 = Analysis(
    [os.path.join(project_dir, "fsd10.py")],
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
    [],
    [],
    [],
    [],
    name="FSD10",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,   # ✅ 关键
)

# ✅ 关键：把两个 exe + 所有依赖/资源收进同一个目录，做到共享
coll = COLLECT(
    exe1,
    exe2,
    a1.binaries + a2.binaries,
    a1.zipfiles + a2.zipfiles,
    a1.datas + a2.datas,
    strip=False,
    upx=True,
    name="evfsd",
)