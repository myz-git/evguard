# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, "icon"), "icon"),
    (os.path.join(project_dir, "static"), "static"),
    (os.path.join(project_dir, "model_cnn"), "model_cnn"),
    (os.path.join(project_dir, "1.2"), "1.2"),
    (os.path.join(project_dir, "2.3"), "2.3"),
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

a = Analysis(
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
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
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="FSD0_only",
)
