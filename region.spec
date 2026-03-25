# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, "evguard.cfg"), "."),
    (os.path.join(project_dir, "1920_1080.png"), "."),
    (os.path.join(project_dir, "3440_1440.png"), "."),
]

hiddenimports = []
hiddenimports += collect_submodules("pyautogui")
hiddenimports += collect_submodules("pynput")
hiddenimports += collect_submodules("PIL")

excludes = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]

a = Analysis(
    [os.path.join(project_dir, "region.py")],
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
    name="region",
    icon=os.path.join(project_dir, "icon", "eva.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="region_pkg",
)
