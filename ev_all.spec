# -*- mode: python ; coding: utf-8 -*-
import os
import rapidocr
import cnocr
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, "icon"), "icon"),
    (os.path.join(project_dir, "static"), "static"),
    (os.path.join(project_dir, "model_cnn"), "model_cnn"),
]

# 只补 rapidocr 真实缺失的 yaml（避免全量带 rapidocr 数据）
rapidocr_dir = os.path.dirname(rapidocr.__file__)
datas += [(os.path.join(rapidocr_dir, "default_models.yaml"), "rapidocr")]

datas += collect_data_files("cnocr", include_py_files=False)
datas += collect_data_files("cnstd", include_py_files=False)
datas += collect_data_files("rapidocr", include_py_files=False)


# 最小 hiddenimports：点名，不要 collect_submodules 全量递归
hiddenimports = [
    "torch",
    "torch.nn",
    "torchvision.transforms",
    "cnocr",
    "cnstd",
    "rapidocr",
    "pynput",
    "pyautogui",
    "PIL",
    "PIL.Image",
    "matplotlib",
]

excludes = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "polars",
    "_polars_runtime_32",
#    "scipy",
#    "scipy.libs",        
]

# ========= FSD0 =========
a_fsd0 = Analysis(
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
pyz_fsd0 = PYZ(a_fsd0.pure, a_fsd0.zipped_data, cipher=block_cipher)

exe_fsd0 = EXE(
    pyz_fsd0,
    a_fsd0.scripts,
    [], [], [], [],
    name="FSD0",
    icon=os.path.join(project_dir, "icon", "eva.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
    contents_directory="_internal",   # keep shared libs under _internal
)

# ========= FSD10 =========
a_fsd10 = Analysis(
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
pyz_fsd10 = PYZ(a_fsd10.pure, a_fsd10.zipped_data, cipher=block_cipher)

exe_fsd10 = EXE(
    pyz_fsd10,
    a_fsd10.scripts,
    [], [], [], [],
    name="FSD10",
    icon=os.path.join(project_dir, "icon", "eva.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
    contents_directory="_internal",
)

# ========= GuardA =========
a_ga = Analysis(
    [os.path.join(project_dir, "guarda.py")],
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
pyz_ga = PYZ(a_ga.pure, a_ga.zipped_data, cipher=block_cipher)

exe_ga = EXE(
    pyz_ga,
    a_ga.scripts,
    [], [], [], [],
    name="GuardA",
    icon=os.path.join(project_dir, "icon", "eva.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
    contents_directory="_internal",
)

# ========= GuardB =========
a_gb = Analysis(
    [os.path.join(project_dir, "guardb.py")],
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
pyz_gb = PYZ(a_gb.pure, a_gb.zipped_data, cipher=block_cipher)

exe_gb = EXE(
    pyz_gb,
    a_gb.scripts,
    [], [], [], [],
    name="GuardB",
    icon=os.path.join(project_dir, "icon", "eva.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
    contents_directory="_internal",
)

# ========= COLLECT: 4 EXEs + shared deps/resources =========
coll = COLLECT(
    exe_fsd0,
    exe_fsd10,
    exe_ga,
    exe_gb,
    a_fsd0.binaries + a_fsd10.binaries + a_ga.binaries + a_gb.binaries,
    a_fsd0.zipfiles + a_fsd10.zipfiles + a_ga.zipfiles + a_gb.zipfiles,
    a_fsd0.datas + a_fsd10.datas + a_ga.datas + a_gb.datas,
    strip=False,
    upx=False,
    name="FsdGuard",   # dist/FsdGuard/
)
