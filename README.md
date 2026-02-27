



### 建立环境

```
cd D:\Workspace\git\
git clone git@github.com:myz-git/evguard.git

cd D:\Workspace\git\evguard\
conda create -n evguard python=3.11.7
conda activate evguard
```



### 在虚拟环境中安装包

```
pip install numpy==1.26.4
pip install opencv-python
pip install pillow
pip install pyautogui
pip install pynput
pip install cnocr
pip install pyttsx3
pip install matplotlib
pip install onnxruntime     
pip install pydub
pip install cryptography
pip install pyinstaller
##for cpu  torch
pip install torch torchvision --index-url   https://download.pytorch.org/whl/cpu

```



### 添加cnocr hook

为解决打包cnocr问题,需要在hooks目录下添加hook-cnocr.py文件(新建或复制现有的)

```
C:\Users\Administrator\.virtualenvs\evguard-2n2QTSJC\Lib\site-packages\PyInstaller\hooks
```



```
vi hook-cnocr.py

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("cnocr")
```



#### 打包: 

```
#清理
cd /d D:\Workspace\git\evguard
rmdir /s /q build
rmdir /s /q dist
del /q *.spec.bak 2>nul

#打包
pyinstaller --clean  -y ev_all.spec
```



##  Git提交

```

```



