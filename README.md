



## 1. 软件功能

### 1.1 FSD

实现端到端自动导航,详细见<FSD说明.pdf>

### 1.2  Guard

实现被动预警,详细参见<GuardA.pdf>
实现主动防御,详细参见<GuardB.pdf>



## 2. 初始化仓库

### 2.1 准备：GitHub 上建一个空仓库

在 GitHub 创建仓库 `evguard`，**不要勾选** README/.gitignore/license（保持空仓库）。
 拿到地址：`https://github.com/myz-git/evguard.git`

------

### 2.2 进入项目目录

```
cd /d D:\Workspace\git\evguard
```

------

### 2.3 彻底重来：删掉旧 .git (可选但推荐）

如果你确定要重新来（你之前就是这个诉求）：

```
rmdir /s /q .git
```

------

### 2.4 初始化仓库 + 固化 CRLF 规则 + 忽略构建产物

####  init + main

```
git init
git branch -M main
```

#### 写 `.gitattributes`（**解决 CRLF/LF 的关键**）

用记事本新建 `D:\Workspace\git\evguard\.gitattributes`，内容如下（直接粘贴）：

```
* text=auto eol=lf
*.bat text eol=crlf
*.cmd text eol=crlf
*.ps1 text eol=crlf
```

####  写 `.gitignore`（避免 build/ 之类污染仓库）

新建 `D:\Workspace\git\evguard\.gitignore`，内容如下：

```
*.log
*.pyc
.DS_Store
license*
__pycache__/
local/
pic/
traindata/*
machine/*
model/*
build/
dist/
debug_icons/
licensing/
keys/
~

```

------

### 2.5 第一次提交

#### 先提交规则文件

```
git add .gitattributes .gitignore
git commit -m "chore: add attributes and ignore"
```

#### 再把项目文件全部加入并提交

```
git add .
git commit -m "feat: initial commit"
```

#### 归一化换行符（**CRLF/LF 收尾**）

```
git add --renormalize .
git commit -m "chore: normalize line endings"
```

> 如果最后这条提示 “nothing to commit”，也没关系，说明已经是规范状态了。

------

### 2.6 绑定 GitHub 远端并 push

```
git remote add origin https://github.com/myz-git/evguard.git
git push -u origin main

#如果提示
! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'https://github.com/myz-git/evguard.git'
#可以强制用本地覆盖远端
git push -u --force origin main

git status
git remote -v
```

## 3. 建立虚拟环境

```
cd /d D:\Workspace\git\evguard
conda create -n evguard python=3.11.7
conda activate evguard
```

在虚拟环境中安装包

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

添加cnocr hook

为解决打包cnocr问题,需要在hooks目录下添加hook-cnocr.py文件(新建或复制现有的)
注意替换evguard-2n2QTSJC 这个路径名为实际名

```
C:\Users\Administrator\.virtualenvs\evguard-2n2QTSJC\Lib\site-packages\PyInstaller\hooks
```

```
notepad  hook-cnocr.py

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("cnocr")
```

## 4. 打包

### 4.1 本地打包

```
#清理
cd /d D:\Workspace\git\evguard
rmdir /s /q build
rmdir /s /q dist
del /q *.spec.bak 2>nul
pyinstaller --clean  -y ev_all.spec
```



### 4.2 更新git 版本

完成代码改动-> 更新Readme.md-> commit -> tag->push

```
# 1. 添加变更
git add .

# 2. 提交变更
## 查看历史log及当前tag
git log
git tag -l

##查看源端tag
git ls-remote --tags origin
git commit -m "提交说明"

# 3. 创建新的 tag
## 打版本
git tag v3.4

# 4. 推送代码及版本到远程仓库
## 推送代码
git push
## 推送版本标签
git push origin v3.4
```

如需要删除tag

```
#删除本地错误 tag
git tag -d v1.2.0
#删除远程错误 tag
git push origin :refs/tags/v1.2.0
```

覆盖本地

```
git -C D:\Workspace\git\evguard checkout -- fsd0.py fsd10.py guard_common.py
```



## UI设计

按当前界面，先定义一版固定“区域编号”供后续沟通：

1. A 整体窗口（EvGuard 主窗体）
2. B 顶部横幅区（黑底 EVE 图）
3. C 顶部信息栏（“EvGuard 控制台”+ RUN/STOP/SYNC + 右侧按钮）
4. C1 标题文字（EvGuard 控制台）
5. C2 统计芯片区（RUN / STOP / SYNC）
6. C3 REFRESH 按钮
7. C4 EMERGENCY EXIT 按钮
8. D 主内容区（中间左右分栏总区域）
9. D1 左侧功能卡列（四个功能卡整体）

1. D1-1 FSD0 卡
2. D1-2 FSD10 卡
3. D1-3 GUARDA 卡
4. D1-4 GUARDB 卡
5. D1-5 GUARDC 卡
6. D2 右侧日志面板（整体）
7. D2-1 日志工具条（“日志/过滤/清空/自动滚动”）
8. D2-2 日志文本显示区（深蓝大框）
9. D2-3 日志滚动条
10. E 底部状态条（“就绪 | 系统稳定”）

后续你可以直接这样下指令：

- 改 C2：隐藏 STOP 芯片
- 改 D1-2：按钮改窄一点
- 改 D2-1：把“清空”放到最右侧

### UI控制台输出

self.manager.log(...)：会进 UI 控制台
子进程 print(...)：会进 UI 控制台
子进程 log_message(...)：会进 UI 控制台
_set_status_message(...)：不会进 UI 控制台，只进底部状态栏
messagebox.show...(...)：不会进 UI 控制台，只弹窗
About 窗口内容：不会进 UI 控制台
