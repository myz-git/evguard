



## 1. 初始化仓库

### 1.1 准备：GitHub 上建一个空仓库

在 GitHub 创建仓库 `evguard`，**不要勾选** README/.gitignore/license（保持空仓库）。
 拿到地址：`https://github.com/myz-git/evguard.git`

------

### 1.2 进入项目目录

```
cd /d D:\Workspace\git\evguard
```

------

### 1.3 彻底重来：删掉旧 .git (可选但推荐）

如果你确定要重新来（你之前就是这个诉求）：

```
rmdir /s /q .git
```

------

### 1.4 初始化仓库 + 固化 CRLF 规则 + 忽略构建产物

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

### 1.5 第一次提交

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

### 1.6 绑定 GitHub 远端并 push

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

## 2. 建立虚拟环境

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



### 3. 打包: 

### 本地打包

```
#清理
cd /d D:\Workspace\git\evguard
rmdir /s /q build
rmdir /s /q dist
del /q *.spec.bak 2>nul

#打包
pyinstaller --clean  -y ev_all.spec
```



### 更新git 版本

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
git tag v3.3

# 4. 推送代码及版本到远程仓库
## 推送代码
git push
## 推送版本标签
git push origin v3.3
```

如需要删除tag

```
#删除本地错误 tag
git tag -d v1.2.0
#删除远程错误 tag
git push origin :refs/tags/v1.2.0
```

