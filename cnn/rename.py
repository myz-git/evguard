import os
import glob
import argparse
import logging

# 日志配置，与train.py, task.py等一致
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evguard.log', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)

def rename_files(directory):
    """重命名指定目录下的PNG文件为<目录名>-<index>.png"""
    # 构建目录路径
    traindata_dir = '../traindata'
    target_dir = os.path.join(traindata_dir, directory)
    
    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        logging.error(f"Directory {target_dir} not found")
        raise FileNotFoundError(f"Directory {target_dir} not found")

    # 获取所有PNG文件
    png_files = glob.glob(os.path.join(target_dir, '*.png'))
    if not png_files:
        logging.warning(f"No PNG files found in {target_dir}")
        raise FileNotFoundError(f"No PNG files found in {target_dir}")

    logging.info(f"Found {len(png_files)} PNG files in {target_dir}")

    # 重命名文件
    for index, old_path in enumerate(png_files, start=1):
        # 构建新文件名
        new_filename = f"{directory}-{index}.png"
        new_path = os.path.join(target_dir, new_filename)
        
        # 检查新文件名是否已存在
        if os.path.exists(new_path):
            logging.warning(f"File {new_path} already exists, skipping")
            continue
        
        # 重命名
        try:
            os.rename(old_path, new_path)
            logging.info(f"Renamed {old_path} to {new_filename}")
        except Exception as e:
            logging.error(f"Failed to rename {old_path} to {new_filename}: {e}")

    logging.info(f"Renaming completed for {target_dir}, processed {len(png_files)} files")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Rename PNG files in traindata\<directory> to <directory>-<index>.png")
    parser.add_argument('directory', type=str, help="Directory name (e.g., jiku2-1)")
    args = parser.parse_args()

    try:
        rename_files(args.directory)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
