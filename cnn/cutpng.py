import os
from PIL import Image

# 定义输入和输出文件夹路径
input_folder = "input_pngs"  # 包含20x20 PNG图片的文件夹
output_folder = "output_pngs"  # 保存18x18裁剪后图片的文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):  # 只处理PNG文件
        # 打开图片
        input_path = os.path.join(input_folder, filename)
        img = Image.open(input_path)

        # 裁剪图片：从(2,2)到(18,18)，即上下左右各去掉2像素
        cropped_img = img.crop((2, 2, 18, 18))

        # 保存裁剪后的图片到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path, "PNG")

        print(f"已处理: {filename}")

print("所有图片裁剪完成！")