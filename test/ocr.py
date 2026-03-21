from cnocr import CnOcr

img_fp = 'ocr.png'
ocr = CnOcr(rec_root="./",det_root="./")  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

print(out)