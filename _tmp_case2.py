from cnocr import CnOcr
from PIL import Image
img = Image.open('ocr.png').convert('RGB')
ocr = CnOcr(rec_root='./', det_root='./')
out = ocr.ocr(img)
texts = [x.get('text') for x in out]
print('CASE=rec_root_PIL')
print('PIN999', any('PIN999' in t for t in texts))
print('促进', any('促进' in t for t in texts))
print('|'.join(texts[:30]))
