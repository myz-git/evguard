from cnocr import CnOcr
img_fp = 'ocr.png'
ocr = CnOcr(rec_root='./', det_root='./')
out = ocr.ocr(img_fp)
texts = [x.get('text') for x in out]
print('CASE=ocr_py_style')
print('PIN999', any('PIN999' in t for t in texts))
print('促进', any('促进' in t for t in texts))
print('|'.join(texts[:30]))
