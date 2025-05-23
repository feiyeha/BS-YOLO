from cnocr import CnOcr
ocr = CnOcr()
img_path = '29.jpg'  # 图片路径
result = ocr.ocr(img_path)
print(result)