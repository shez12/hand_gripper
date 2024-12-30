import cv2
import numpy as np

# 创建一个简单的测试图像
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
test_image[200,300,:]=255
# 显示测试图像
cv2.imshow('Test Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()