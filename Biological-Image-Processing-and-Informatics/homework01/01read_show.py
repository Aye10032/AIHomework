import cv2
from matplotlib import pyplot as plt

img = cv2.imread('axon01.tif', cv2.IMREAD_UNCHANGED)

min_gray = img.min()
max_gray = img.max()
print(min_gray, max_gray)
plt.hist(img.reshape(-1), bins=150, color='skyblue', edgecolor='black', range=(min_gray, max_gray))

plt.xlabel('Value')
plt.ylabel('Frequency')

plt.show()

img_out = img.astype('uint8')
cv2.imshow('test', img_out)
cv2.waitKey(0)
