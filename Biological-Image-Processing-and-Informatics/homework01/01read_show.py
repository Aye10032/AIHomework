import cv2

img = cv2.imread('axon01.tif', cv2.IMREAD_UNCHANGED)

img_out = img.astype('uint8')
cv2.imshow('test', img_out)
cv2.waitKey(0)
