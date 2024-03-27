import cv2

img = cv2.imread('axon01.tif', cv2.IMREAD_UNCHANGED)

img_8 = img.astype('uint8')

bbox = cv2.selectROI('test', img_8)
print(bbox)

img_out = img_8[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
cv2.imwrite('axon01_cut.tif', img_out)

img_16_out = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
cv2.imwrite('axon01_cut_16bit.tif', img_out)
cv2.waitKey(0)
