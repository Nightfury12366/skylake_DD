import cv2


img_path = '/home/skylake/skylake_files/pic_sky_project/imgs/Image_batch_01_00053.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('test', img)
cv2.waitKey(0)

