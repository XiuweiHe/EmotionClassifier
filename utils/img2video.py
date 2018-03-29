import os
import cv2

img_dir = '../S052'
img_root = '../test_image/'
fps = 15
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(img_root+'emotion.avi',fourcc, fps, (640,490))
for file in os.listdir(img_dir):
    dir_path = os.path.join(img_dir,file)
    for img in os.listdir(dir_path):
        img_file = dir_path + '/'+ img
        frame = cv2.imread(img_file)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()