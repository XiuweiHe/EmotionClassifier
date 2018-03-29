import os
import cv2
from PIL import Image
def im2video(img_dir=None,video_dir=None):
    """convert the serial images to video
    :param img_dir: a directory of images
    :type: str
    :param video_dir: a directory of video to store
    """
    img_dir = '../S052'
    img_root = '../test_image/'
    fps = 15
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(img_root+'emotion.mp4',fourcc, fps, (640,490))
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
def im2gif():
    img_dir = '../S052'
    img_root = '../test_image/'
    frame = []
    for file in os.listdir(img_dir):
        dir_path = os.path.join(img_dir, file)
        for img in os.listdir(dir_path):
            img_file = dir_path + '/' + img
            img = Image.open(img_file)
            frame.append(img)
            # write the flipped frame
    img.save(img_root+'emotion.gif',save_all=True, append_images=frame,loop=1,duration=1,comment=b"aaabb")

if __name__=="__main__":
    im2video()
    # im2gif()