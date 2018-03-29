import os
import cv2
import sys
import dlib
import numpy as np
import shutil
import random
import tensorflow as tf
import keras
from time import time
from collections import Counter
from mxnet import nd
import scipy.io as io
sys.path.append('../face-frontalization')
cur_path = os.path.dirname(__file__)
parent_path = os.path.dirname(cur_path)


IMAGE_DIR_CK = 'E:/faceLib/extended-cohn-kanade-images/cohn-kanade-images'
LABEL_DIR_CK = 'E:/faceLib/Emotion_labels/Emotion'
OUT_PUT_DIR_CK = parent_path + '\data\emotion_image'
FACE_DATA_DIR_FER2013 = parent_path + "/data/fer2013/fer2013.csv"
OUT_PUT_DIR_FER2013 = parent_path + '/data/fer2013_preprocess'

def random_mask(ndimg, size, n_chanel= 3,flag=0):
    w, h = ndimg[0].shape # 获取图像的尺寸
    w_ = random.randint(0, w-size) #确定起始坐标的位置范围
    h_ = random.randint(0, h-size)
    if flag==0:
        # 随机遮盖的形状是一个正方形
        ndimg[w_:w_+size, h_:h_+size, :] = nd.zeros((size, size, n_chanel)) # 用黑色来遮盖
        return ndimg
    elif flag==1:
        # 随机遮盖的形状是一个长方形
        w_size = random.randint(0, size-1)
        h_size = random.randint(0, size-1)
        # 用随机噪声来遮盖
        ndimg[w_:w_+w_size, h_:h_+h_size, :] = mx.ndarray.random_uniform(low=0, high=255, shape=(w_size, h_size, n_chanel))
        return ndimg

def image_augmentaion(image_data):
    # An example of creating multiple augmenters
    augs = mx.image.CreateAugmenter(data_shape=image_data.shape, rand_mirror=True,
                                    mean=None,std=None, brightness=0, contrast=0, rand_gray=0,
                                    saturation=0, pca_noise=0.05, inter_method=10)
    # dump the details
    for aug in augs:
        image_data = aug(image_data)
    return image_data
# train_augs = [
#     # image.ResizeAug(250), # 将短边resize至250
#     image.HorizontalFlipAug(.5), # 0.5概率的水平翻转变换
#     # image.HueJitterAug(.6), # -0.6~0.6的随机色调
#     image.BrightnessJitterAug(.25), # -0.5~0.5的随机亮度
#     # image.RandomCropAug((230,230)), # 随机裁剪成（230,230）
#     image.ContrastJitterAug(.25) #调整对比度
# ]
# 获得transform函数
def transform(data):
    data = nd.array(data) # 部分数据增强接受`float32`
    data = nd.transpose(data, (2,0,1)) # 改变维度顺序为（c, w, h）
    data = image_augmentaion(data)
    data = random_mask(data, 32, n_chanel= 1,flag=1) # 执行random_mask, 随机遮盖

    return data
def get_data_ck(input_dir=OUT_PUT_DIR_CK, image_size=(224,224), b_gray_chanel=True,test_percent=0.3):
    """ Gets the data from a directory of images organised by classification.
    :param input_dir: The directory of images.
    :type input_dir: str
    :return: A list of images and labels.
    :rtype: list of tuples each containing an image and a int
    """
    data, label, num = [], [], 0
    for folder in sorted(os.listdir(input_dir)):
        if folder == 'train' or folder == 'test':
            continue
        for image_file in os.listdir(input_dir + '/' + folder):
            if b_gray_chanel:
                n_chanel = 1
                image = cv2.imread(input_dir + '/' + folder + '/' + image_file, cv2.IMREAD_GRAYSCALE)
            else:
                n_chanel = 3
                image = cv2.imread(input_dir + '/' + folder + '/' + image_file)
            image = cv2.resize(image,image_size)
            data.append((image))
            label.append(num)
        num += 1
    print('source data statistic: ', Counter(label),'\n',len(label))
    height, weight = image_size[0],image_size[1]

    from sklearn.model_selection import train_test_split
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=test_percent, shuffle=True)

    train_data = np.reshape(train_data, (-1, height, weight, n_chanel))
    test_data = np.reshape(test_data, (-1, height, weight, n_chanel))

    print('number of train samples:', len(train_data))
    print('number of test samples:', len(test_data))
    # parameters = {'train':Counter(train_label),'test':Counter(test_label)}
    # io.savemat('param.txt',parameters)
    print('train data statistic: ', Counter(train_label))
    print('test data statistic: ', Counter(test_label))
    return [train_data, train_label], [test_data, test_label]
if __name__ == "__main__":
    data = get_data_ck()
