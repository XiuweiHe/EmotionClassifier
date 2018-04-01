import os
import cv2
import dlib
import numpy as np
import shutil
import random
import tensorflow as tf
import keras
from time import time
from collections import Counter
import scipy.io as io

cur_path = os.path.dirname(__file__)
parent_path = os.path.dirname(cur_path)


IMAGE_DIR_CK = 'E:/faceLib/extended-cohn-kanade-images/cohn-kanade-images'
LABEL_DIR_CK = 'E:/faceLib/Emotion_labels/Emotion'
OUT_PUT_DIR_CK = parent_path + '\data\emotion_image'
FACE_DATA_DIR_FER2013 = parent_path + "/data/fer2013/fer2013.csv"
OUT_PUT_DIR_FER2013 = parent_path + '/data/fer2013_preprocess'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('random_flip_up_down', False, 'If uses flip')
flags.DEFINE_boolean('random_flip_left_right', True, 'If uses flip')
flags.DEFINE_boolean('random_brightness', True, 'If uses brightness')
flags.DEFINE_boolean('random_contrast', False, 'If uses contrast')
flags.DEFINE_boolean('random_saturation', False, 'If uses saturation')
flags.DEFINE_integer('image_size', 224, 'image size.')
flags.DEFINE_boolean('resize', False, 'If uses image resize')
"""
#flags examples
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
"""
def pre_process(images):

    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.15)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    if FLAGS.random_saturation:
        images = tf.image.random_saturation(images, 0.3, 0.5)
    # if FLAGS.resize:
    #     new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    #     images = tf.image.resize_images(images, new_size)
    return images

class DataManager(object):
    """
    load the dataset CK+ and fer2013
    """
    def __init__(self, dataset_name='CK+', dataset_path=None, num_classes = 8,image_size=(224, 224),b_gray_chanel = True):
        """

        :param dataset_name: select the dataset "CK" or "fer2013"
        :param dataset_path: the dataset location dir
        :param num_classes: the classes number of dataset
        :param image_size: the image size output you want
        :param b_gray_chanel: if or not convert image to gray

        :return the tuple have image datas and image labels
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.b_gray_chanel = b_gray_chanel
        self.num_classes = num_classes
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'CK+':
            self.dataset_path = IMAGE_DIR_CK
        elif self.dataset_name == 'fer2013':
            self.dataset_path = FACE_DATA_DIR_FER2013
        else:
            raise Exception('Incorrect dataset name, please input CK+ or fer2013')

    def get_data(self):
        if self.dataset_name == 'CK+':
            data = self._load_ck()
        elif self.dataset_name == 'fer2013':
            data = self._load_fer2013()
        return data
    def _load_ck(self):
        # get the training data from the source dataset
        # self.build_dataset(self.dataset_path,LABEL_DIR_CK,OUT_PUT_DIR_CK,'png')

        # preprocess the select data, you can change augment it or not
        self.get_data_ck(OUT_PUT_DIR_CK,augmentation=False,test_percent=0.3)

        # get the training data and testing data separately
        # data_ck = self.get_data_ck_split(data_dir=OUT_PUT_DIR_CK)
        # print(np.shape(data_ck))
        # return data_ck

    def build_dataset(self, image_dir, label_dir, output_dir,itype='png'):
        """ Builds a dataset using data augmentation and normalization built for the CK+ Emotion Set.
        :param image_dir: A directory of input images.
        :type image_dir: str
        :param label_dir: A directory of input labels.
        :type label_dir: str
        :param output_dir: A directory for new images to be sorted.
        :type output_dir: str
        :param itype: File type for output images.
        :type itype: str

        :return: The number of images.
        :rtype: int
        """
        print('start process')
        start = time()
        image_files = []
        for outer_folder in os.listdir(image_dir):
            if os.path.isdir(image_dir + '/' + outer_folder):
                for inner_folder in os.listdir(image_dir + '/' + outer_folder):
                    if os.path.isdir(image_dir + '/' + outer_folder + '/' + inner_folder):
                        for input_file in os.listdir(image_dir + '/' + outer_folder + '/' + inner_folder):
                            if input_file.split('.')[1] != itype:
                                break
                            label_file = label_dir+'/'+outer_folder+'/'+inner_folder+'/'+input_file[:-4] + '_emotion.txt'
                            if os.path.isfile(label_file):
                                read_file = open(label_file, 'r')
                                label = int(float(read_file.readline().split('.')[0]))
                                for i in range(-1, -4, -2):
                                    image_file = sorted(os.listdir(image_dir + '/' + outer_folder + '/' + inner_folder))[i]
                                    if image_file.split('.')[1] == itype:
                                        image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+image_file, label))
                                neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[0]
                                if neutral_file.split('.')[1] != itype:
                                    neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[1]
                                image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+neutral_file, 0))
        print (len(image_files))
        print ('-------------Files   Collected---------------------')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            for i in range(self.num_classes):
                os.makedirs(output_dir+'/'+str(i))

        detector, count = dlib.get_frontal_face_detector(), 0
        for image_file in image_files:
            image = cv2.imread(image_file[0])
            detections = detector(image, 1)
            for _, detection in enumerate(detections):
                left, right, top, bottom = detection.left() - 10, detection.right() + 10,\
                                           detection.top() - 10, detection.bottom() + 10

                # show the images
                # img = cv2.rectangle(image, (left, bottom), (right,top), (255, 128, 0), 2)
                # img = cv2.rectangle(img,(detection.left(),detection.bottom()),(detection.right(),detection.top()),(255,0,0),2)
                # cv2.imshow("facedetect",img)
                # if cv2.waitKey() == 'q':
                #     break
                # cv2.destroyAllWindows()

                face = image[top:bottom, left:right]
                face = cv2.resize(face, (299, 299))
                name = output_dir + '/' + str(image_file[1]) + '/' + image_file[0][-21:-4] + str(count) + '.' + itype
                cv2.imwrite(name, face)
                count += 1
            if count % 100 == 0:
                print ('Current count = ' + str(count))
        print('end process: %f'%(time()-start))
        return count

    def get_data_ck(self,input_dir, augmentation= True, test_percent = 0.3):
        """ Gets the data from a directory of images organised by classification.
        :param input_dir: The directory of images.
        :type input_dir: str
        :param augmentation: if or not augment the data
        :type augmentation: bool
        :param test_percent: the test data percent of total dataset
        :type test_percent: float

        """
        data, label, num = [], [] ,0
        for folder in sorted(os.listdir(input_dir)):
            if folder =='train' or folder == 'test':
                continue
            for image_file in os.listdir(input_dir +'/'+ folder):
                image = cv2.imread(input_dir +'/'+ folder + '/' + image_file)
                data.append((image))
                label.append(num)
            num += 1
        # h, w, c = data[0].shape
        # data = np.reshape(data, (-1, h, w, c))
        print('source data statistic: ', Counter(label))

        from sklearn.model_selection import train_test_split
        train_data, test_data, train_label, test_label =  train_test_split(data, label,test_size=test_percent,shuffle=True)

        train_dir, test_dir = input_dir +'/train', input_dir + '/test'
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            os.makedirs(train_dir)
            os.makedirs(test_dir)
            for i in range(self.num_classes):
                os.makedirs(train_dir + '/' + str(i))
                os.makedirs(test_dir + '/' + str(i))
        count = 0
        for n in range(len(train_data)):
            image = train_data[n]
            name = train_dir + '/' + str(train_label[n]) + '/' + '_' + str(count) + '.' + 'png'
            cv2.imwrite(name, image)
            count += 1
            if count % 20 == 0:
                print('%f numbers samples processed'%(count))
        count = 0
        for n in range(len(test_data)):
            image = test_data[n]
            name = test_dir + '/' + str(test_label[n]) + '/' +  '_' + str(count) + '.' + 'png'
            cv2.imwrite(name, image)
            count += 1
            if count % 20 == 0:
                print('%f numbers samples processed' % (count))

    def get_data_ck_split(self,data_dir = './data'):
        """Get the training and testing data from the splited data set
        :param data_dir: the directory of image data set
        :type data_dir: str
        :return: the list contained training and testing data
        :type: list
        """
        train_data, train_label, test_data, test_label = [], [], [], []

        for folder in os.listdir(data_dir):
            if folder == 'train':
                num = 0
                for sub_folder in sorted(os.listdir(data_dir + '/'+ folder)):
                    for image_file in os.listdir(data_dir + '/' + folder + '/'+sub_folder):

                        if self.b_gray_chanel:
                            image = cv2.imread(data_dir + '/' + folder + '/'+sub_folder + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                        else:
                            image = cv2.imread(data_dir + '/' + folder + '/' + sub_folder + '/' + image_file)
                        try:
                            image = cv2.resize(image, self.image_size)
                        except:
                            print(num)
                            raise
                        train_data.append(image)
                        train_label.append(num)
                    num += 1
            elif folder == 'test':
                num = 0
                for sub_folder in sorted(os.listdir(data_dir + '/'+ folder)):
                    for image_file in os.listdir(data_dir + '/' + folder + '/'+sub_folder):
                        if self.b_gray_chanel:
                            image = cv2.imread(data_dir + '/' + folder + '/'+sub_folder + '/' + image_file,cv2.IMREAD_GRAYSCALE)
                        else:
                            image = cv2.imread(data_dir + '/' + folder + '/' + sub_folder + '/' + image_file)
                        try:
                            image = cv2.resize(image, self.image_size)
                        except:
                            print(num)
                            raise
                        test_data.append(image)
                        test_label.append(num)
                    num += 1
        height, weight = self.image_size[0], self.image_size[1]
        if self.b_gray_chanel:
            n_chanel = 1
        else:
            n_chanel = 3
        train_data = np.reshape(train_data,(-1,height,weight,n_chanel))
        test_data = np.reshape(test_data,(-1,height, weight,n_chanel))

        print('number of train samples:',train_data.shape[0])
        print('number of test samples:',test_data.shape[0])
        # parameters = {'train':Counter(train_label),'test':Counter(test_label)}
        # io.savemat('param.txt',parameters)
        # print the number of samples in each category
        print('train data statistic: ',Counter(train_label))
        print('test data statistic: ', Counter(test_label))
        return [train_data, train_label, test_data, test_label]


    def _load_fer2013(self):
        """ load the dataset of fer2013 for the file fer2013.csv
        :return: a list contains the training ,private test and public test set
        :type: list
        """
        # fer2013 dataset:
        # Training       28709
        # PrivateTest     3589
        # PublicTest      3589

        # emotion labels from FER2013:
        # emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
        #          'Sad': 4, 'Surprise': 5, 'Neutral': 6}
        num_classes = 7
        with open(self.dataset_path) as f:
            content = f.readlines()

        lines = np.array(content)
        num_of_instances = lines.size
        print("number of instances: ", num_of_instances)
        print("instance length: ", len(lines[1].split(",")[1].split(" ")))

        # ------------------------------
        # initialize train set, val set and test set
        x_train, y_train, x_test, y_test = [], [], [], []
        x_val, y_val = [], []
        # ------------------------------
        # transfer train, val and test set data
        for i in range(1, num_of_instances):

            emotion, img, usage = lines[i].split(",")

            val = img.split(" ")

            pixels = np.array(val, 'float32')

            emotion = keras.utils.to_categorical(emotion, num_classes)
            face = pixels.reshape((48, 48))
            face = cv2.resize(face.astype('uint8'),self.image_size)
            face.astype('float32')
            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(face)
            elif 'PublicTest' in usage:
                y_val.append(emotion)
                x_val.append(face)
            elif 'PrivateTest' in usage:
                y_test.append(emotion)
                x_test.append(face)

        # ------------------------------
        # data transformation for train ,val, and test sets
        x_train = np.expand_dims(np.asarray(x_train),-1)
        y_train = np.array(y_train, 'float32')
        x_val = np.expand_dims(np.asarray(x_val), -1)
        y_val = np.array(y_val, 'float32')
        # x_test = np.expand_dims(np.asarray(x_test), -1)
        # y_test = np.array(y_test, 'float32')

      
        # , x_test, y_test
        return x_train, y_train, x_val, y_val

def split_data_ck(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def preprocess_input(x,v2 = True):
    """normalize the data to [0,1] and select transform it to [-0.5,0.5] or not"""
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

if __name__ == "__main__":

    DataManager(dataset_name='CK+',image_size=(224,224)).get_data()

    # data = DataManager(dataset_name='fer2013',image_size=(64,64)).get_data()
    # build_dataset(IMAGE_DIR_CK,LABEL_DIR_CK,OUT_PUT_DIR_CK,'png',augmentation=False)
