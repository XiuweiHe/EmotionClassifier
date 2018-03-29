import numpy as np
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
import mxnet as mx
from mxnet import image
from collections import namedtuple
from src import models
import cv2
import matplotlib.pyplot as plt
"""
constant variable
"""
image_dir = 'E:/faceLib/extended-cohn-kanade-images/cohn-kanade-images'
label_dir = 'E:/faceLib/Emotion_labels/Emotion'
output_dir = './image'
emotion_label={0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust',
             4: 'fear', 5:'happy', 6:'sadness', 7:'surprise'}

"""
set compute device
"""
def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
"""
vgg minimam unit
"""
def vgg_block(num_convs, channels):
    out = nn.HybridSequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm(),
            nn.Activation('relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
def vgg_stack(architecture):
    out = nn.HybridSequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

def vgg_11(num_outputs,architecture):
    # num_outputs = 10
    # architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
    net = nn.HybridSequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        net.add(
            vgg_stack(architecture),
            nn.Flatten(),
            # nn.Dense(1024, activation="relu"),
            # nn.BatchNorm(),
            # nn.Dropout(.2),
            nn.Dense(1024, activation="relu"),
            nn.BatchNorm(),
            nn.Dropout(.5),
            nn.Dense(num_outputs))
    return net

"""
fine-tune the vgg16_bn
"""
def Classifier():
    net = nn.HybridSequential()
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(8))
    return net
def vgg11_bn():
    vgg11_bn = mx.gluon.model_zoo.vision.vgg11_bn()
    vgg11_bn.load_params('.\model\\vgg\\vgg11_bn-ee79a809.params', ctx=mx.gpu(0))
    featuresnet = vgg11_bn.features
    for _, w in featuresnet.collect_params().items():
        w.grad_req = 'null'
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(featuresnet)
        net.add(Classifier())
        net[1].collect_params().initialize(init=mx.init.Xavier(), ctx=mx.gpu(0))
    net.hybridize()
    return net
"""
finetue the incepton_v3 net
"""
def inception_v3(num_outputs):
    net = vision.Inception3(classes=num_outputs)
    # net = nn.HybridSequential()
    # net.add(features)
    # net.add(nn.Dense(num_outputs))

    net.hybridize()
    return net

from mxnet import gluon
from mxnet import init
from utils.builddata import DataManager
from utils.builddata import *
from time import time
from utils import get_data
from utils import utils

# def getData(dataset_name = 'CK+'):
#     if dataset_name == 'CK+':
#         face_data_set = DataManager(dataset_name=dataset_name,image_size=(96,96),b_gray_chanel=True).get_data()
#
#         training_data = [face_data_set[0], face_data_set[1]]
#         testing_data = [face_data_set[2], face_data_set[3]]
#         print(training_data[0].shape)
#         return training_data, testing_data
#     elif dataset_name == 'fer2013':
#         x_train, y_train, x_val, y_val, x_test, y_test = DataManager(dataset_name=dataset_name,image_size=(96,96)).get_data()
#         # x_test = np.append(x_test,x_val, axis=0)
#         # y_test = np.append(y_test,y_val, axis=0)
#         # x = np.append(x_train,x_test, axis=0)
#         # y = np.append(y_train,y_test, axis=0)
#         # index = np.random.permutation(np.arange(len(x)))
#         # train_len = int((1-0.2)* len(x))
#         # x, y = x[index], y[index]
#         # x_train, y_train, x_test, y_test = x[:train_len],y[:train_len], x[train_len:], y[train_len:]
#
#         y_train = y_train.argmax(axis = 1)
#         y_test = y_test.argmax(axis = 1)
#         # x_train = gluon.data.DataLoader([x_train,y_train],128,shuffle=True)
#         # x_train_aug = []
#         # nd.array(x_train)
#         # for i in range(len(x_train)):
#         #     data_aug = transform(x_train[i])
#         #     x_train_aug.append(data_aug)
#         # nd.array(x_train_aug)
#         # nd.stack([x_train,x_train_aug],axis=0)
#         # nd.stack([y_train,y_train],axis=0)
#         return [x_train, y_train], [x_test, y_test]
#     else:
#         print('input dataset does not exist!')

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        data = self.dataset[:]
        X = nd.array(data[0])
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            if self.transform is not None:
                yield self.transform(X[i*self.batch_size:(i+1)*self.batch_size],
                                     y[i*self.batch_size:(i+1)*self.batch_size])
            else:
                yield (X[i*self.batch_size:(i+1)*self.batch_size],
                       y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size

def transform_data(train_data, test_data, batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # Transform a batch of examples.
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x width x channel to batch x channel x height x width
        return nd.transpose(data.astype('float32'), (0,3,2,1))/255, label.astype('float32')
    # Transform later to avoid memory explosion.
    train_data = DataLoader(train_data, batch_size, shuffle=True, transform=transform_mnist)
    test_data = DataLoader(test_data, batch_size, shuffle=False, transform=transform_mnist)
    return (train_data, test_data)



if __name__ == "__main__":
    while(True):
        # command = input()
        command = 'train'
        if command == 'train':
            # data_name = input()
            data_name = 'ck'
            if data_name == 'ck':
                # train_data, test_data = getData(dataset_name='CK+')
                train_data, test_data = get_data.get_data_ck(image_size=(224,224),b_gray_chanel=False)
                batch_size = 8
                n_class = 8
            elif data_name == 'fer2013':
                # train_data, test_data = getData(dataset_name='fer2013')
                batch_size = 32
                n_class = 7
            else:
                print('input data set does not exist!')
                continue
            # print('traing samples number: %d'%(len(train_data[0])),'\n',
            #       'testing samples number: %d'%(len(test_data[0])))
            train_data, test_data = transform_data(train_data, test_data, batch_size, resize=None)
            ctx = try_gpu()
            net = vgg11_bn()
            # net = vision.vgg11(pretrained= False, ctx = ctx)
            # net = vision.vgg11_bn(pretrained=False, ctx=ctx)
            # net = models.GoogLeNet(num_classes=n_class)

            # arc = ((1,64), (1,128))
            # net = vgg_11(num_outputs=n_class,architecture=arc)
            # net.initialize(ctx=ctx, init=init.Xavier())
            print(net)

            # net.hybridize()
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
            trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.001, 'wd': 0.001})

            num_epoch = 50
            loss, train_acc, test_acc = utils.train(train_data, test_data, net, loss, trainer, ctx, num_epoch,print_batches=None)
            print('training process completed', '\n', 'save the model parameters')

            # plot the change of the parameter in the training process
            plt.figure(1)
            plt.xlabel('number of epochs')
            plt.ylabel('loss train_acc test_acc')
            x = [i for i in range(num_epoch)]
            plt.plot(x,loss,'r-s',label="loss")
            plt.plot(x,train_acc,'g-o',label= "train_acc")
            plt.plot(x,test_acc,'b-+',label= "test_acc")
            plt.legend(loc= 'upper left', fontsize=16)
            plt.savefig('./params.png',format= 'png',dpi=300)
            plt.show()
            if data_name == 'ck':
                model_file_name = './model/vgg/ck-vgg11'
                try:
                    os.makedirs(model_file_name)
                except OSError:
                    if not os.path.isdir(model_file_name):
                        raise
                net.export(model_file_name,50)
            elif data_name =='fer2013':
                model_file_name = './model/vgg/fer2013-vgg11'
                try:
                    os.makedirs(model_file_name)
                except OSError:
                    if not os.path.isdir(model_file_name):
                        raise
                net.export(model_file_name, 50)
            print('traing process is complete')
            break
        elif command == 'test':
            # n_class = 8
            # arc = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
            # net = vgg_11(num_outputs=n_class, architecture=arc)
            symnet = mx.symbol.load('./model/vgg/vgg11-symbol.json')
            # print(symnet)
            mod = mx.mod.Module(symbol=symnet, context=mx.cpu())
            mod.bind(data_shapes=[('data', (1, 1, 96, 96))],for_training=False)
            mod.load_params('./model/vgg/vgg11-0050.params')
            batch = namedtuple('batch', ['data'])
            # net.load_params('./model/vgg/vgg11.params',ctx= try_gpu())
            img_dir = 'E:\GithubProject\EmotionClassifier\\test_image\S056_003_00000010.png'
            img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
            face = get_face(img)
            face = nd.array(np.reshape(face,(-1, 1, 96, 96)))
            mod.forward(data_batch=batch([face]),is_train=False)
            out = mod.get_outputs()
            prob = out[0]
            predicted_labels = prob.argmax(axis=1)
            # emotion = net(face)
            print(predicted_labels)

        elif command == 'q':
            break
        else:
            print("please input the command 'train' or 'test' or 'q'to exist")
