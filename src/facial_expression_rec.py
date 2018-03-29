import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from utils.builddata import DataManager
# ------------------------------
# cpu - gpu configuration
config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 56})  # max: 1 gpu, 56 cpu
sess = tf.Session(config=config)
keras.backend.set_session(sess)
# ------------------------------
# variables
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 5
# ------------------------------
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
x_train,y_train, x_val, y_val, x_test, y_test = DataManager(dataset_name='fer2013').get_data()

x_train /= 255  # normalize inputs between [0, 1]
x_val /= 255

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')
# ------------------------------
# construct CNN structure
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
# batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# ------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )

# ------------------------------

fit = True

if fit == True:
    # model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)  # train for randomly selected one
else:
    model.load_weights('/data/facial_expression_model_weights.h5')  # load weights

# ------------------------------
""""""
#overall evaluation
score = model.evaluate(x_val, y_val)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])



# ------------------------------
# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()


# ------------------------------

monitor_testset_results = False

if monitor_testset_results == True:
    # make predictions for test set
    predictions = model.predict(x_val)

    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            # print(i) #predicted scores
            # print(y_val[index]) #actual scores

            testing_img = np.array(x_val[index], 'float32')
            testing_img = testing_img.reshape([48, 48])

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1

# ------------------------------
# make prediction for custom image out of test set

# img = image.load_img("C:/Users/IS96273/Desktop/jackman.png", grayscale=True, target_size=(48, 48))
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
#
# x /= 255
#
# custom = model.predict(x)
# emotion_analysis(custom[0])
#
# x = np.array(x, 'float32')
# x = x.reshape([48, 48])
#
# plt.gray()
# plt.imshow(x)
# plt.show()
# ------------------------------

