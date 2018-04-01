import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from vgg_face_model.vggface import VGGFace
from vgg_face_model.models import VGG16, RESNET50, SENET50
from keras.engine import Model
import numpy as np
import matplotlib.pyplot as plt
import os

cur_path = os.path.dirname(__file__)
parent_path = os.path.dirname(cur_path)
base_path = parent_path + '/trained_models/emotion_models/'
# ------------------------------
# ------------------------------
# variables
num_classes = 8  # angry, disgust, fear, happy, sad, surprise, neutral,contempt
emotion_labels = ['neutral', 'anger', 'contempt', 'disgust',
              'fear', 'happy', 'sadness', 'surprise']
batch_size = 16
epochs = 500
# ------------------------------
#------------------------------------------------
# read the CK+ facial expression dataset
train_data_dir ='../data/emotion_image/train'
validation_data_dir = '../data/emotion_image/test'
save_to_dir='../data/emotion_image/train_aug'
try:
    os.makedirs(save_to_dir)
except OSError:
    pass
img_width, img_height = 224, 224
train_datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        brightness_range=(0.05,0.25),
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # zoom_range=0.1,
        horizontal_flip=True
        )

train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        # save_to_dir='../data/emotion_image/train_aug',
        # save_prefix='aug',
        # classes= emotion_labels,
        # save_format='png'
        )

validation_generator = test_datagen.flow_from_directory(
        directory= validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        # classes=emotion_labels,
        class_mode='categorical')

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(train_generator.n, 'trian samples')
print(validation_generator.n, 'validation samples')
#------------------------------------------------
# ------------------------------
# construct CNN structure
# input_shape = x_train[0].shape
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (7, 7), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))

# 2nd convolution layer
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )
model_names = './weights.h5'
model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,
                                       save_best_only=True)
callbacks = [model_checkpoint]
model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n/batch_size,
                        verbose=1,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=callbacks)  # train for randomly selected one

#-------------------------------------------
# construct the pretrained model vgg16 resnet50
#
# keras.backend.set_image_dim_ordering('tf')
# def setup_to_transfer_learn(model, base_model):
#   """Freeze all layers and compile the model"""
#   for layer in base_model.layers:
#     layer.trainable = False
#   # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# def add_new_last_layer(base_model, nb_classes):
#   """Add last layer to the convnet
#   Args:
#     base_model: keras model excluding top
#     nb_classes: # of classes
#   Returns:
#     new keras model with last layer
#   """
#   # Classification block
#   last_layer = base_model.get_layer('avg_pool').output
#   x = Flatten(name='flatten')(last_layer)
#   predictions = Dense(nb_classes, activation='softmax', name='classifier')(x)
#   model = Model(input=base_model.input, output=predictions)
#   return model
#
# def setup_to_finetune(model):
#   """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
#   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
#   Args:
#     model: keras model
#   """
#   n = 10000
#   for i in range(len(model.layers)):
#       if model.layers[i].name == 'activation_37':
#           n = i
#       if n < i:
#           model.layers[i].trainable = True
#       else:
#           model.layers[i].trainable = False
#   # for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
#   #    layer.trainable = False
#   # for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
#   #    layer.trainable = True
#
# def pretrain_model(model_type = 'vgg16'):
#     if model_type == 'vgg16':
#         hidden_dim = 512
#         vgg_model= VGG16(include_top=False, weights='vggface',
#                          input_shape=(224, 224, 3),pooling='avg')
#         last_layer = vgg_model.get_layer('pool5').output
#         x = Flatten(name='flatten')(last_layer)
#         x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#         x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#         out = Dense(num_classes, activation='softmax', name='fc8')(x)
#         model = Model(vgg_model.input, out)
#         return model
#     elif model_type == 'resnet50':
#         base_model = RESNET50(include_top=False, weights='vggface',
#                              input_shape=(224, 224, 3),pooling='avg')
#
#         model = add_new_last_layer(base_model, num_classes)
#         # setup_to_transfer_learn(model, base_model)
#         setup_to_finetune(model)
#         model.summary()
#         return model
#     else:
#         print('model input errors')
#
# model = pretrain_model(model_type='resnet50')
# # ------------------------------
# # model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9)
# #               , loss='categorical_crossentropy'
# #               , metrics=['accuracy']
# #               )
#
# model.compile(loss='categorical_crossentropy'
#               , optimizer=keras.optimizers.Adam()
#               , metrics=['accuracy']
#               )
#
# # ------------------------------
#
# fit = True
#
# if fit == True:
#     # callbacks
#     model_type = ['vgg16','resnet50']
#     patience = 25
#     log_file_path = base_path + 'ck' + '_emotion_training.log'
#     csv_logger = CSVLogger(log_file_path, append=False)
#     early_stop = EarlyStopping('val_loss', patience=patience)
#     # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
#     #                               patience=int(patience / 5), verbose=1)
#     trained_models_path = base_path + 'ck' +'_'+ model_type[1]
#     model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
#     model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,
#                                        save_best_only=True)
#     # keras.callbacks.LearningRateScheduler(reduce_lr)
#     callbacks = [model_checkpoint, csv_logger, early_stop
#                  # ,reduce_lr
#                  ]
#     model.fit_generator(train_generator,
#                         steps_per_epoch=train_generator.n/batch_size,
#                         verbose=1,
#                         validation_data=validation_generator,
#                         epochs=epochs,
#                         callbacks=callbacks)  # train for randomly selected one
# else:
#     model.load_weights('/data/facial_expression_model_weights.h5')  # load weights

# ------------------------------
# plot the training history parameters
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()



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

