"""
image dataset: Cohn-Kanade (CK and CK+) database
URL:http://www.consortium.ri.cmu.edu/ckagree/
"""
import sys
import os
import cv2
import time
import numpy
import select
import builddata
import emotionclassifier

averages, running = [], True

image_dir = 'E:/faceLib/extended-cohn-kanade-images/cohn-kanade-images'
label_dir = 'E:/faceLib/Emotion_labels/Emotion'
output_dir = './image'
resource_dir =  './resource'
# try:
#     os.makedirs(output_dir)
#     os.makedirs(resource_dir)
# except OSError:
#     if not os.path.isdir(output_dir) and not os.path.isdir(resource_dir):
#         raise

session_path = 'model/model.ckpt'
try:
    os.makedirs(session_path)
except OSError:
    if not os.path.isdir(session_path):
        raise
def bulid_data_set():
    """Create the face image dataset from the source dataset
    """
    start = time.clock()
    count = builddata.build_thing(image_dir, label_dir, output_dir, resource_dir)
    end = time.clock()
    print ('Augmented Dataset built ' + str(count) + ' images in ' + str(end - start) + 's')


def train():
    """ Train the facial expression classification model
    """
    start = time.clock()
    faces = builddata.get_data(output_dir, 8)
    training_data, testing_data = emotionclassifier.divide_data(faces, 0.2)
    print ('number of training examples = ' + str(len(training_data)))
    print ('number of testing examples  = ' + str(len(testing_data)) + '\n')
    classifier = emotionclassifier.EmotionClassifier(8, session_path)
    accuracy = classifier.train(training_data, testing_data, epochs=100, intervals=1)
    end = time.clock()
    print ('Testing Accuracy: ' + '{:.9f}'.format(accuracy))
    print ('Training Time: ' + '{:.2f}'.format(end - start) + 's')

def accuracy():
    """Output the accuarcay of the classification model
    """
    start = time.clock()
    faces = builddata.get_data(output_dir, 8)
    _, testing_data = emotionclassifier.divide_data(faces, 0.3)
    print ('number of testing examples  = ' + str(len(testing_data)) + '\n')

    classifier = emotionclassifier.EmotionClassifier(8, session_path)
    accuracy = classifier.accuracy(testing_data)
    end = time.clock()
    print ('Accuracy: ' + str(accuracy) + ' in ' + str(end - start) + 's')


def runvideo():
    """Recognize the facial expression of the video frame with in faces
    """
    video = cv2.VideoCapture(0)
    classifier = emotionclassifier.EmotionClassifier(8, session_path)
    num=0
    time_cost =[]
    num_frame =[]
    if video.grab():
        while running:
            start = time.clock()
            _, frame = video.read()
            cv2.imshow('facial recognition',frame)
            if cv2.waitKey(1):
                face = builddata.get_face_from_frame(frame)
                if face:
                    temp = []
                    print ('Face Found')
                    classification = classifier.classify(face)
                    end = time.clock()
                    temp =float('{:.3f}'.format((1000*(end-start))))
                    num =num+1
                    num_frame.append(num)
                    time_cost.append(temp)
                    print (' classified as '+str(classification))
                    print(num,'\n',temp)
                    if num ==120:
                        import matplotlib.pyplot as plt
                        import scipy as sc
                        sc.savetxt("time_cost.txt",time_cost,fmt = '%.3f',delimiter = ' ')
                        sc.savetxt("num_frame.txt",num_frame,fmt = '%.2f',delimiter = ' ')
                        plt.figure()
                        plt.plot(num_frame,time_cost)
                        plt.xlabel("frames")
                        plt.ylabel("time cost(ms)")
                        plt.show()
                        dlib.hit_enter_to_continue()
                else:
                    print ('Face Not Found')
            
        cv2.destroyAllWindows()
           
    else:
        print ('No Camera')
import dlib
import glob
from skimage import io
import os
def runfile():
    """Recognize the facial expression of face pictures
    """
    predictor_path = 'E:/GithubProject/EmotionClassifier-master/shape_predictor_68_face_landmarks.dat'
    faces_folder_path = 'E:/GithubProject/EmotionClassifier-master/image/'
    
    classifier = emotionclassifier.EmotionClassifier(8, session_path)
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(predictor_path)
    #win = dlib.image_window()
    i=0
    label = []
    time_cost_per_frame = []
    nframe = []
    num = 0
    for im_file in os.listdir(faces_folder_path):
        for f in glob.glob(os.path.join(faces_folder_path,im_file,'*.png')):
            
            if i%5==0:
                print("Processing file: {}".format(f))
                img = io.imread(f)
               
                #win.clear_overlay()
                #win.set_image(img)
                
                face = []
                #face = builddata.get_face(img)
                face.append(cv2.resize(img,(88,88)))
                start = time.clock()
                classification = classifier.classify(face)
                end = time.clock()
                num = num+1
                nframe.append(num)
                temp =float('{:.3f}'.format((1000*(end-start))))
                time_cost_per_frame.append(temp)
                #label.append(classification)
                #print (' classified as '+str(numpy.argmax(classification[0]))+' in '+str(end-start)+'s')
                print (classification)
                print (temp)
                print (num)
                if num==120:
                    import matplotlib.pyplot as plt
                    import scipy as sc
                    sc.savetxt("time_cost_per_frame.txt",time_cost_per_frame,fmt = '%.3f',delimiter = ' ')
                    sc.savetxt("nframe.txt",nframe,fmt = '%.2f',delimiter = ' ')
                    plt.figure()
                    plt.plot(nframe,time_cost_per_frame)
                    plt.xlabel("帧数")
                    plt.ylabel("每帧耗时(ms)")
                    plt.show()
                    dlib.hit_enter_to_continue()
            i = i+1      
        i=0

if __name__ == '__main__':
    while 1:       
        # bulid_data_set()
        train()
        accuracy()
        runvideo()
        #runfile()
       

