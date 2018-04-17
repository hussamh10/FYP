import numpy as np
import cv2
from cv2 import imread
import os
import random

arr = []

def getImage(main_dir, source, pre, name, ext):
    path = os.path.join(main_dir, source, pre, name + ext)
    img = imread(path, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = cv2.resize(img, (224, 224))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255

    return img

def getRandom(min, max):
    return random.randint(min, max)

def generateYNET(image_limit, folder_limit, main_dir, frame_pre, audio_pre, frame_ext, audio_ext):
    folder = 1
    while folder <= folder_limit:
        i = 1
        while True:
            source = str(folder)

            fname = os.path.join(main_dir, source, '', str(i+1) + frame_ext)
            if not os.path.isfile(fname):
                break
             
            x = getImage(main_dir, source, frame_pre, str(i), frame_ext)
            a = getImage(main_dir, source, audio_pre, str(i), audio_ext)
            g = getImage(main_dir, source, frame_pre, str(i+1), frame_ext)

            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

            i += 1

            yield([x, a], g)

        folder += 1
        i = 1
        if folder > folder_limit:
            folder = 1

def generateENET(folder_limit, main_dir, frame_pre, frame_ext):
    folder = 1
    while folder < folder_limit:
        i = 2
        while True:
            source = str(folder)

            fname = os.path.join(main_dir, source, '', str(i+2) + frame_ext)

            if not os.path.isfile(fname):
                break

            x1 = getImage(main_dir, source, '', str(i), frame_ext)
            x2 = getImage(main_dir, source, '', str(i+1), frame_ext)
            g  = getImage(main_dir, source, '', str(i+2), frame_ext)

            x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
            x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

            x12 = np.concatenate([x1, x2], axis=3)

            i += 1

            yield(x12, g)


        folder += 1
        if folder == folder_limit:
            folder = 1


def generateENETRandom(image_limit, folder_limit, main_dir, frame_pre, frame_ext):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)
        
        #dataPath = main_dir + "\\" + source + "\\"
        #print (dataPath) #commented out for performance gains
        #a, b, files = os.walk(dataPath).__next__() #a and b not required by us

        #i = getRandom(2, len(files) - 10) #-10 just to be safe :p ; also starting from 2 since 2nd video had corrupted first frame (LUL) and I'm too lazy to rename all the images
        
        i = getRandom(2, arr[folder] - 1) #starting from 2 since 2nd video had corrupted first frame (LUL) and I'm too lazy to rename all the images
        x1 = getImage(main_dir, source, '', str(i), frame_ext)
        x2 = getImage(main_dir, source, '', str(i+1), frame_ext)
        g  = getImage(main_dir, source, '', str(i+2), frame_ext)

        x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

        x12 = np.concatenate([x1, x2], axis=3)

        yield(x12, g)

def countFolderImages(folder_limit, main_dir):
    arr.append(0) #chaipi to start indexing from 1
    for i in range(1,folder_limit+1):
        dataPath = main_dir + "\\" + str(i) + "\\"
        a, b, files = os.walk(dataPath).__next__() #a and b not required by us
        arr.append(len(files)-3) #just to be safe
    print(arr)
