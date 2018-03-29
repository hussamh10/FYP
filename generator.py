import numpy as np
import cv2
from cv2 import imread
import os
import random


def getImage(main_dir, source, pre, name, ext):
    path = os.path.join(main_dir, source, pre, name + ext)
    print(path)

    img = imread(path, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = cv2.resize(img, (224, 224))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255

    return img


def getRandom(min, max):
    return random.randint(min, max)

def generateENET(image_limit, folder_limit, main_dir, frame_pre, frame_ext):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)

        i = getRandom(1, image_limit)

        x1 = getImage(main_dir, source, '', str(i), frame_ext)
        x2 = getImage(main_dir, source, '', str(i+1), frame_ext)
        g  = getImage(main_dir, source, '', str(i+2), frame_ext)

        x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

        x12 = np.concatenate([x1, x2], axis=3)

        print('')
        yield(x12, g)

        i += 1

def generateYNET(image_limit, folder_limit, main_dir, frame_pre, audio_pre, frame_ext, audio_ext):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)

        i = getRandom(1, image_limit)

        x = getImage(main_dir, source, frame_pre, str(i), frame_ext)
        a = getImage(main_dir, source, audio_pre, str(i), audio_ext)
        g = getImage(main_dir, source, frame_pre, str(i+1), frame_ext)

        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

        print('')
        i += 1

        yield([x, a], g)
