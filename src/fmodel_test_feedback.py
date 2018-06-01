from cv2 import imread
import cv2
import os
from fcnet import getFCNet as get_unet
import keras
import numpy as np

def save(data, i, path):
    data = data.astype('float32')
    cv2.imwrite(path + str(i) + ".jpg",data)

def getImage(i, source, main_dir, ext, size):
    name = source + str(i) + ext #for TT data

    path = os.path.join(main_dir, '' , name)
    print(path)
    img = imread(path, 0)
    img = cv2.resize(img, size)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def ok(start,videoType):
    i = start
    dir  = '..\\testing\\with_feedback\\' + str(videoType) + '\\e\\'
    dir2 = '..\\testing\\with_feedback\\' + str(videoType) + '\\y\\'
	
    frame_ext = '.jpg'
    x1 = getImage(i, '', dir , frame_ext, (224, 224))
    x2 = getImage(i, '', dir2, frame_ext, (224, 224))
    x3 = getImage(i, '', dir2, frame_ext, (224, 224))
    print("-------------------------------")

    x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
    x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
    x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

    return x1, x2, x3

def test():
    md = get_unet()
    frames = 200
    md.load_weights('..\\checkpoints\\fcnet1.hdf5')
    for videoType in range(1,2):
        tag = '..\\testing\\with_feedback\\' + str(videoType) + '\\f\\'
        for start in range(1,frames):
            x1, x2, hy = ok(start,videoType)
            
            
            hp = md.predict([x1, x2])
            
            if not os.path.exists(tag):
                os.makedirs(tag)
            
            i = start
            for p, g in zip(hp, hy):
                p *= 255;
                save(p.reshape((224, 224)), i, tag) #output
test()
