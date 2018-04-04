from cv2 import imread
import cv2
import os
from enet import get_unet
from hussam_data import getData as gd
from matplotlib import pyplot as plt
import keras
import numpy as np

def save(data, i, path):
    data = data.astype('float32')
    img = plt.imshow(data, interpolation='nearest')
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig(path + str(i) + ".png", bbox_inches='tight')

def show(x):
    plt.imshow(x.reshape(x.shape[1], x.shape[2]), cmap='gray')
    plt.show()

def getImage(i, source, main_dir, ext, size):
    name = str(i) + ext
    print(main_dir + source + '/' + name)

    path = os.path.join(main_dir, source , name)
    img = imread(path, 0)

    img = cv2.resize(img, size)
    img = img.reshape((img.shape[0], img.shape[1], 1))

    img = img.astype('float32')
    img /= 255

    return img

def ok():
    i = 12
    dir = '../data/10'
    frame_ext = '.jpg'
    x1 = getImage(i  , '', dir, frame_ext, (224, 224))
    x2 = getImage(i+1, '', dir, frame_ext, (224, 224))
    x3 = getImage(i+2, '', dir, frame_ext, (224, 224))

    x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
    x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
    x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

    x1_2 = np.concatenate([x1, x2], axis=3)

    return x1_2, x3, x1, x2

def test(start, video, tag):
    md = get_unet()

    md.load_weights('enet_4-april.hdf5') 

    h, hy, x1, x2 = ok()#gd(start = start, end = start + 5, video=video)
    
    hp = md.predict(h)

    if not os.path.exists(tag):
        os.makedirs(tag)

    i = start
    for p, g in zip(hp, hy):
        save(p.reshape((224, 224)), i, tag+'x3')
        save(g.reshape((224, 224)), i, tag+'gt')
        save(x1.reshape((224, 224)), i, tag+'x1')
        save(x2.reshape((224, 224)), i, tag+'x2')
        i += 1

    print("Done")

parent = 'testing'


test(100, 1,  parent +'/similar_training_data/')
test(1, 2,  parent +'/traingin_data2/')
test(1, 4,  parent +'/traingin_data3/')
test(1, 6,  parent +'/traingin_data4/')
test(1, 10,  parent +'/test2/')
test(1, 22,  parent +'/test3/')
test(1, 23,  parent +'/test4/')
test(1, 24,  parent +'/test5/')
test(1, 28,  parent +'/test6/')
test(1, 42,  parent +'/test7/')
