from cv2 import imread
import numpy as np
import numpy as np
import cv2
import os

def getImage(i, source, main_dir):
    name = str(i) + '.jpg'

    path = os.path.join(main_dir, source, name)
    img = imread(path, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = cv2.resize(img, (224, 224))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255
    return img


def getData(end, start=1, video=1):
    dir = os.path.join("..", 'data', 'data', str(video))
    inputs = getUnetData(end, start, dir)
    labels = getLabels(end, start, dir)

    return inputs, labels

def getUnetData(end, start=0, dir='..\\data\\data\\'): #data/unet/imgs/1.jpg
    imgs = []
    img_src1 = ''
    img_src2 = ''

    for i in range(start, end):
        image1 = getImage(i, img_src1, dir)
        image2 = getImage(i+1, img_src2, dir)
        image1 = image1.reshape((224, 224, 1))
        image2 = image2.reshape((224, 224, 1))

        image = np.concatenate([image1, image2], axis=2)
        imgs.append(image)
        
    imgs = np.array(imgs)
    return imgs

def getLabels(end, start=0, dir='data\\ynet\\'): #data/ynet/labels
    labels = []
    label_src = ''

    for i in range(start, end):
        labels.append(getImage(i+2, label_src, dir))

    labels = np.array(labels)
    return labels
