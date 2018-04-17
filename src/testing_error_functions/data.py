from cv2 import imread
from cv2 import resize
from matplotlib import pyplot as plt

def getImages():
	x = imread('data/1.png', 0)
	y = imread('data/2.png', 0)
	
	x = x.astype('float32')
	x /= 255
	y = y.astype('float32')
	y /= 255
	
	x = resize(x, (224, 224))
	y = resize(y, (224, 224))
	
	x = x.reshape((224, 224))
	y = y.reshape((224, 224))
	
	return x, y


def print():
	x, y = getImages()

	plt.imshow(x, cmap='gray')
	plt.show()
	
	plt.imshow(y, cmap='gray')
	plt.show()
	
if __name__ == '__main__':
	print()