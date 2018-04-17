import keras.losses as loss
from data import getImages
import tensorflow as tf
import numpy as np

def MSE():
	sess = tf.Session()
	x, y = getImages()
	err = np.mean(np.mean(sess.run(loss.MSE(x, x))))
	print("MSE Error on same image: " + str(err))
	err = np.mean(np.mean(sess.run(loss.MSE(x, y))))
	print("MSE Error on blurred difference: " + str(err))
	sess.close()

def BCE():
	sess = tf.Session()
	x, y = getImages()
	#err = np.mean(np.mean(sess.run(loss.binary_crossentropy(x, x))))
	#print("BCE Error on same image: " + str(err))
	err = np.mean(np.mean(sess.run(loss.binary_crossentropy(x, y))))
	print("BCE Error on blurred difference: " + str(err))
	sess.close()
	
MSE()
BCE()