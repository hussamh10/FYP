import keras.losses as loss
from data import getImages
import tensorflow as tf

def MSE():
	sess = tf.Session()
	x, y = getImages()
	print("MSE Error on same image: " + str(sess.run(loss.MSE(x, x))))
	print("MSE Error on blurred difference: " + str(sess.run(loss.MSE(x, y))))
	
	
def BCE():
	sess = tf.Session()
	x, y = getImages()
	print("BCE Error on same image: " + str(sess.run(loss.binary_crossentropy(x, x))))
	print("BCE Error on blurred difference: " + str(sess.run(loss.binary_crossentropy(x, y))))
	
	

MSE()
BCE()