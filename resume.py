from time import time
from enet import myUnet

enet = myUnet(224, 224)
enet.load_weights(' osama put hdf5 name here')

enet.train(170, 'osama data directory here', '', '.jpg', str(time()))

