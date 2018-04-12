from time import time
from enet import myUnet

enet = myUnet(224, 224)

enet.train(170, 'data\\', '', '.jpg', str(time()))

