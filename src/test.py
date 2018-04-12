from matplotlib import pyplot as plt
from generator import generateENETRandom as gen
from generator import countFolderImages as st

st(2, '..\\data\\')

g = gen(12000, 2, '..\\data\\', '', '.jpg')

for i in range(1, 1000000):
    next(g)
    print('k')
