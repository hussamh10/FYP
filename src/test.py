from matplotlib import pyplot as plt
from generator import generateENETRandom as gen
from generator import countFolderImages as st

st(4, '..\\drums\\')

g = gen(12000, 4, '..\\drums\\', '', '.jpg')

for i in range(1, 1000000):
    next(g)
    print('k')
