from matplotlib import pyplot as plt
from generator import generateYNET as gen
from generator import countFolderImages as st

g = gen(100, 2, '..\\drums\\', '', 'audio\\', '.jpg', '.png')

for i in range(1, 1000000):
    next(g)
    print('k')
