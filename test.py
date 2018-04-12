from matplotlib import pyplot as plt
from generator import generateENET as gen

g = gen(2, 'data/', '', '.jpg')

for i in range(1, 1000000):
    next(g)
