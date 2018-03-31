from matplotlib import pyplot as plt
from generator import generateENET as gen

g = gen(170, '../data/', '', '.jpg')

for i in range(1, 100):
    next(g)
