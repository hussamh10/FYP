from generator import generateENET as gen

g = gen(100, 10, '..\\data\\data\\', '.jpg', '.png')

for i in range(1, 100):
    next(g)
