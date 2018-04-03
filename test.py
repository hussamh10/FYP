from matplotlib import pyplot as plt
from generator import generateENET as gen

def show(x):
    plt.imshow(x.reshape(x.shape[1], x.shape[2]), cmap='gray')
    plt.show()


g = gen(4, '../data/', '', '.jpg')

for i in range(1, 100):
    (a, b, c) = next(g)
    show(a)
    show(b)
    show(c)
