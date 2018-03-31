from matplotlib import pyplot as plt
from generator import generateENET as gen

g = gen(100, 10, '..\\data\\clean\\', '', '.jpg')

for i in range(1, 100):
    x1, x2, gt = next(g)

    plt.imshow(x1.reshape(224, 224), cmap='gray')
    plt.show()

    plt.imshow(x2.reshape(224, 224), cmap='gray')
    plt.show()

    plt.imshow(gt.reshape(224, 224), cmap='gray')
    plt.show()

