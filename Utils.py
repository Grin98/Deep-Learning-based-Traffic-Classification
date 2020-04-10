import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_flow_pic(pic):
    x = pic.numpy()
    x = x.transpose()
    # changes values to 1 so that the image would be in black and white with no gray scale
    x[x >= 1] = 1
    plt.imshow(x, cmap='binary')
    plt.gca().invert_yaxis()
    plt.show()
