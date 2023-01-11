from image_generator import *

g = custom_data_gen('data/train', batch_size=1)
for i in g:
    print(i[0].shape)
    plot_image_from_Lab(i[0][0])
    plot_image_from_Lab(i[0][0], grayscale=True)
    break
