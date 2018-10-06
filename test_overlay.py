from PIL import Image
import numpy as np
import itertools
import random

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int8")
    return data

bg = load_image('test_bg.jpg')
icon = load_image('test_icon.jpg')

# Find all unique possible locations to place icon over bg
bg_height, bg_width, _ = bg.shape
icon_height, icon_width, _ = icon.shape

# Must start with y + icon_height and x + icon_width room to grow
start_coors = [(x, y) for x,y in itertools.product(range(0, bg_width-icon_width), range(0, bg_height-icon_height))]
random.shuffle(start_coors)

# Select n random places (should be config driven)
n = 10
start_coors = start_coors[:n]

print(start_coors)

for start_y,start_x in start_coors:
    bg_instance = np.array(bg)
    
    for x in range(start_x, start_x + icon_width):
        for y in range(start_y, start_y + icon_height):
            bg_instance[x][y] = icon[x-start_x][y-start_y]

    img = Image.fromarray(bg_instance, 'RGB')
    img.show()
    