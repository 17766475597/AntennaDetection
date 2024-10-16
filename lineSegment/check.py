import os
image_dir = os.listdir('all_data1')

jpgImg = [f for f in image_dir if f.endswith('.jpg')]
for f in jpgImg:
    mask = f[:-3] + 'png'
    print(mask)
    if f not in image_dir:
        print(mask)

