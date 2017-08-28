import os
from PIL import Image
import pandas
import pickle
import numpy as np

data_dir = '../Data/celebA_data/img_align_celeba/'
save_dir = '../Data/celebA_data/resized_celebA_hair/celebA/'

image_size = 64
crop_size = 150

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_list = os.listdir(data_dir)

label = []
black_hair_idx = 9
brown_hair_idx = 12
data = pandas.read_csv('../Data/celebA_data/list_attr_celeba.csv', usecols=(0, black_hair_idx, brown_hair_idx)).values.squeeze()

new_data = []
cnt = 0
for i in range(len(img_list)):
    if data[i][1] + data[i][2] == 0:  # if only one of label equals to 1
        img = Image.open(data_dir + data[i][0])
        c_x = (img.size[0] - crop_size) // 2
        c_y = (img.size[1] - crop_size) // 2
        img = img.crop([c_x, c_y, c_x + crop_size, c_y + crop_size])
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img.save(save_dir + data[i][0], 'JPEG')

        if data[i][1] == -1:
            data[i][1] = 0

        new_data.append(data[i][1])

        cnt += 1
        if cnt % 1000 == 0:
            print('Resizing %s images...' % cnt)

print('Total %d images are resized.' % cnt)

with open('../Data/celebA_data/hair_label.pkl', 'wb') as fp:
    pickle.dump(np.array(new_data), fp)