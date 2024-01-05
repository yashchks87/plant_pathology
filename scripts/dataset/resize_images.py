from PIL import Image
import numpy as np
import multiprocessing as mp
import glob
from itertools import repeat
import os, pickle
import argparse


def resize_image(data):
    try:
        img_path, new_path, img_size = data
        img_name = img_path.split('/')[-1]
        new_img_path = new_path + img_name
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img.save(new_img_path)
        return True
    except:
        return img_path

def resize_images(folder_path, new_path, cpu_count = 10, store_pickle = False, img_size = 224):
    assert os.path.exists(folder_path) == True, 'Input folder path does not exist'
    if os.path.exists(new_path) == False:
        os.makedirs(new_path)
    files = glob.glob(folder_path + '*.jpg')
    data = (zip(files, repeat(new_path), repeat(img_size)))
    with mp.Pool(cpu_count) as p:
        returns = list(p.map(resize_image, data))
    issues = []
    for x in returns:
        if x != True:
            issues.append(x)
    if len(issues) != 0:
        if store_pickle == True:
            pickle.dump(issues, open('./' + 'issues.pkl', 'wb'))
        else:
            return issues
    else:
        'No issues found'

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', help = 'Path to folder containing images')
    parser.add_argument('--new_path', help = 'Path to folder to store resized images')
    parser.add_argument('--cpu_count', type=int, help = 'Number of CPUs to use', default = 10)
    parser.add_argument('--store_pickle', type=bool, help = 'Store pickle file of issues', default = False)
    parser.add_argument('--img_size', type = int, help = 'Size of image', default = 224)
    args = parser.parse_args()
    resize_images(args.folder_path, args.new_path, args.cpu_count, args.store_pickle, args.img_size)


# python resize_images.py --folder_path ../../../plant_path_data/train_images/ --new_path ../../../plant_path_data/resized_train_224/ --cpu_count 28 --store_pickle False --img_size 224