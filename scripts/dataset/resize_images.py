import PIL
import numpy as np
import multiprocessing as mp
import glob
from itertools import repeat
import os, pickle


def resize_image(data):
    try:
        img_path, new_path, img_size = data
        img_name = img_path.split('/')[-1]
        new_img_path = new_path + img_name
        img = PIL.Image.open(img_path)
        img = img.resize((img_size, img_size))
        img.save(new_img_path)
        return True
    except:
        return img_path

def resize_images(folder_path, new_path, cpu_count = 10, store_pickle = False, img_size = 224):
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

    
