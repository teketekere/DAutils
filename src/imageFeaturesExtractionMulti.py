from collections import defaultdict
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
import gc
from multiprocessing import Pool
import multiprocessing
from copy import deepcopy
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    

# global vars
## detect this path depends on envs
images_path = '../input/ants/'

features_path = '../features/'

# define functions

def load_image(img, usecv2=False):
    path = images_path + img
    try:
        im = cv2.imread(path) if usecv2==True else IMG.open(path)
    except Exception as e:
        print('Cannot open img: ', images_path + img)    
    return im

def crop_horizontal(im):
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))
    return im1, im2

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1

    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis_black(img):
    im = load_image(img)
    im1, im2 = crop_horizontal(im)

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        print('Calculation error')
        return None

    dark_percent = (dark_percent1 + dark_percent2)/2 
    return dark_percent

def perform_color_analysis_white(img):
    im = load_image(img)
    im1, im2 = crop_horizontal(im)

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        print('Calculation error')
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    return light_percent

def average_pixel_width(img):
    im = load_image(img)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(img):
    im = load_image(img, usecv2=True)
    arr = np.float32(im)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(im.shape)

    dominant_color = palette[np.argmax(np.unique(labels, return_counts=True)[1: ])]
    return dominant_color

def get_average_color(img):
    im = load_image(img, usecv2=True)
    average_color = [im[:, :, i].mean() for i in range(im.shape[-1])]
    return average_color

def get_size(img):
    filename = images_path + img
    st = os.stat(filename)
    return st.st_size

def get_dimensions(img):
    im = load_image(img)
    img_size = im.size
    return img_size

def get_blurrness_score(img):
    im = load_image(img, usecv2=True)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(im, cv2.CV_64F).var()
    return fm

def get_arraylike_features(feature, inputcol, method, outputcol):
    retdf = pd.DataFrame()
    retdf[outputcol] = feature[inputcol].apply(method)
    print(outputcol, ' done')
    return retdf

def get_imagefeatures(features, imgcol, prefix='', n_workers=1):
    prefix = prefix + '_'
    feature_params = [[perform_color_analysis_black, prefix+'dullness'],
                      [perform_color_analysis_white, prefix+'whiteness'],
                      [average_pixel_width,          prefix+'average_pixel_width'],
                      [get_dominant_color,           prefix+'dominant_color'],
                      [get_average_color,            prefix+'average_color'],
                      [get_size,                     prefix+'image_size'],
                      [get_dimensions,               prefix+'dim_size'],
                      [get_blurrness_score,          prefix+'blurrness']
                     ]
    params = [(features, imgcol, m, c) for m, c in feature_params]
    with timer('Image features extraction'):
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(get_arraylike_features, params)
        
        for r in results:
            features[r.columns[0]] = r
        features[prefix+'dominant_red'] = features[prefix+'dominant_color'].apply(lambda x: x[0]) / 255
        features[prefix+'dominant_green'] = features[prefix+'dominant_color'].apply(lambda x: x[1]) / 255
        features[prefix+'dominant_blue'] = features[prefix+'dominant_color'].apply(lambda x: x[2]) / 255
        features[prefix+'average_red'] = features[prefix+'average_color'].apply(lambda x: x[0]) / 255
        features[prefix+'average_green'] = features[prefix+'average_color'].apply(lambda x: x[1]) / 255
        features[prefix+'average_blue'] = features[prefix+'average_color'].apply(lambda x: x[2]) / 255
        features[prefix+'width'] = features[prefix+'dim_size'].apply(lambda x : x[0])
        features[prefix+'height'] = features[prefix+'dim_size'].apply(lambda x : x[1])
        features.drop(prefix+'dominant_color', axis=1, inplace=True)        
        features.drop(prefix+'average_color', axis=1, inplace=True)
        features.drop(prefix+'dim_size', axis=1, inplace=True)
    gc.collect()
    return features


def save_features(features, filename, imgcol, deleteimgpath=True):
    if deleteimgpath == True:
        features.drop(imgcol, axis=1, inplace=True)
    filepath = features_path + filename
    features.to_feather(filepath)

if __name__ == '__main__':
    imgs = os.listdir(images_path)
    features = pd.DataFrame()
    features['imagepath'] = imgs

    numcpu = multiprocessing.cpu_count()
    print(f'use {numcpu} cpus')
    get_imagefeatures(features, 'imagepath', prefix='debug', n_workers=numcpu)

    save_features(features, 'sample.feather', imgcol='imagepath', deleteimgpath=True)
