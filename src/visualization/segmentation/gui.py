import pandas as pd
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import json
import os

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv(
    '../../data/data_tr_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon']

# create image
X = df.as_matrix()
min_lat, max_lat = X[:, 0].min(), X[:, 0].max()
min_lon, max_lon = X[:, 1].min(), X[:, 1].max()
WINDOW_SIZE = 600


def map2img(lat, lon):
    x = WINDOW_SIZE*(lon-lonloc)/mapsize
    y = WINDOW_SIZE*(lat-latloc)/mapsize
    if isinstance(x, float):
        return int(x), int(y)
    return x, y


def img2map(x, y):
    return mapsize*y/WINDOW_SIZE+latloc, mapsize*x/WINDOW_SIZE+lonloc


def city_image(latloc, lonloc, mapsize):
    res = WINDOW_SIZE
    city = np.zeros((res, res), np.uint8)
    
    X = df.as_matrix().copy()
    X[:, 0], X[:, 1] = map2img(X[:, 0], X[:, 1])
    X = X.astype(int)
    
    X[:, 0][X[:, 0] < 0] = 0
    X[:, 0][X[:, 0] >= res] = res-1
    X[:, 1][X[:, 1] < 0] = 0
    X[:, 1][X[:, 1] >= res] = res-1
    
    city[X[:, 0], X[:, 1]] = 255
    return np.repeat(city[:, :, np.newaxis], 3, 2)

print('build city matrix...')

mapsize = 4
latloc = min_lat
lonloc = min_lon

if os.path.exists('roads.json'):
    with open('roads.json', 'r') as f:
        roads = json.load(f)
else:
    roads = [[]]


def redraw():
    clip = city_image(latloc, lonloc, mapsize)
    print('loc:', latloc, lonloc)
    road_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, road in enumerate(roads):
        color = road_colors[i % len(road_colors)]
        for pt1, pt2 in zip(road[:-1], road[1:]):
            cv2.line(clip, map2img(*pt1), map2img(*pt2), color)
        for pt in road:
            cv2.circle(clip, map2img(*pt), 4, (255, 0, 0))
    cv2.imshow('win', clip)


def zoom(value):
    global mapsize, latloc, lonloc
    if value > 0:
        mapsize = mapsize/2
    else:
        mapsize = mapsize*2
    redraw()


def mouse_cb(event, x, y, value, data):
    if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
        zoom(value)
    elif event == cv2.EVENT_LBUTTONDOWN:
        roads[-1].append(img2map(x, y))
        redraw()

print('show')
cv2.namedWindow('win')
cv2.setMouseCallback('win', mouse_cb)

while True:
    redraw()
    key = cv2.waitKey()
    if key in (27, 255):  # escape
        break

    move = mapsize/2
    if key == 83:  # right
        latloc = latloc + move
    elif key == 81:  # left
        latloc = latloc - move
    elif key == 82:  # up
        lonloc = lonloc - move
    elif key == 84:  # down
        lonloc = lonloc + move
    elif key == 10:  # return
        roads.append([])
    elif key == ord('c'):
        mapsize = 4
        latloc = min_lat
        lonloc = min_lon
    elif key == ord('+'):
        zoom(1)
    elif key == ord('-'):
        zoom(-1)
    else:
        print('key unknown: %d' % key)

cv2.destroyWindow('win')

with open('roads.json', 'w') as f:
    json.dump(roads, f)
