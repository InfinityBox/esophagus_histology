import os
import glob
import numpy as np
import openslide
import cv2
import json
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from shapely.geometry import LineString


# def isPoiWithinPoly(poi, poly):
#     point_index = []
#     for i in range(len(poly)-1):
#         x1, y1 = poly[i]
#         x2, y2 = poly[i+1]
#         dis = ((x1-poi[0]) **2 + (y1-poi[1]) **2) **0.5 + ((x2-poi[0]) **2 + (y2-poi[1]) **2) ** 0.5
#         if i == 0:
#             dis_min = dis
#         else:
#             if dis_min > dis:
#                 dis_min = dis
#
#         point_index.append(i)
#
#     return point_index
#
# def intersect(vertices):
#     l1 = LineString(vertices[i])
#     l2 = LineString(vertices[i-1])
#     inter = l1.intersection(l2)
#
#     if not any(inter.bounds):
#         return None
#     else:
#         x1, y1 = inter.bounds[0], inter.bounds[1]
#         if inter.bounds[0] != inter.bounds[2]:
#             x2, y2 = inter.bounds[0], inter.bounds[1]
#         else:
#             x2, y2 = None, None
#     return (x1, y1, x2, y2)


def takeminx(elem):
    return elem[0][0]


def combinelines(lines):
    i = 0
    min = []
    for m in lines:
        m = np.asarray(m)
        n = len(m)
        if m[0][0] > m[n-1][0]:
            m = np.flip(m, 0)
            lines[i] = np.flip(lines[i], 0)
        else:
            lines[i] = m
        min.append([m[0], m[n-1], i])
        i += 1

    min = sorted(min, key=takeminx)
    index = []
    for i in range(len(min)):
        index.append(min[i][2])
        if i > 1:
            x1, y1 = min[i-2][1][0], min[i-2][1][1]
            x2, y2 = min[i-1][0][0], min[i-1][0][1]
            x3, y3 = min[i][0][0], min[i][0][1]
            dis1 = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            dis2 = ((x3-x1)**2 + (y3-y1)**2) ** 0.5
            if dis2 < dis1:
                assert print(i, ':distance wrong')
    line = []
    for l in index:
        line.append(lines[l])
    line = np.concatenate(line)
    return line


level = 5
RGB_min = 50
json_path = 'C:\\Users\\An\\Desktop\\585595_28.vsi - 20x.json'
npy_path = 'C:\\Users\\An\\Desktop\\585595_28.npy'
slide_path = 'G:\\esophageal\\tif\\585595_28.tif'
img_save_path = 'C:\\Users\\An\\Desktop\\585595_27.png'

slide = openslide.OpenSlide(slide_path)
img_RGB = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))

img_HSV = rgb2hsv(img_RGB)

background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
tissue_RGB = np.logical_not(background_R & background_G & background_B)
tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
min_R = img_RGB[:, :, 0] > RGB_min
min_G = img_RGB[:, :, 1] > RGB_min
min_B = img_RGB[:, :, 2] > RGB_min

tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

w, h = slide.level_dimensions[level]
mask_tumor = np.zeros((h, w))  # the init mask, and all the value is 0
factor = int(slide.level_downsamples[level])

with open(json_path) as f:
    dicts = json.load(f)


vertices = []

for i in range(7):
    exec('lines'+str(i)+'=[]')
    for key, value in dicts.items():
        if value['properties']['classification']['name'] == '_{}'.format(i):
            eval('lines{}'.format(i)).append(value['geometry']["coordinates"])

    exec('line'+str(i)+'= combinelines(lines{})'.format(i))
    exec('vertices'+ str(i) +'= (line{} / factor).astype(np.int32)'.format(i))
    exec('vertices.append(vertices{})'.format(i))
    if i == 0:
        v = np.insert(vertices[0], 0, [0, mask_tumor.shape[0]], axis=0)
        v = np.insert(v, len(v), [mask_tumor.shape[1], mask_tumor.shape[0]], axis=0)
    # elif i == 6:
    #     v = np.insert(vertices[6], 0, [0, 0], axis=0)
    #     v = np.insert(v, len(v), [mask_tumor.shape[1], 0], axis=0)
    else:
        # inter = intersect(vertices)
        # if inter is not None:
        #     if None in inter:
        #         x1, y1, _, _ = inter
        #         coord = np.array([x1, y1])
        #         ind = isPoiWithinPoly(coord, vertices[i])
        #     else:
        #         x1, y1, x2, y2 = inter
        #         coord1 = np.array([x1, y1])
        #         coord2 = np.array([x2, y2])
        #         ind1 = isPoiWithinPoly(coord1, vertices[i])
        #         ind2 = isPoiWithinPoly(coord2, vertices[i])
        #     v = np.concatenate((vertices[i], np.flip(vertices[i-1], 0)))
        # else:
        v = np.concatenate((vertices[i], np.flip(vertices[i-1], 0)))

    cv2.fillPoly(mask_tumor, [v], ((i+1)*30))
    mask_tumor = mask_tumor * tissue_mask

np.save(npy_path, mask_tumor)
plt.imshow(mask_tumor)
plt.axis('off')
plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0, dpi=300)