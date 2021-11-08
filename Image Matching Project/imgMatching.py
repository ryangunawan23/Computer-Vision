import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

"""
    STEP BY STEP
    1. baca interest image OK
    2. baca target image OK
    3. bikin model feature matching OK
    4. dapatkan keypoints dan descriptor dari interest image OK
    5. loop, no 4 OK
    6. matching antara tiap scene image dengan interest image OK
    7. Kalkulasi banyak matcher yg ada tiap komparasi OK
    8. tampilkan interest omage dengan scene image yg sesuai
"""

base_path = './Dataset/Data'

scene_images = []

target_image = cv.imread('Dataset/Object.jpg')
target_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))

for i in os.listdir(base_path):
    file = i.split('.')
    if file[1] == 'jpg':
        image_path = cv.imread(base_path + '/' +i)
        image_path = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
        image_path = clahe.apply(image_path)
        scene_images.append(image_path)

SIFT = cv.xfeatures2d.SIFT_create()

# kp + ds target image
target_kp, target_ds = SIFT.detectAndCompute(target_image, None)

KDTREE_INDEX = 1 # 0 1 untuk SIFT and SURF, 5 for ORB
TREE_CHECKS = 50

FLANN = cv.FlannBasedMatcher(dict(algorithm = KDTREE_INDEX), dict(checks=TREE_CHECKS))

all_mask = []
scene_index = -1
total_match = 0
scene_keyp = None
final_match = None

for index, i in enumerate(scene_images):
    scene_kp, scene_ds = SIFT.detectAndCompute(i, None)
    matcher = FLANN.knnMatch(target_ds, scene_ds, 2)

    # scene_mask = [[0,0] for j in range(0, len(matcher))]
    scene_mask = []
    for j in range(0, len(matcher)):
        scene_mask.append([0,0])
    match_count = 0

    
    # [1, 0] -> 1st match
    # [0, 1] -> 2nd match
    # [0, 0] -> tidak ambil match samsek
    

    for j, (first_match, second_match) in enumerate(matcher):
        if first_match.distance < 0.7 * second_match.distance:
            scene_mask[j] = [1,0]
            match_count +=1
    
    all_mask.append(scene_mask)

    if total_match < match_count:
        total_match = match_count
        scene_index = index
        scene_keyp = scene_kp
        final_match = matcher

result = cv.drawMatchesKnn(
    target_image, target_kp,
    scene_images[scene_index], scene_keyp,
    final_match, None,
    matchColor = [0,255,0],
    singlePointColor=[255,0,0],
    matchesMask=all_mask[scene_index]
)

plt.imshow(result, cmap='gray')
plt.show()








