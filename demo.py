import os
import cv2
import pathlib
import numpy as np
from VPdetection import VPdetection as vpd
from VPdetection import rotatePanorama as rot

def img_concat(img_a, img_b):
    img_a = cv2.resize(img_a, (img_a.shape[1]//2, img_a.shape[0]))
    img_b = cv2.resize(img_b, (img_b.shape[1]//2, img_b.shape[0]))
    img_ab = np.concatenate((img_a, img_b), axis=1)
    return img_ab

def find_vp_and_score(img_rot):
    ## find best vp
    QUALITY = 7
    XYZ_VOTE_Q1 = 27
    XYZ_VOTE_Q2 = 370
    XYZ_VOTE_DIFF = 0.007
    NORM_DIFF = 0.01
    ONEPARA_MAX_VP = np.eye(3)
    ONEPARA_MAX_SCORE = 0
    for j in range(QUALITY):
        try:
            vp_new, vp_new_score = vpd(img_rot, XYZ_VOTE_DIFF, NORM_DIFF, XYZ_VOTE_Q1, XYZ_VOTE_Q2)
            print("vp: ", vp_new)
        except:
            vp_new_score = -1

        if(vp_new_score > ONEPARA_MAX_SCORE):
            ONEPARA_MAX_VP = vp_new
            ONEPARA_MAX_SCORE = vp_new_score

    return ONEPARA_MAX_VP, ONEPARA_MAX_SCORE


def img_alignment_compare(image_folder, output_folder='./output_img'):
    OUTPATH = pathlib.Path(output_folder)
    OUTPATH.mkdir(parents=True, exist_ok=True)
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    print("images: ", images)

    id=0
    for image in images:
        img_ori = cv2.imread(image_folder + '/' + image)
        width, height = (img_ori.shape[1], img_ori.shape[0])

        vp, score = find_vp_and_score(img_ori)
        img_inv = (rot(img_ori, vp)).astype(np.uint8)

        rnd = img_concat(img_ori, img_inv)
        cv2.imwrite(output_folder + "/%d_" % id + "ori_and_inv.jpg", rnd)
        id = id+1


image_folder = './test_img'
img_alignment_compare(image_folder)