##利用VPdetection來校正影片，輸入是原360影片輸出是校正過後的影片
###########################################################

import os
import cv2
import random
import sys
import glob
import random
import time
import ntpath
import pathlib
import itertools
import threading
import math as m
import numpy as np
from PIL import Image
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from VPdetection import VPdetection as vpd
from VPdetection import rotatePanorama as rot
from multiprocessing import Process, Pool
ntpath.basename("a/b/c")
from itertools import zip_longest



# def random_shaking_video(image_folder, video_name):
#     print("[RANDOM SHAKING SHAKING]")

#     fps = 30
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#     images = sorted(images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

#     print("start generate random rotations")
#     #################### slerp rnd rotations
#     FRAMENUM = len(images)


#     RND_VPLIST = [random_rotation_matrix() for _ in range(FRAMENUM//5)]


    
#     key_rots = R.from_matrix(RND_VPLIST)
#     key_times = np.linspace(0, len(RND_VPLIST)-1, num=len(RND_VPLIST), endpoint=True)
#     slerp = Slerp(key_times, key_rots)
#     times = np.linspace(0, len(RND_VPLIST)-1, num=FRAMENUM, endpoint=True)
#     interp_rots = slerp(times)
#     interp_rots = interp_rots.as_matrix()


#     for i in range(FRAMENUM):

#         img_fix = cv2.imread(os.path.join(image_folder, images[i]))
#         img_fix = cv2.resize(img_fix, (width, height))

#         rnd_rotation = interp_rots[i]
#         img_rot = rot(img_fix, rnd_rotation).astype(np.uint8)
#         video.write(img_rot)
        
#     cv2.destroyAllWindows()
#     video.release()


def random_rotation_matrix(pitch=None, roll=None, yaw=None):

    yaw = [0, 5, 10, 20, 30]
    pitch = [0, 5, 10, 20, 30]
    roll = [0, 5, 10, 20, 30]
    rnd_yaw = random.choice(yaw)
    rnd_pitch = random.choice(pitch)
    rnd_roll = random.choice(roll)


    rnd_matrix = produce_rotation_matrix(roll = rnd_roll, pitch = rnd_pitch, yaw = rnd_yaw)

    rot_str = 'y'+str(rnd_yaw)+'p'+str(rnd_pitch)+'r'+str(rnd_roll)
    return rot_str, rnd_matrix

############################################


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions


def produce_rotation_matrix(yaw=0, pitch=0, roll=0, degrees=True):
    '''
        @yaw   (z): 頭水平轉
        @pitch (x): 頭上下轉
        @roll  (y): 臉正對前方轉
    '''
    if yaw == 0 and pitch == 0 and roll == 0:
        return np.eye(3)
    seq, angles = [], []
    if yaw != 0:
        seq.append('z')
        angles.append(yaw)
    if pitch != 0:
        seq.append('x')
        angles.append(pitch)
    if roll != 0:
        seq.append('y')
        angles.append(roll)
    rot = R.from_euler(''.join(seq), angles, degrees=degrees)
    return rot.as_matrix().reshape(3,3)

def is_unitary(matrix: np.ndarray) -> bool:

    unitary = True
    n = matrix.shape[0]
    error = np.linalg.norm(np.eye(n) - matrix.dot(matrix.transpose().conjugate()))

    if not(error < np.finfo(matrix.dtype).eps * 10.0 *n):
        unitary = False

    return unitary

def Rdiff(RAO, RBO):
    # RBA = RAO.T @ RBO
    RAOT = R.from_matrix(RAO).inv().as_matrix()
    RBA = RAOT @ RBO
    RBA = R.from_matrix(RBA)
    return RBA.as_euler('xyz', degrees=True)

def mat2euler(M):
    return R.from_matrix(M).as_euler('xyz', degrees=True)

def test_Rot():
    RAO = R.from_euler('xyz', [0, 88, 0], degrees=True).as_matrix()
    RBO = R.from_euler('xyz', [88, 0, 88], degrees=True).as_matrix()

    print("RAO: ", RAO)
    print("RBO: ", RBO)
    print("Rdiff(RAO, RBO):", Rdiff(RAO, RBO))

    RAO_ = R.from_matrix(RAO)
    RBO_ = R.from_matrix(RBO)
    anglesA = RAO_.as_euler('xyz', degrees=True)
    anglesB = RBO_.as_euler('xyz', degrees=True)
    print("anglesA:　", anglesA)
    print("anglesB:　", anglesB)

def Rx(theta):
    return np.matrix([[ 1, 0         , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                    [ m.sin(theta), m.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

# def preclude_Rz(rot_matrix):
#     ang = mat2euler(rot_matrix)   ## in xyz order and degree convention
#     ang = [ang[0], ang[1], 0] ## preclude z axis rotation
#     pre_rot_matrix = R.from_euler('xyz', ang, degrees=True).as_matrix()
#     return pre_rot_matrix



###先把影片讀成一張一張的frame然後輸出成圖片到資料夾內
###尚未平行化 應該可以平行化
def frame2pic(video_path, START_FRAME_NUM, FRAME_NUM, DELAY=1, out_dir = "./videoframes", resize=0.5):
    print("[frame2pic]")

    OUTPATH = pathlib.Path(out_dir)
    OUTPATH.mkdir(parents=True, exist_ok=True)

    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = videoCapture.read()

    fram_num = 0
    fram_id = 0
    while success: 
        eps_time = time.time()
        success, frame = videoCapture.read()

        if(fram_id >= START_FRAME_NUM and fram_id % DELAY == 0 and success):
            ###resize frame
            height, width, layers = frame.shape
            frame = cv2.resize(frame, (int(width*resize), int(height*resize)))
            cv2.imwrite(out_dir + "/%d.jpg" % fram_id, frame)
            fram_num = fram_num + 1

        fram_id = fram_id + 1
        if(fram_num == FRAME_NUM):
            break

    # cv2.destroyAllwindows()
    videoCapture.release()

##Below is parallel frame processing
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


##對於一個固定的參數以及一張圖片找到最好的VP
def testParallel_Util_online(PARAMETER, FILEPATH):

    FILENAME = path_leaf(FILEPATH)
    img_rot = Image.open(FILEPATH)
    img_rot = np.array(img_rot)[..., :3].astype(np.uint8)
    img_rot_resize = Image.fromarray(img_rot).resize((1024, 512))
    img_rot_resize = np.array(img_rot_resize)[..., :3]

    ##load parameter and display
    ROUND = PARAMETER[0]
    QUALITY = PARAMETER[1]
    XYZ_VOTE_Q1 = PARAMETER[2]
    XYZ_VOTE_Q2 = PARAMETER[3]
    XYZ_VOTE_DIFF = PARAMETER[4]
    NORM_DIFF = PARAMETER[5]

    print("PROCESSING FRAME NO.", FILENAME)
    
    ##testing
    ONEPARA_MAX_SCORE = 0
    ONEPARA_MAX_VP = np.identity(3)

    ## find best vp
    for j in range(QUALITY):
        try:
            vp_new, vp_new_score = vpd(img_rot_resize, XYZ_VOTE_DIFF, NORM_DIFF, XYZ_VOTE_Q1, XYZ_VOTE_Q2)
        except:
            vp_new_score = -1

        if(vp_new_score > ONEPARA_MAX_SCORE):
            ONEPARA_MAX_VP = vp_new
            ONEPARA_MAX_SCORE = vp_new_score

    return ONEPARA_MAX_VP, ONEPARA_MAX_SCORE

##對於一個固定的參數平行化去找所有圖片最好的VP
##找到之後用內插的方式找到所有VP
def testPrallel_online(test_file_name, PARAMETERS, IMGINFOLIST=None, OUTPATH_STR = './test_out'):
    ##create root folder
    OUTPATH = pathlib.Path(OUTPATH_STR)
    OUTPATH.mkdir(parents=True, exist_ok=True)

    ##load image
    FILEPATHS = glob.glob(test_file_name)

    ##load parameter and set file name
    ROUND = PARAMETERS['ROUND'][0]
    QUALITY = PARAMETERS['QUALITY'][0]
    XYZ_VOTE_Q1 = PARAMETERS['XYZ_VOTE_Q1'][0]
    XYZ_VOTE_Q2 = PARAMETERS['XYZ_VOTE_Q2'][0]
    XYZ_VOTE_DIFF = PARAMETERS['XYZ_VOTE_DIFF'][0]
    NORM_DIFF = PARAMETERS['NORM_DIFF'][0]
    PARAMETER = [ROUND,QUALITY,XYZ_VOTE_Q1,XYZ_VOTE_Q2,XYZ_VOTE_DIFF,NORM_DIFF]


    #parameters for online test
    FRAMENUM = 100
    SCENE_GUARD_DELAY = 1 
    ANGLE_DIFF_TOL = 5 
    SCORE_WEIGHT_GL = 0.2
    INHERIT_WEIGHT_GL = 0.8
    INHERIT_WEIGHT_VP_LC = 0.9
    INHERIT_WEIGHT_NEWVP_LC = 0.1
    DETECTING_DELAY = 5 ##in frames
    USING_VP_DELAY = 5  ##>=DETECTING_DELAY

    SORTED_PATHS = sorted(FILEPATHS, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    if(IMGINFOLIST is None):
        # inputs = list(grouper('ABCDEFG', 5, 'x'))
        # inputs = [(PARAMETER, SORTED_PATH) for SORTED_PATH in SORTED_PATHS]
        inputs = [(PARAMETER, SORTED_PATHS[i*DETECTING_DELAY]) for i in range(len(SORTED_PATHS)//DETECTING_DELAY)]
        with Pool(processes=4) as pool:
            results = pool.starmap(testParallel_Util_online, inputs)
        IMGINFOLIST = results
        np.save(OUTPATH_STR + '/imginfolist.npy', IMGINFOLIST)

    processingVP_interpolate(SORTED_PATHS, OUTPATH_STR, IMGINFOLIST, FRAMENUM,\
         USING_VP_DELAY, DETECTING_DELAY, SCENE_GUARD_DELAY, ANGLE_DIFF_TOL,\
              SCORE_WEIGHT_GL, INHERIT_WEIGHT_GL, INHERIT_WEIGHT_VP_LC,\
                   INHERIT_WEIGHT_NEWVP_LC)


##IMGINFOLIST => PROCESSEDVP
def processingVP_interpolate(INPUTPATH, OUTPATH_STR, IMGINFOLIST, FRAMENUM, USING_VP_DELAY, DETECTING_DELAY, SCENE_GUARD_DELAY, ANGLE_DIFF_TOL, SCORE_WEIGHT_GL, INHERIT_WEIGHT_GL, INHERIT_WEIGHT_VP_LC, INHERIT_WEIGHT_NEWVP_LC, SLERP=1):

    VPLIST = []
    vp = np.identity(3)
    vp_score = 0
    change_count = 0

    for i in range(FRAMENUM):
        VP, SCORE = IMGINFOLIST[i//DETECTING_DELAY]

        #detect new vp
        if(i%USING_VP_DELAY==0):
            
            ##stabalize
            angle_new = mat2euler(VP)
            angle_vp = mat2euler(vp)
            if(abs(angle_vp[0] - angle_new[0]) > ANGLE_DIFF_TOL \
                or abs(angle_vp[1] - angle_new[1]) > ANGLE_DIFF_TOL \
                or abs(angle_vp[2] - angle_new[2]) > ANGLE_DIFF_TOL):

                change_count = change_count + 1
                if(change_count > SCENE_GUARD_DELAY):
                    change_count = 0
                    vp = VP
            else:
                change_count = 0

                ##change vp, vpw is vp weight
                score_vpw = (vp_score/(vp_score+SCORE))*SCORE_WEIGHT_GL + INHERIT_WEIGHT_VP_LC*INHERIT_WEIGHT_GL
                score_bvpw = (SCORE/(vp_score+SCORE))*SCORE_WEIGHT_GL + INHERIT_WEIGHT_NEWVP_LC*INHERIT_WEIGHT_GL
                vpw = score_vpw/(score_vpw + score_bvpw)
                bvpw = score_bvpw/(score_vpw + score_bvpw)
                vp = vp*vpw + VP*bvpw
                vp_score = SCORE

            if(SLERP):
                # pre_vp = preclude_Rz(vp)
                pre_vp = vp
                VPLIST.append(pre_vp)
        if not SLERP:
            # pre_vp = preclude_Rz(vp)
            pre_vp = vp
            VPLIST.append(pre_vp)


    #VPLIST interpolation
    if(SLERP):
        key_rots = R.from_matrix(VPLIST)
        key_times = np.linspace(0, len(VPLIST)-1, num=len(VPLIST), endpoint=True)
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, len(VPLIST)-1, num=FRAMENUM, endpoint=True)
        interp_rots = slerp(times)
        interp_rots = interp_rots.as_matrix()
    else:
        interp_rots = VPLIST
    

    for i in range(FRAMENUM):
        vp = interp_rots[i]
        FILEPATH = INPUTPATH[i]
        img_rot = Image.open(FILEPATH)
        img_rot = np.array(img_rot)[..., :3].astype(np.uint8)
        FRAME = img_rot / 255.0
        FILENAME = path_leaf(FILEPATH)

        #write frame in pic

        vp = vp[2::-1]
        rot_matrix = rotation_matrix_from_vectors(vp[0], np.array([0,0,1]))
        img_inv = rot(FRAME, R=rot_matrix)
        # img_inv = rot(FRAME, vp)

        path_inv_img = OUTPATH_STR + '/' + FILENAME
        Image.fromarray((img_inv * 255).astype(np.uint8)).save(path_inv_img, quality=100)


##產生校正前影片與校正後影片的對比影片
##應該可平行化但尚未實行
def PRODUCE_COMPARE_VIDEO(ori_img_folder, image_folder, video_name):
    print("[PRODUCE_COMPARE_VIDEO]")
    fps=30
    ori_images = [img for img in os.listdir(ori_img_folder) if img.endswith(".jpg")]
    ori_images = sorted(ori_images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for i in range(len(images)):
        img_ori = cv2.imread(os.path.join(ori_img_folder, ori_images[i]))
        img_fix = cv2.imread(os.path.join(image_folder, images[i]))
        img_ori = cv2.resize(img_ori, (width//2, height))
        img_fix = cv2.resize(img_fix, (width//2, height))

        image = np.concatenate((img_ori, img_fix), axis=1)
        video.write(image)
        
    cv2.destroyAllWindows()
    video.release()

def PRODUCE_VIDEO(image_folder, video_name):
    print("[PRODUCE_VIDEO]")
    fps=30
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for i in range(len(images)):
        img_fix = cv2.imread(os.path.join(image_folder, images[i]))
        img_fix = cv2.resize(img_fix, (width, height))
        video.write(img_fix)
        
    cv2.destroyAllWindows()
    video.release()

def PERFORME_VIDEO(test_file_name, INFOLIST=None):
    print("[PERFORME_VIDEO]")
    #setting parameter
    ROUND = [3] 
    QUALITY = [7]
    XYZ_VOTE_Q1 = [27]
    XYZ_VOTE_Q2 = [370]
    XYZ_VOTE_DIFF = [0.007]
    NORM_DIFF = [0.01]
    PARAMETERS = {'ROUND':ROUND, 'QUALITY':QUALITY, 'XYZ_VOTE_Q1':XYZ_VOTE_Q1, 'XYZ_VOTE_Q2':XYZ_VOTE_Q2, 'XYZ_VOTE_DIFF':XYZ_VOTE_DIFF, 'NORM_DIFF':NORM_DIFF}

    #call model
    testPrallel_online(test_file_name, PARAMETERS, INFOLIST)

# video ==> pic in './videoframes/*.jpg' ==> alignmented_pis & infolist in './test_out' 
# ==> alignmented_video in ./
def video_alignment(video_path, video_start, frameNum, output_video_name = 'alignmented_video.mp4', INFOLIST=None):
    # video_path = 'room.mp4'
    # video_start = 0
    # frameNum = 100
    frame2pic(video_path, video_start, frameNum, DELAY=1)

    test_file_name = './videoframes/*.jpg'
    # infolist_name = './test_out/imginfolist_per5_0to1000.npy'
    # INFOLIST = np.load(infolist_name, allow_pickle=True)
    PERFORME_VIDEO(test_file_name, INFOLIST=INFOLIST)

    image_folder = './test_out'
    video_name = output_video_name
    # PRODUCE_COMPARE_VIDEO(ori_img_folder, image_folder, video_name)
    PRODUCE_VIDEO(image_folder, video_name)


def img_alignment_compare(image_folder, output_folder='./img_comparision'):
    OUTPATH = pathlib.Path(output_folder)
    OUTPATH.mkdir(parents=True, exist_ok=True)


    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images = sorted(images, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    id=0
    for image in images:
        img_ori = cv2.imread(image_folder + '/' + image)
        width, height = (img_ori.shape[1], img_ori.shape[0])
        img_ori = cv2.resize(img_ori, (width, height))

        rot_str, rotation_rnd = random_rotation_matrix()
        img_rnd = rot(img_ori, rotation_rnd).astype(np.uint8)
        vp_rnd, score_rnd = find_vp_and_score(img_rnd)
        img_rnd_inv = (rot(img_rnd, vp_rnd)).astype(np.uint8)


        cv2.imwrite(output_folder + "/%d_ori.jpg" % id, img_ori)
        rnd = img_concat(img_rnd, img_rnd_inv)
        cv2.imwrite(output_folder + "/%d_" % id + rot_str + "_and_inv.jpg", rnd)

        # cv2.imwrite(output_folder + "/%d_roll20.jpg" % id, img_roll20)
        # cv2.imwrite(output_folder + "/%d_pitch20.jpg" % id, img_pitch20)
        # cv2.imwrite(output_folder + "/%d_roll20_inv.jpg" % id, img_roll20_inv)
        # cv2.imwrite(output_folder + "/%d_pitch20_inv.jpg" % id, img_pitch20_inv)
        id = id +1

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
        except:
            vp_new_score = -1

        if(vp_new_score > ONEPARA_MAX_SCORE):
            ONEPARA_MAX_VP = vp_new
            ONEPARA_MAX_SCORE = vp_new_score

    return ONEPARA_MAX_VP, ONEPARA_MAX_SCORE



if __name__ == '__main__':
    # video_path = 'room.mp4'
    # video_path = 'random_shaking_video.mp4'
    # video_start = 0
    # frameNum = 200
    # video_alignment(video_path, video_start, frameNum)

    # video_path = 'room.mp4'
    # video_start = 0
    # frameNum = 200
    # frame2pic(video_path, video_start, frameNum, DELAY=1)

    # img_folder = '/home/johnlai/360ved_ali/videoframes'
    # video_name = 'random_shaking_video.mp4'
    # random_shaking_video(img_folder,video_name)
    
    # image_folder = '/home/johnlai/FuSta/FuSta_out'
    # video_name = 'Fusta_alignmented.mp4'
    # PRODUCE_VIDEO(image_folder, video_name)

    # image_folder = '/home/johnlai/pano_lsd_align_test/pano_lsd_align_test/test_img'
    # img_alignment_compare(image_folder)

    
