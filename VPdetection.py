import cv2
import numpy as np
import sys
import random
import numba as nb
from numba import prange

def VPdetection(img, XYZ_VOTE_DIFF = 0.008, NORM_DIFF=0.05, XYZ_VOTE_Q1 = 10, XYZ_VOTE_Q2 = 150):
    ##PREPROCESSING
    if(img.shape[2] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    imgH = img.shape[0]
    imgW = img.shape[1]
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    binary=cv2.Canny(img, lowThresh, high_thresh)
    edgeset, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  #CHAIN_APPROX_TC89_KCOS 
    min_length_segment = imgH/20
    min_length_segment_vp = imgH/6

    ##FIND VP
    vpLines_normal, vpLines_LinePoints = edgeVote(edgeset, min_length_segment_vp, min_length_segment, imgH, imgW, XYZ_VOTE_DIFF, XYZ_VOTE_Q1, XYZ_VOTE_Q2)
    vp, best_vp_score = findVP(vpLines_normal, vpLines_LinePoints, NORM_DIFF)

        
    return vp, best_vp_score


##(3,) ==> scalar
def norm(v):
    s = v[0]**2+v[1]**2+v[2]**2
    return np.sqrt(s)

##(3,) cross (3,) ==> (3,)
def cross(a, b):
    c = np.array((a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]))
    return c

@nb.njit(fastmath=True)
def edgeVote(edgeset, min_length_segment_vp, min_length_segment, imgH, imgW, threshold1 , iter1, iter2):
    # threshold1 = 0.008, iter1 = 3, iter2 = 30
    
    vpLines_normal = []
    vpLines_LinePointsSum = []
    num_edges = len(edgeset)

    for i in range(num_edges):
        if(edgeset[i].shape[0] > min_length_segment_vp):
            theta = -(edgeset[i][:, 0, 1]-imgH/2)*np.pi/imgH
            phi = (edgeset[i][:,0, 0]-imgW/2)*2*np.pi/imgW

            xyz = np.zeros((edgeset[i].shape[0], 3))
            xyz[:, 0] = np.sin(phi) * np.cos(theta)
            xyz[:, 1] = np.cos(theta) * np.cos(phi)
            xyz[:, 2] = np.sin(theta)
            
            N=xyz.shape[0]
            
            for _ in range(iter1):
                if(xyz.shape[0] > N * 0.1):
                    best_inliers_num = 0
                    id = np.random.randint(0, xyz.shape[0]-1, iter2*2)

                    for k in range(0, 2*iter2, 2):
                        id0 = id[k]
                        id1 = id[k+1]

                        if(id0==id1):
                            continue

                        n = np.cross(xyz[id0, :], xyz[id1, :])
                        n = n / np.linalg.norm(n)

                        inliers = np.abs(n @ xyz.T) < threshold1 
                        inliers_num = np.count_nonzero(inliers)

                        if inliers_num > best_inliers_num:
                            best_inliers_num = inliers_num
                            bestInliers = inliers


                    if(best_inliers_num > 0):
                        bestOutliers = np.invert(bestInliers)

                        if(best_inliers_num > min_length_segment_vp):
                            u,s,vh = np.linalg.svd(xyz[bestInliers,:])
                            vpLines_normal.append(vh.T[:, 2])  ## (3,)
                            vpLines_LinePointsSum.append(best_inliers_num)

                        xyz = xyz[bestOutliers, :]
                        
                        if xyz.shape[0] < min_length_segment:
                            break
                else:
                    break

    return vpLines_normal, vpLines_LinePointsSum


def findVP(vpLines_normal, vpLines_LinePointsSum, NORMDIFF = 0.05):
    ##(3,) ==> scalar
    def norm(v):
        s = v[0]**2+v[1]**2+v[2]**2
        return np.sqrt(s)


    ##(3,) cross (3,) ==> (3,)
    def cross(a, b):
        c = np.array([a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]])
        return c

    ## vpLines: (N,3)
    iterations = 1000
    sum_OldPoints = 0

    if(len(vpLines_normal) < 3):
        vp = np.identity(3)
        return vp
   
    normals = np.array(vpLines_normal).T
    pointsSum = np.array(vpLines_LinePointsSum)
    for k in range(iterations):
        
        id = random.sample(range(0, normals.shape[1]), 3)
        n1 = normals[:,id[0]]
        n2 = normals[:,id[1]]
        n3 = normals[:,id[2]]
        v1 = cross(n1,n2)
        v2 = cross(v1,n3)
        v3 = cross(v1,v2)
        v1 = (v1/norm(v1)).reshape(1,3)
        v2 = (v2/norm(v2)).reshape(1,3)
        v3 = (v3/norm(v3)).reshape(1,3)

        v = np.concatenate((v1, v2, v3), axis=0)
        angle = np.abs(v@normals)  #(3,N)
        vpmin = np.min(angle, axis = 0) #(N,)
        
        sum_NewPoints = np.sum(pointsSum[vpmin < NORMDIFF])
            
        if sum_NewPoints > sum_OldPoints:
            sum_OldPoints = sum_NewPoints
            bestVPs = v


    best_vp_score = sum_OldPoints
    ids = np.argmax(np.abs(bestVPs),axis = 0)
    vp = bestVPs[ids, :]


    #axis selection
    vx = vp[0,:].reshape(1,3)
    vy = vp[1,:].reshape(1,3)
    vz = vp[2,:].reshape(1,3)
    vp_x = np.concatenate((vx,-vx), axis=0)
    vp_y = np.concatenate((vy,-vy), axis=0)
    vp_z = np.concatenate((vz,-vz), axis=0)

    idx = np.where(vp_x[:,0]>0)[0]
    idy = np.where(vp_y[:,1]>0)[0]
    idz = np.where(vp_z[:,2]>0)[0]

    vp = np.concatenate((vp_x[idx,:],vp_y[idy,:],vp_z[idz,:]), axis=0)

    return vp, best_vp_score


def rotatePanorama(img, vp=None, R=None):
    '''
    Rotate panorama
        if R is given, vp (vanishing point) will be overlooked
        otherwise R is computed from vp
    '''
    
    Px, Py = rotateUtil(img.shape, vp, R)
    
    ## boundary
    sphereH, sphereW, C = img.shape
    imgNew = np.zeros((sphereH+2, sphereW+2, C), np.float64)
    imgNew[1:-1, 1:-1, :] = img
    imgNew[1:-1, 0, :] = img[:, -1, :]
    imgNew[1:-1, -1, :] = img[:, 0, :]
    imgNew[0, 1:sphereW//2+1, :] = img[0, sphereW-1:sphereW//2-1:-1, :]
    imgNew[0, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[-1, 1:sphereW//2+1, :] = img[-1, sphereW-1:sphereW//2-1:-1, :]
    imgNew[-1, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[0, 0, :] = img[0, 0, :]
    imgNew[-1, -1, :] = img[-1, -1, :]
    imgNew[0, -1, :] = img[0, -1, :]
    imgNew[-1, 0, :] = img[-1, 0, :]

    rotImg = warpImageFast(imgNew, Px+1, Py+1)

    return rotImg


def rotateUtil(shape, vp=None, R=None):

    sphereH, sphereW, C = shape
    
    TX = np.zeros((sphereH*sphereW, ))
    TY = np.zeros((sphereH*sphereW, ))
    y = np.arange(1, sphereH+1) 
    for i in prange(sphereW):
        TX[i*sphereH:(i+1)*sphereH] = i+1
        TY[i*sphereH:(i+1)*sphereH] = y


    ANGx = (TX - sphereW/2 - 0.5) / sphereW * 3.14159 * 2
    ANGy = -(TY - sphereH/2 - 0.5) / sphereH * 3.14159
    xyzNew = np.zeros((ANGx.shape[0], 3))
    xyzNew[:, 0] = np.cos(ANGy) * np.sin(ANGx)
    xyzNew[:, 1] = np.cos(ANGy) * np.cos(ANGx)
    xyzNew[:, 2] = np.sin(ANGy)


    if R is None:
        # R = np.linalg.inv(vp.T)
        R = np.asfortranarray(np.linalg.inv(vp.T))
    else:
        R = np.asfortranarray(R)
    xyzOld = np.linalg.solve(R, xyzNew.T).T


    x = xyzOld[:, 0]
    y = xyzOld[:, 1]
    z = xyzOld[:, 2]

    normXY = np.sqrt(x ** 2 + y ** 2)
    normXY[normXY < 0.000001] = 0.000001
    normXYZ = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    v = np.arcsin(z / normXYZ)
    u = np.arcsin(x / normXY)

    valid = (y < 0) & (u >= 0)
    u[valid] = 3.14159 - u[valid]
    valid = (y < 0) & (u <= 0)
    u[valid] = -3.14159 - u[valid]
    u[np.isnan(u[:])] = 0
    v[np.isnan(v[:])] = 0

    Px = (u + 3.14159) / (2*3.14159) * sphereW + 0.5
    Py = (-v + 3.14159/2) / 3.14159 * sphereH + 0.5
    Px_re = np.zeros((sphereH, sphereW))
    Py_re = np.zeros((sphereH, sphereW))
    for i in prange(sphereW):
        Px_re[:, i] = Px[i*sphereH:(i+1)*sphereH]
        Py_re[:, i] = Py[i*sphereH:(i+1)*sphereH]
    
    Px = Px_re
    Py = Py_re

    return Px, Py


def warpImageFast(im, XXdense, YYdense):
    minX = max(1., np.floor(XXdense.min()) - 1)
    minY = max(1., np.floor(YYdense.min()) - 1)

    maxX = min(im.shape[1], np.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], np.ceil(YYdense.max()) + 1)

    im = im[int(round(minY-1)):int(round(maxY)),
            int(round(minX-1)):int(round(maxX))]

    map_x = (XXdense - minX).astype('float32')
    map_y = (YYdense - minY).astype('float32')
    im_warp = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR)

    return im_warp