# _*_ coding: utf-8 _*_
# @Time    :2022/2/16 17:23
# @Author  :LiuZhihao
# @File    :imgmatching_optimize.py


import numpy as np
import cv2
from matplotlib import pyplot as plt


def create_sift():
    # 创造sift
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(leftImage, None)
    kp2, des2 = sift.detectAndCompute(rightImage, None)  # 返回关键点信息和描述符
    # print(kp2, des2)
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # 指定索引树要被遍历的次数

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    print("matches", matches[0])
    for i, (m, n) in enumerate(matches):
        # if m.distance < 0.07 * n.distance:
        if m.distance < 0.7 * n.distance:  # 在多机多目标数据集上0.6会出现误识别，此参数可调！ default 0.7
            matchesMask[i] = [1, 0]
    return kp1, kp2, matches, matchesMask

def compute_matrics(matches, kp1, kp2):
    MIN_MATCH_COUNT = 5
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # 计算变换矩阵：
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        
        matchesMask = None
    return good, matchesMask, M

def compute_mapped_points(M, pts):
    h, w, channels = leftImage.shape
    # cv2.circle(leftImage, (int(0.530203*w), int(0.247687*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.6994767*w), int(0.343945*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.75781*w),int(0.3990441*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.80594*w),int(0.2731414*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.7552083333333334*w), int(0.7824074074074074*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.43828125*w),    int(0.7384259259259259*h)), 2, (255, 255, 0), 50)
    # cv2.circle(leftImage, (int(0.24557291666666667*w),    int(0.6689814814814815*h)), 2, (255, 255, 0), 50)
    cv2.circle(leftImage, (int(480), int(630)), 2, (255, 255, 0), 50)
    cv2.circle(leftImage, (int(40), int(670)), 2, (255, 255, 0), 50)

    dst = cv2.perspectiveTransform(pts, M)
    print(dst)
    for i in range(len(dst)):
        # print(dst[i][0][0])
        img2 = cv2.circle(rightImage, (int(dst[i][0][0]), int(dst[i][0][1])), 2, (0, 255, 255), 50)
    # img2 = cv2.polylines(rightImage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  # 在目标图片外的点会出错
    cv2.imshow("point", img2)
    cv2.waitKey()
    # plt.imshow(img2)
    # plt.show()

def draw_matchingpoints(kp1, kp2, good, matchesMask):
    # 绘制对应点连线
    drawParams = dict(matchColor=(0, 255, 0), singlePointColor=None,
                      matchesMask=matchesMask, flags=2)  # flag=2只画出匹配点，flag=0把所有的点都画出
    img3 = cv2.drawMatches(leftImage, kp1, rightImage, kp2, good, None, **drawParams)
    cv2.imshow("matching",img3)
    cv2.waitKey()
    cv2.imwrite("matching.jpg", img3)

def draw_fusing_images(leftImage, rightImage, M):
    result = cv2.warpPerspective(leftImage, M,
                                 (leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
    # 融合方法1
    result.astype(np.float32)
    result = result / 2
    result[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
    # result = result/result.max()*255
    # print(result)

    # cv2.imshow("fuse", result)  ######??????????????出错？？？
    # cv2.waitKey()
    cv2.imwrite("matching3.jpg", result)
    

if __name__ == "__main__":
    leftImage = cv2.imread('/home/chenguanlin/MDOT/Two-MDOT/test/two/md2008/md2008-1/img/00000001.jpg')
    rightImage = cv2.imread('/home/chenguanlin/MDOT/Two-MDOT/test/two/md2008/md2008-2/img/00000001.jpg')
    kp1, kp2, matches, matchesMask = create_sift()
    good, matchesMask, M = compute_matrics(matches, kp1, kp2)
    print(M)

    pts = np.array([
        [int(480), int(630)],
        [int(40), int(670)]
    ]).reshape(-1, 1, 2).astype(np.float32)

    compute_mapped_points(M, pts)

    draw_matchingpoints(kp1, kp2, good, matchesMask)

    draw_fusing_images(leftImage, rightImage, M)

