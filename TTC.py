# -*- coding: utf-8 -*-
"""
@Project: TTC
@File   : TTC.py
@Author : Zhang P.H
@Date   : 2023/1/27
@Desc   :
"""
import math
import cv2
import numpy as np

# 加载图像
img_T = cv2.imread("img_T.jpg", cv2.IMREAD_GRAYSCALE)
img_I = cv2.imread("img_I.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("demo.jpg", cv2.IMREAD_COLOR)

box_T = np.array([100, 100, 150, 200]).astype(np.int)
box_I = np.array([360, 250, 390, 330]).astype(np.int)
box_T_center = np.array([box_T[0] + box_T[2], box_T[1] + box_T[3]]) // 2

cv2.rectangle(img, (box_T[1], box_T[0]), (box_T[3], box_T[2]), color=(0, 255, 0), thickness=1)
cv2.rectangle(img, (box_I[1], box_I[0]), (box_I[3], box_I[2]), color=(0, 0, 255), thickness=1)
# cv2.imshow("", img)
# cv2.waitKey(-1)


# 求解T的梯度
img_T_Gu = (img_T[:, 2:] - img_T[:, 0:-2]) / 2
img_T_Gv = (img_T[2:, :] - img_T[0:-2, :]) / 2
img_T_Gu = np.pad(img_T_Gu, ((0, 0), (1, 1)), 'constant', constant_values=255)
img_T_Gv = np.pad(img_T_Gv, ((1, 1), (0, 0)), 'constant', constant_values=255)
# cv2.imshow("", np.hstack([img_T_Gu, img_T_Gv]))
# cv2.waitKey(-1)

# 随机采样
points_idx = np.arange((box_T[3] - box_T[1]) * (box_T[2] - box_T[0]))
points_idx_select = np.random.choice(points_idx, 200, replace=False)

# 求解Jacobi和Hessian矩阵
J = np.zeros((3, 200))
H = np.zeros((3, 3))
for idx in range(200):
    idx_select = points_idx_select[idx]
    # print(idx, idx_select)
    v = idx_select // (box_T[2] - box_T[0]) + box_T[1]
    u = math.fmod(idx_select, (box_T[2] - box_T[0])) + box_T[0]
    v_c = v - box_T_center[1]
    u_c = u - box_T_center[0]
    J_tmp = np.array([img_T_Gu[int(v), int(u)], img_T_Gv[int(v), int(u)], img_T_Gu[int(v), int(u)] * u_c + img_T_Gv[int(v), int(u)] * v_c]).T
    J[:, idx] = J_tmp
    H += J_tmp * J_tmp.T

# GN迭代
p = np.array([(box_I[0] + box_I[2]) - (box_T[0] + box_T[2]), (box_I[1] + box_I[3]) - (box_T[1] + box_T[3]), 0])
for it in range(20):
    err_sum = np.zeros((3, 1))
    for idx in range(200):
        idx_select = points_idx_select[idx]
        # print(idx, idx_select)
        v_T = idx_select // (box_T[2] - box_T[0]) + box_T[1]
        u_T = math.fmod(idx_select, (box_T[2] - box_T[0])) + box_T[0]
        u = (p[2] + 1) * u_T + p[0]
        v = (p[2] + 1) * v_T + p[1]
        err = ((int(u) + 1 - u) * (int(v) + 1 - v) * img_I[int(v), int(u)] +
               (u - int(u)) * (int(v) + 1 - v) * img_I[int(v), int(u) + 1] +
               (u - int(u)) * (v - int(v)) * img_I[int(v) + 1, int(u) + 1] +
               (int(u) + 1 - u) * (v - int(v)) * img_I[int(v) + 1, int(u)]) \
              - img_T[int(v_T), int(u_T)]
        err_sum += J[:, idx].reshape(3, 1) * err

    print(err_sum)
    delta_p = np.linalg.inv(H) @ err_sum
    p[0] -= delta_p[0] / delta_p[2]
    p[1] -= delta_p[1] / delta_p[2]
    p[2] += 1 / delta_p[2]
