"""
1.生成一张模拟图，并写上真值
2.在图上做TTC
"""
import os
import sys
import cv2
import numpy as np

scale = 0.7
u_offset = 300
v_offset = 200

box_T = np.array([100, 100, 150, 200])
box_I = box_T * scale
box_I[0::2] = box_I[0::2] + u_offset
box_I[1::2] = box_I[1::2] + v_offset
box_T = box_T.astype(np.int)
box_I = box_I.astype(np.int)
print(box_T, box_I)

img = np.zeros((480, 960), np.uint8) + 150

img_T = np.copy(img)
img_T[box_T[0]:box_T[2], box_T[1]:box_T[3]] = 50  # 光度不变
img_T = cv2.GaussianBlur(img_T, (9, 9), 0)
cv2.imwrite("img_T.jpg", img_T)

img_I = np.copy(img)
img_I[box_I[0]:box_I[2], box_I[1]:box_I[3]] = 50  # 光度不变
img_I = cv2.GaussianBlur(img_I, (9, 9), 0)
cv2.imwrite("img_I.jpg", img_I)

img[box_T[0]:box_T[2], box_T[1]:box_T[3]] = 50  # 光度不变
img[box_I[0]:box_I[2], box_I[1]:box_I[3]] = 50  # 光度不变
img = cv2.GaussianBlur(img, (9, 9), 0)
cv2.putText(img, "GT scale:{} u_offset:{} v_offset:{}".format(scale, u_offset, v_offset), (600, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=10)
cv2.putText(img, "GT T {}-{}-{}-{}".format(box_T[0], box_T[1], box_T[2],box_T[3]), (600, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=10)
cv2.putText(img, "GT I {}-{}-{}-{}".format(box_I[0], box_I[1], box_I[2],box_I[3]), (600, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=10)
cv2.imwrite("demo.jpg", img)
