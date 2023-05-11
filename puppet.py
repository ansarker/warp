from __future__ import print_function
import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from PIL import Image
from utils import get_bodypose, hookes_spring
import json

img = cv2.imread("images/12.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.imread("images/person.jpg", cv2.IMREAD_GRAYSCALE)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

pose_rendered = cv2.imread("images/12_rendered.png")
pose_rendered = cv2.cvtColor(pose_rendered, cv2.COLOR_BGR2RGB)

cloth = np.zeros_like(img)
cloth[mask > 10] = img[mask > 10]
cloth_gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)

# extract the contour of the largest component
contours, hierarchy = cv2.findContours(cloth_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = contours[0]

triangle_mask = img.copy()

x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
step_size = 20
x, y = np.meshgrid(np.arange(x_min, x_min+x_max, step_size), np.arange(y_min, y_min+y_max, step_size))
points = np.vstack((x.ravel(), y.ravel())).T

# Generate triangles using Delaunay triangulation
tri = cv2.Subdiv2D((x_min, y_min, x_min+x_max, y_min+y_max))
for p in points:
    tri.insert(tuple(map(int, p)))

# Filter out triangles that are outside the contour
triangles = np.array(tri.getTriangleList())
new_triangles = []
for t in triangles:
    pts = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]])
    if np.all(cv2.pointPolygonTest(contour, tuple(map(int, pts[0])), False) >= 0):
        new_triangles.append(pts)
triangles = np.array(new_triangles)

for t in triangles:
    t = t.flatten()
    pt1 = (int(t[0]), int(t[1]))
    pt2 = (int(t[2]), int(t[3]))
    pt3 = (int(t[4]), int(t[5]))

    cv2.line(triangle_mask, pt1, pt2, (0, 210, 125), 1)
    cv2.line(triangle_mask, pt2, pt3, (0, 210, 125), 1)
    cv2.line(triangle_mask, pt3, pt1, (0, 210, 125), 1)

cv2.drawContours(triangle_mask, [contour], -1, (255, 0, 0), thickness=2)

"""
    Add keypoints onto the person body
"""
posemap = triangle_mask.copy()
with open("./images/12_keypoints.json") as f:
    pose_label = json.load(f)
    pose_data = pose_label['people'][0]['pose_keypoints_2d']
    pose_data = np.array(pose_data).astype(np.int32)
    pose_data = pose_data.reshape((-1, 3))[:, :2]
    # img_points = get_bodypose(Image.fromarray(posemap), pose_data)
    for pose in pose_data:
        center = tuple(pose)
        posemap = cv2.circle(posemap, center, 5, (0, 127, 255), 3)

keys = [1, 2, 3, 4, 5, 6, 8, 9, 12]
pose_data = pose_data[keys]

hookes_spring(mesh=np.float64(triangles), points=np.float32(pose_data), img=cloth)

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# axs[0, 0].imshow(img, plt.cm.gray)
# axs[0, 0].set_title('person')
# axs[0, 1].imshow(mask, plt.cm.gray)
# axs[0, 1].set_title('mask')
# axs[0, 2].imshow(cloth, plt.cm.gray)
# axs[0, 2].set_title('cloth')
# axs[1, 0].imshow(pose_rendered, plt.cm.gray)
# axs[1, 0].set_title('pose_rendered')
# axs[1, 1].imshow(triangle_mask, plt.cm.gray)
# axs[1, 1].set_title('triangle_mask')
# axs[1, 2].imshow(posemap, plt.cm.gray)
# axs[1, 2].set_title('points')
# plt.show()

# cv2.imshow("psoemap", posemap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()