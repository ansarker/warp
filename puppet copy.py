from __future__ import print_function
import cv2
import numpy as np


def get_displacement_vector(point, img):
    x, y = point
    step_size = 10
    
    gx, gy = np.gradient(img)
    
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
    # clip indices to ensure they are within bounds of the image
    x_min = int(max(x - step_size, 0))
    x_max = int(min(x + step_size, img.shape[1] - 1))
    y_min = int(max(y - step_size, 0))
    y_max = int(min(y + step_size, img.shape[0] - 1))

    gx = gx[y_min:y_max, x_min:x_max]
    gy = gy[y_min:y_max, x_min:x_max]
        
    gradient_matrix = np.array([gx.flatten(), gy.flatten()]).T
    hessian_matrix = np.matmul(gradient_matrix.T, gradient_matrix)
    inv_hessian = np.linalg.inv(hessian_matrix)
    return (-1 * np.matmul(inv_hessian, gradient_matrix.T)).T

def update_control_points(ctrl_pts, img):
    for i in range(len(ctrl_pts)):
        point = ctrl_pts[i]
        displacement_vector = get_displacement_vector(point, img)
        ctrl_pts[i] = point + displacement_vector
    return ctrl_pts

img = cv2.imread("images/12.jpg",  cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("images/person.jpg", cv2.IMREAD_GRAYSCALE)

# Get the bounding box of nonzero pixels
pts = cv2.findNonZero(mask)
x, y, w, h = cv2.boundingRect(pts)

# Extract the region of interest from the original image using the bounding box
roi = img[y:y+h, x:x+w]

# Define the control points
num_points = 5
step_x = w / (num_points-1)
step_y = h / (num_points-1)
x_start = x
y_start = y

ctrl_pts = []
for i in range(num_points):
    x = int(x_start + i * step_x)
    for j in range(num_points):
        y = int(y_start + j * step_y)
        ctrl_pts.append([x, y])
ctrl_pts = np.array(ctrl_pts, dtype=np.float32)

num_iterations = 10
step_size = 10

for i in range(num_iterations):
    for j, point in enumerate(ctrl_pts):
        ctrl_pts[j] = np.clip(ctrl_pts[j], 0, np.array(img.shape[:2])[::-1] - 1)
    ctrl_pts = update_control_points(ctrl_pts, img)

x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
mesh = np.dstack((x, y)).astype(np.float32)

deformed_mesh = mesh + ctrl_pts
output = cv2.remap(img, deformed_mesh[:, :, 0], deformed_mesh[:, :, 1], cv2.INTER_LINEAR)

cv2.imshow('Puppet Warp Input', img)
cv2.imshow('Puppet Warp ROI', roi)
cv2.imshow('Puppet Warp Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
