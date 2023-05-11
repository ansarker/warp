import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt

# shirt = cv2.imread('images/src.jpeg')
shirt = cv2.imread('images/shirt.png')
shirt = cv2.resize(shirt, (384, 512))
shirt = cv2.cvtColor(shirt, cv2.COLOR_BGR2RGB)
# _, shirt = cv2.threshold(shirt, 0, 127, cv2.THRESH_BINARY)

# person = cv2.imread('images/dst.jpg')
person = cv2.imread('images/person.jpg')
person = cv2.resize(person, (384, 512))
person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
# _, person = cv2.threshold(person, 0, 127, cv2.THRESH_BINARY)

# Detect and compute keypoints
sift = cv2.SIFT_create(edgeThreshold=7)
kp_shirt, desc_shirt = sift.detectAndCompute(shirt, None)
kp_person, desc_person = sift.detectAndCompute(person, None)

kp_pe = cv2.drawKeypoints(person, kp_person, None)
kp_sh = cv2.drawKeypoints(shirt, kp_shirt, None)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc_shirt, desc_person)
matches = sorted(matches, key=lambda x: x.distance)

matched_ = cv2.drawMatches(shirt, kp_shirt, person, kp_person, matches, None)

# Estimate transformation
src_pts = np.float32([kp_shirt[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_person[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

src_pts = np.squeeze(src_pts)
dst_pts = np.squeeze(dst_pts)

"""
    Warp algorithm 1
"""
# Estimate affine transformation matrix using RANSAC
M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

# Warp reference image to fit target image
warped_shirt = cv2.warpAffine(shirt, M, (person.shape[1], person.shape[0]))

person_mask = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
# person_mask = cv2.threshold(person_mask, 0, 127, cv2.THRESH_BINARY)[1]
# person_mask[person_mask > 20] = 255

warped_mask = cv2.cvtColor(warped_shirt, cv2.COLOR_BGR2GRAY)
# warped_mask = cv2.threshold(warped_mask, 0, 127, cv2.THRESH_BINARY)[1]
# warped_mask[warped_mask > 20] = 255

MAX_ITERATIONS = 100000
EPSILON = 1e-10

warp_matrix = np.eye(2, 3, dtype=np.float32)
cc, warp_matrix = cv2.findTransformECC(warped_mask, person_mask, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, MAX_ITERATIONS, EPSILON))

warp_matrix = np.vstack([warp_matrix, [0, 0, 1]])

# Apply the transformation to mask2 using the warpPerspective function
aligned_mask = cv2.warpPerspective(warped_mask, warp_matrix, (person_mask.shape[1], person_mask.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

cv2.imwrite('output/person.png', person)
cv2.imwrite('output/warped_shirt.png', warped_shirt)
cv2.imwrite('output/aligned_mask.png', aligned_mask)

fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(person)
axs[0, 0].set_title('person')
axs[0, 1].imshow(person_mask, plt.cm.gray)
axs[0, 1].set_title('person mask')
axs[1, 0].imshow(warped_shirt)
axs[1, 0].set_title('Warped')
axs[1, 1].imshow(warped_mask, plt.cm.gray)
axs[1, 1].set_title('Warped mask')

axs[0, 2].imshow(aligned_mask, plt.cm.gray)
axs[0, 2].set_title('aligned_mask ')
plt.show()

# cv2.imshow('dst', kp_pe)
# cv2.imshow('src', kp_sh)
# cv2.imshow('matched', matched_)
# cv2.imshow("warped", warped_shirt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()