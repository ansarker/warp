import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# draw the body keypoint and lims
def draw_bodypose(img, poses):
    stickwidth = 5
    njoint = 25
    limbSeq = [[1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], 
               [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [0, 15], [0, 16], [15, 17], 
               [16, 18], [11, 24], [11, 22], [14, 21], [14, 19], [22, 23], [19, 20]]
    colors = [
        [255, 0, 85],  # 0 neck
        [255, 85, 0],  # 1 right shoulder
        [255, 170, 0],  # 2 right arm
        [255, 255, 0],  # right hand
        [170, 255, 0],  # 4 left shoulder
        [85, 255, 0],  # 5 left arm
        [0, 255, 0],  # 6 left hand
        [255, 0, 0],  # 7 backbone
        [0, 255, 85],  # 8 right pelvic
        [0, 255, 170],  # 9 right thigh
        [0, 170, 255],
        [0, 170, 255],  # 11 left pelvic
        [0, 85, 255],  # 12 left thigh
        [85, 0, 255],
        [255, 0, 170],  # 14 right eye
        [170, 0, 255],  # 15 left eye
        [255, 0, 255],  # 16 right ear
        [85, 0, 255],  # 17 left ear
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 255, 255],
        [170, 255, 255],
        [85, 255, 255],
        [0, 255, 255],
    ]

    for i in range(njoint):
        for n in range(len(poses)):
            pose = poses[n][i]
            if pose[2] <= 0:
                continue
            x, y = pose[:2]
            cv2.circle(img, (int(x), int(y)), 3, colors[i], thickness=-1)

    for pose in poses:
        for limb, color in zip(limbSeq, colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <= 0 or p2[2] <= 0:
                continue
            cur_canvas = img.copy()
            X = [p1[1], p2[1]]
            Y = [p1[0], p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_bodypose(im, pose_data):
    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

        # parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        # agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        
    return agnostic

"""
    Hooke's Law
"""
def hookes_spring(mesh, points, img):
    # Set up initial velocities
    VEL = np.zeros_like(mesh)

    # Set up material properties
    K = 0.6  # spring constant
    D = 0.1  # damping constant

    # Set up simulation parameters
    DT = 0.1  # time step
    NUM_STEPS = 1000  # number of simulation steps

    # Run simulation
    for i in range(NUM_STEPS):
        # Compute forces on each vertex
        forces = np.zeros_like(mesh, dtype=np.float64)
        for j in range(mesh.shape[0]):
            for k in range(3):
                # Compute spring force on current vertex
                curr_pos = mesh[j, k]
                next_pos = mesh[j, (k+1)%3]
                spring_vec = next_pos - curr_pos
                spring_length = np.linalg.norm(spring_vec)
                spring_dir = spring_vec / spring_length
                spring_force = K * (spring_length - 100)
                
                # Compute damping force on current vertex
                damping_force = -D * VEL[j, k]
                
                # Compute total force on current vertex
                total_force = (spring_force + damping_force) * spring_dir
                
                # Add force to current vertex
                forces[j, k] += total_force
        
        # Update velocities and positions
        for j in range(mesh.shape[0]):
            for k in range(3):
                # Apply forces to current vertex
                force = forces[j, k]
                VEL[j, k] += force * DT
                mesh[j, k] += VEL[j, k] * DT
                
                # Constrain key points
                if j == 0 and k in (0, 1):
                    mesh[j, k] = points[k]
                elif j == 2 and k == 1:
                    mesh[j, k] = points[2]
        
        img_out = img.copy()
        for j in range(mesh.shape[0]):
            for k in range(3):
                cv2.circle(img_out, tuple(mesh[j, k].astype(np.int32)), 3, (255, 0, 0), -1)
        cv2.imshow('image', img_out)
        cv2.waitKey(10)
        
    #     # Visualize current mesh
    #     plt.clf()
    #     triangulation = Triangulation(mesh[:, :, 0].flatten(), mesh[:, :, 1].flatten())
    #     plt.triplot(triangulation, lw=0.5, color='black')
    #     # plt.tricontourf(triangulation, mesh[:, :, 1].flatten(), cmap='coolwarm', alpha=0.8)
    #     plt.scatter(points[:, 0], points[:, 1], c='r')
    #     plt.imshow(img)
    #     # plt.xlim(0, 800)
    #     # plt.ylim(0, 800)
    #     plt.draw()
    #     plt.pause(0.01)

    # plt.show()

if __name__ == "__main__":
    KEYS = np.array([(200, 200), (400, 200), (300, 400), (100, 100), (200, 100), (300, 250)], dtype=np.float32)

    # Set up initial mesh
    MESH = np.array([[(100, 100), (200, 100), (150, 200)],
                    [(200, 100), (300, 100), (250, 200)],
                    [(150, 200), (250, 200), (200, 300)],
                    [(250, 200), (350, 200), (300, 300)],
                    [(200, 300), (300, 300), (250, 400)]], dtype=np.float64)

    hookes_spring(mesh=MESH, points=KEYS)