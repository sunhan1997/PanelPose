"""
Create the RT which can full of the whole space []
Author: sunhan
"""
import numpy as np
import cv2
from lib.pysixd_stuff import view_sampler
import math
from lib.meshrenderer import meshrenderer_phong
import random
import os
import yaml
import glob



# ##############################################   config：start  ############################################
FOR_R = True
VIS = True
bbox_VIS = False
random_light = True
render_near = 0.1
render_far = 10000 # unit: should be mm
K = np.array(  [[621.399658203125, 0, 313.72052001953125],
              [0, 621.3997802734375, 239.97579956054688],
              [0, 0, 1]])
# K = np.array(  [[567.53720406, 0, 312.66570357], [0, 569.36175922, 257.1729701], [0, 0, 1]])
IM_W, IM_H = 640, 480
# ply_model_paths = [str('/home/robot/Downloads/sunhan/MP6D/models_cad/obj_01.ply')]
ply_model_paths = [str('/home/robot/6D_ws/6D_PanelPose2/work_space/CAD/panel1/panel1.ply')]
max_rel_offset = 0.2  # used change the abs bbox
# ##############################################   config：end  ############################################

Renderer = meshrenderer_phong.Renderer(ply_model_paths,samples=1,vertex_scale=float(1)) # float(1) for some models

# sunhan add : get the bbox from depth
def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]


# yaml_gt = os.path.join(  "/home/robot/6D_ws/EfficientPose/Linemod_preprocessed/data/00/gt.yml")
# yaml_info = os.path.join(  "/home/robot/6D_ws/EfficientPose/Linemod_preprocessed/data/00/info.yml")
# outpt_gt = yaml.load(open(yaml_gt), Loader=yaml.FullLoader)
# outpt_info = yaml.load(open(yaml_info), Loader=yaml.FullLoader)
#
# for i in range(1000):
#     RT = np.identity(4)
#     RT[:3, :3] = np.array(outpt_gt[i][0]['cam_R_m2c']).reshape(3, 3)
#     RT[0:3, 3] = np.array(outpt_gt[i][0]['cam_t_m2c'])
#     RT = RT[0:3, 0:4]
#     obj_bb = np.array(outpt_gt[i][0]['obj_bb']) + np.array([-5, -5, 10, 10])
#     intrinsic_matrix = np.array(outpt_info[i]['cam_K']).reshape(3, 3)  #### for TLESS
#
#     bg = cv2.imread('/home/robot/6D_ws/EfficientPose/Linemod_preprocessed/data/00/bg.png')
#     # depth = cv2.imread('/home/robot/6D_ws/EfficientPose/Linemod_preprocessed/data/00/depth/{}.png'.format(i),cv2.IMREAD_ANYDEPTH)
#
#     bgr, depth = Renderer.render(
#         obj_id=0,
#         W=IM_W,
#         H=IM_H,
#         K=intrinsic_matrix,
#         R=RT[:3, :3],
#         t=RT[0:3, 3],
#         near=render_near,
#         far=render_far,
#         random_light=random_light
#     )
#
#     mask = (depth > 1e-8).astype('uint8')
#     mask[mask>0] = 1
#
#     bg[mask>0]  =  bgr[mask>0]
#
#     cv2.imshow('bg', bg)
#     cv2.waitKey(1)
#     cv2.imwrite('/home/robot/6D_ws/EfficientPose/Linemod_preprocessed/data/00/rgb2/{}.png'.format(i),bg)

def angle2Rmat(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
#
# R1 = view_sampler.sample_views(500, radius=600,azimuth_range=(0,  2 * math.pi),elev_range=( 0, math.pi)    )
# R2 = view_sampler.sample_views(500, radius=600,azimuth_range=(0,  2 * math.pi),elev_range=( -math.pi, 0)    )
# ############################################  for vis my r   ########################################
# RT = []
# # t_all = np.array([[0,0,800],[0,0,700] , [0, 0,600], [0, 0,500],[0, 0,400],[0, 0,500] ])
# t_all = np.array([0,0,950])
# # for i in range(len(t_all)):
# for i in range(4):
#     for idx in range(len(R1[0]) + len(R2[0])):
#         if idx < len(R1[0]):
#             R = R1[0][idx]['R']
#             # t = t_all[i]
#             t = t_all - np.array([0,0,50])
#             print('R1 ------------------')
#         else:
#             R = R2[0][idx - len(R1[0])]['R']
#             # t = t_all[i]
#             t = t_all - np.array([0,0,50])
#             print('R1 +++++++++++++++++++++')
#
#         bgr, depth = Renderer.render(
#             obj_id=0,
#             W=IM_W,
#             H=IM_H,
#             K=K.copy(),
#             R=R,
#             t=t,
#             near=render_near,
#             far=render_far,
#             random_light=random_light
#         )
#         mask = (depth > 1e-8).astype('uint8')
#         show_msk = (mask / mask.max() * 255).astype("uint8")
#         cv2.imshow('bgr', bgr)
#         cv2.waitKey(1)
#
#         T = np.identity(4)
#         T[:3, : 3] = R
#         T[0:3, 3] = t.reshape(3)
#         RT.append(T)
#         print('R all is saved!:  {} '.format(idx))
# np.save('/home/robot/6D_ws/AAE_torch/tools/RT_4MP.npy', RT)


# ###############################  sampling for panelpose  #####################################
RT=[]
t_all = np.array([[0,0,450] ,[0,0,500], [0,0,400], [0,0,370],[0,0,350]])
bg_img_paths = glob.glob('/home/robot/Downloads/sunhan/SUN2012pascalformat/JPEGImages/*.jpg')
file_list = bg_img_paths[:200]

for i_t in range(len(t_all)):
    for idx in range(2000):
        theta_x = random.uniform(-math.pi / 8, math.pi / 8)
        theta_y = random.uniform(-math.pi / 8, math.pi / 8)
        theta_z = random.uniform(-math.pi/2, math.pi/2)
        x = random.uniform(-20, 20)
        y = random.uniform(-20, 20)
        if idx < 250:
            theta = np.array([0, 0, theta_z])
        else:
            theta = np.array([theta_x, theta_y, theta_z])
        R = angle2Rmat(theta)

        # if idx < 10:
        #     if idx < 4:
        #         theta = np.array([0, 0, theta_z])
        #     else:
        #         theta = np.array([theta_x, theta_y, theta_z])
        #     R = angle2Rmat(theta)
        # else:
        #     if idx < 15:
        #         theta = np.array([0, 0, theta_z])
        #     else:
        #         theta = np.array([theta_x, theta_y, theta_z])
        #     R = angle2Rmat(theta)
        #     theta_right = np.array([-math.pi, 0, 0])
        #     R_right = angle2Rmat(theta_right)
        #     R = np.dot(R_right,R)


        t = t_all[i_t] #+ np.array([x,y,300])

        bgr, depth = Renderer.render(
            obj_id=0,
            W=IM_W,
            H=IM_H,
            K=K.copy(),
            R=R,
            t=t,
            near=render_near,
            far=render_far,
            random_light=random_light
        )
        mask = (depth > 1e-8).astype('uint8')
        show_msk = (mask / mask.max() * 255).astype("uint8")


        # fname = file_list[idx]
        # bgr_bg = cv2.imread(fname)
        # bgr_bg = cv2.resize(bgr_bg, (640, 480))
        #
        # bgr_bg[show_msk>0]=bgr[show_msk>0]

        # cv2.imshow('bgr', bgr_bg)
        cv2.imshow('show_msk', show_msk)
        cv2.waitKey(1)
        # cv2.imwrite('/home/robot/6D_ws/6D_PanelPose/maskrcnn/data5/ori/{}.png'.format(idx),bgr_bg)
        # cv2.imwrite('/home/robot/6D_ws/6D_PanelPose/maskrcnn/data5/mask/{}.png'.format(idx),show_msk)

        T = np.identity(4)
        T[:3, : 3] = R
        T[0:3, 3] = t.reshape(3)
        RT.append(T)
        print('R all is saved!:  {} '.format(idx))


np.save('/home/robot/6D_ws/6D_PanelPose2/work_space/CAD/RT_panel_1w.npy', RT)
# np.save('/home/robot/6D_ws/AAE_torch/tools/RT_4omron.npy', RT)






















