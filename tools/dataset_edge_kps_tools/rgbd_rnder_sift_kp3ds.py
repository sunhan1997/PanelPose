#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import ctypes as ct
import pickle as pkl

import random
from random import randint
from random import shuffle
from tqdm import tqdm
from scipy import stats
from glob import glob
from cv2 import imshow, waitKey
from utils import ImgPcldUtils, MeshUtils, PoseUtils, SysUtils
from lib.meshrenderer import meshrenderer_phong
from lib.pysixd.misc import rgbd_to_point_cloud



SO_P = './raster_triangle/rastertriangle_so.so'
RENDERER = np.ctypeslib.load_library(SO_P, '.')

mesh_utils = MeshUtils()
img_pcld_utils = ImgPcldUtils()
pose_utils = PoseUtils()
sys_utils = SysUtils()


def load_mesh_c(mdl_p, scale2m):
    if 'ply' in mdl_p:
        meshc = mesh_utils.load_ply_model(mdl_p, scale2m=scale2m)
    meshc['face'] = np.require(meshc['face'], 'int32', 'C')
    meshc['r'] = np.require(np.array(meshc['r']), 'float32', 'C')
    meshc['g'] = np.require(np.array(meshc['g']), 'float32', 'C')
    meshc['b'] = np.require(np.array(meshc['b']), 'float32', 'C')
    return meshc

def fld(gray):
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(gray)

    show = np.zeros([960, 1280])

    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(show, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
    return show


def gen_one_zbuf_render(args, RT, Renderer):
    if args.extractor == 'SIFT':
        extractor = cv2.xfeatures2d.SIFT_create()
    if args.extractor == 'ORB':  # use orb
        extractor = cv2.ORB_create()

    h, w = args.h, args.w

    K = np.array(args.K).reshape(3, 3)
    # R, T = RT[:3, :3], RT[:3, 3]
    RT[:3, 3] = np.array([0,0,400])
    R, T = RT[:3, :3], np.array([0,0,400])

    bgr, depth = Renderer.render(
        obj_id=0,
        W=w,
        H=h,
        K=K.copy(),
        R=R,
        t=T,
        near=0.1,
        far=10000,
        random_light=False
    )

    msk = (depth > 1e-8).astype('uint8')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.vis:
        imshow("bgr", bgr.astype("uint8"))
        show_zbuf = depth.copy()
        min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
        show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
        show_zbuf = show_zbuf.astype(np.uint8)
        imshow("dpt", show_zbuf)
        show_msk = (msk / msk.max() * 255).astype("uint8")
        imshow("msk", show_msk)
        waitKey(0)

    data = {}
    data['depth'] = depth
    data['rgb'] = rgb
    data['mask'] = msk
    data['K'] = K
    data['RT'] = RT
    data['cls_typ'] = args.obj_name
    data['rnd_typ'] = 'render'

    masks_fe = msk.copy()
    masks_fe[masks_fe>0] = 255
    masks_fe = np.expand_dims(masks_fe, axis=2)
    img_add = np.concatenate([masks_fe, masks_fe], 2)
    img_add = np.concatenate([img_add, masks_fe], 2)

    if args.extractor == 'FLD':
        img_gray = cv2.cvtColor(img_add.astype("uint8"), cv2.COLOR_BGR2GRAY)
        img_fld = fld(img_gray)
        coord_2d = np.nonzero(img_fld > 0)
        coord_2d = np.concatenate(
            (coord_2d[0].reshape(coord_2d[0].shape[0], 1), coord_2d[1].reshape(coord_2d[1].shape[0], 1)), 1)
        kp_idxs = (coord_2d[:, 0], coord_2d[:, 1])

    if args.extractor == 'EDGE':
        img_gray = cv2.cvtColor(img_add.astype("uint8"), cv2.COLOR_BGR2GRAY)
        img_edge = cv2.Canny(img_gray, 50, 200)
        coord_2d = np.nonzero(img_edge > 0)
        coord_2d = np.concatenate(
            (coord_2d[0].reshape(coord_2d[0].shape[0], 1), coord_2d[1].reshape(coord_2d[1].shape[0], 1)), 1)
        kp_idxs = (coord_2d[:, 0], coord_2d[:, 1])

    if args.vis:
        for i in range(len(coord_2d[:, 0])):
            cv2.circle(rgb, (coord_2d[i, 1], coord_2d[i, 0]), 1, (0, 0, 255))
        cv2.imshow('EDGE rgb', rgb)
        cv2.waitKey(1)

    mask_fordpt = np.zeros((depth.shape[0], depth.shape[1]))
    mask_fordpt[coord_2d[:, 0],coord_2d[:, 1]] = 1
    depth = depth * mask_fordpt

    depth_show = depth.copy()
    depth_show[depth_show> 0] = 255
    cv2.imshow('depth_show',depth_show)
    cv2.waitKey(1)

    kp_xyz = img_pcld_utils.dpt_2_cld(depth, 1, K)

    # filter by dpt (pcld)
    kp_xyz, msk = img_pcld_utils.filter_pcld(kp_xyz)
    # kps = [kp for kp, valid in zip(kps, msk) if valid]  # kps[msk]
    # des = des[msk, :]

    # 6D pose of object in cv camer coordinate system
    # transform to object coordinate system
    kp_xyz = (kp_xyz - RT[:3, 3]).dot(RT[:3, :3])

    data['kp_xyz'] = kp_xyz

    return data


def extract_textured_kp3ds(args, mesh_pth, sv_kp=False):
    Renderer = meshrenderer_phong.Renderer([mesh_pth], samples=1,
                                           vertex_scale=float(1))  # float(1) for some models
    poses = np.load('/home/robot/6D_ws/AAE_torch/tools/RT_4panel.npy')
    kp3ds = []
    for pose in poses:
        # transform to object coordinate system
        # data = gen_one_zbuf_render(args, meshc, o2c_pose)
        data = gen_one_zbuf_render(args, pose, Renderer)
        kp3ds += list(data['kp_xyz'])
        # pclds += list(data['dpt_pcld'])

    if sv_kp:
        with open("%s_%s_textured_kp3ds.obj" % (args.obj_name, args.extractor), 'w') as of:
            for p3d in kp3ds:
                print('v ', p3d[0], p3d[1], p3d[2], file=of)
    return kp3ds


def test():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--obj_name", type=str, default="ape",
        help="Object name."
    )
    parser.add_argument(
        "--ply_pth", type=str, default="example_mesh/ape.ply",
        help="path to object ply."
    )
    parser.add_argument(
        '--debug', action="store_true",
        help="To show the generated images or not."
    )
    parser.add_argument(
        '--vis', action="store_true",
        help="visulaize generated images."
    )
    parser.add_argument(
        '--h', type=int, default=480,
        help="height of rendered RGBD images."
    )
    parser.add_argument(
        '--w', type=int, default=640,
        help="width of rendered RGBD images."
    )
    parser.add_argument(
        '--K', type=int, default=[700, 0, 320, 0, 700, 240, 0, 0, 1],
        help="camera intrinsix."
    )
    parser.add_argument(
        '--scale2m', type=float, default=1.0,
        help="scale to transform unit of object to be in meter."
    )
    parser.add_argument(
        '--n_longitude', type=int, default=3,
        help="number of longitude on sphere to sample."
    )
    parser.add_argument(
        '--n_latitude', type=int, default=3,
        help="number of latitude on sphere to sample."
    )
    parser.add_argument(
        '--extractor', type=str, default="ORB",
        help="2D keypoint extractor, SIFTO or ORB"
    )
    parser.add_argument(
        '--textured_3dkps_fd', type=str, default="textured_3D_keypoints",
        help="folder to store textured 3D keypoints."
    )
    args = parser.parse_args()
    args.K = np.array(args.K).reshape(3, 3)

    kp3ds = extract_textured_kp3ds(args, args.ply_pth)


if __name__ == "__main__":
    test()
# vim: ts=4 sw=4 sts=4 expandtab
