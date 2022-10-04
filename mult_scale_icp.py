import copy
import json
import os

import open3d as o3d
from sys import path
import png


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1.0, 0, 0])
    source_temp.transform(transformation)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp,axis])
    # o3d.visualization.draw_geometries([source_temp, target_temp])


def ICP_from_o3d(source,target,init_transformation):
    # draw_registration_result(source, target, init_transformation)
    #### ICP
    voxel_radius = [0.014, 0.01, 0.005, 0.001,0.00001]
    max_iter = [50, 30, 14, 150,200]
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.registration.registration_icp(
            source_down, target_down, 0.02, init_transformation,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=2000))

        init_transformation = result_icp.transformation

    # draw_registration_result(source, target, init_transformation)
    return  init_transformation



# source
