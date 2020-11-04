# import open3d as o3d
from im2mesh.utils.libkdtree import KDTree
import numpy as np
import typing

CUBE_SIDE_LEN = 1.0

threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100, CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20, CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]

def distance_p2p(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''

    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    return dist

def calculate_fscore(gt, pr, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = distance_p2p(gt, pr)
    d2 = distance_p2p(pr, gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

def compute_fscore(gt, pr):
    ''' Compute F-Score.

    Args:
        gt (numpy array): ground truth points array
        pr (numpy array): prediction points array

    '''
    # import pudb; pu.db

    # gt_pcd = o3d.geometry.PointCloud()
    # gt_pcd.points = o3d.utility.Vector3dVector(gt)

    # pr_pcd = o3d.geometry.PointCloud()
    # pr_pcd.points = o3d.utility.Vector3dVector(pr)
    
    # for th in threshold_list:
    #     f, p, r, d1, d2 = calculate_fscore(gt, pr, th=th)

    th = CUBE_SIDE_LEN = 1.0 / 100
    f, p, r= calculate_fscore(gt, pr, th=th)

    return f, p, r