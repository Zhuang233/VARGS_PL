import open3d as o3d
import numpy as np
from pointnet2_ops import pointnet2_utils

def save_point_cloud_as_ply(points, filename="point_cloud.ply"):
    """
    将点云数据保存为 .ply 文件
    :param points: 形状为 [N, 3] 的 NumPy 数组，每一行代表 (x, y, z) 坐标
    :param filename: 输出的 .ply 文件名
    """
    if not isinstance(points, np.ndarray):
        raise TypeError("points 应该是 NumPy 数组")
    if points.shape[1] != 3:
        raise ValueError("points 的形状必须为 [N, 3]，表示 (x, y, z) 坐标")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data