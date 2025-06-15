import os
import torch
import torch.utils.data as data
import numpy as np
from plyfile import PlyData
import math


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_attribute(vertex, attribute):
    # assert 'xyz' in attribute, 'At least need xyz attribute' can free this one actually
    # record the attribute and the index to read it
    attribute_index = {}
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data = np.stack((x, y, z), axis=-1)  # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        # opacity range from 0 to 1
        data = np.concatenate((data, opacity), axis=-1)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)

        scales = np.exp(scales)  # scale normalization

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)

    if "sh" in attribute:
        # get 3 dimension of sphere homrincals
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)

    return data

def normalize_to_range(arr, min_val, max_val, target_min, target_max):
    """将数组归一化到指定范围"""
    return (arr - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

def unnorm_gs(GS, centroid, scale_factor, scale_center, scale_magnitude, attribute=["xyz","opacity", "scale", "sh"]):
    """GS 反正则化
    Args:
        pc (_type_): 正则化后的GS集合, 中心，缩放值
        attribute (list, optional): GS包含的参数(xyz、opacity、scale、sh). Defaults to ["xyz"].

    Returns:
        反正则化GS
    """

    xyz = GS[..., :3]
    GS[..., :3] = xyz * scale_factor + centroid

    # if "opacity" in attribute:
    GS[..., 3] = normalize_to_range(GS[..., 3], max_val=1, min_val=-1, target_min=0, target_max=1)
    gs_scale = GS[..., 4:7]
    if "scale" in attribute:
        gs_scale = (gs_scale * scale_magnitude + scale_center)* scale_factor
        GS[..., 4:7] = gs_scale
        
    if "sh" in attribute:
        GS[..., 11:14] = GS[..., 11:14] * math.sqrt(3) / (2 * 0.28209479177387814)

    return GS

class ShapeNetGaussian(data.Dataset):
    def __init__(self, data_list_path, config):
        self.data_list_path = data_list_path

        self.gs_path = config["gs_path"]
        self.attribute = config["attribute"]
        self.norm_attribute = config["norm_attribute"]
        self.sample_points_num = config["sample_points_num"]

        self.file_list = []

        # print_log(
        #     f"[DATASET] Using Guassian Attribute {self.attribute}",
        #     logger="ShapeNetGS-55",
        # )
        # print_log(
        #     f"[DATASET] sample out {self.sample_points_num} points",
        #     logger="ShapeNetGS-55",
        # )
        # print_log(f"[DATASET] Open file {self.data_list_file}", logger="ShapeNetGS-55")

        with open(self.data_list_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                taxonomy_id = line.split("-")[0]
                model_id = line.split("-")[1].split(".")[0]
                self.file_list.append({"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line})
        # print_log(
        #     f"[DATASET] {len(self.file_list)} instances were loaded",
        #     logger="ShapeNetGS-55",
        # )

        # self.permutation = np.arange(self.npoints)

    def norm_gs(self, pc, attribute=["xyz"]):
        """GS 正则化
        Args:
            pc (_type_): GS集合
            attribute (list, optional): GS包含的参数(xyz、opacity、scale、sh). Defaults to ["xyz"].

        Returns:
            np.ndarray: 正则化后的 GS 集合
            np.ndarray: 中心
            float: 缩放值
        """
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz  -= centroid
        scale_factor = np.max(np.linalg.norm(pc_xyz, axis=1))  # 计算最大欧几里得距离
        pc_xyz /= scale_factor
        # inside a sphere
        pc[..., :3] = pc_xyz

        pc_scale = pc[..., 4:7]

        if "opacity" in attribute:
            # normalize to a -1 to 1 range
            pc[..., 3] = normalize_to_range(pc[..., 3], min_val=0, max_val=1, target_min=-1, target_max=1)

        if "scale" in attribute:
            # normalize to a -1 to 1 range
            pc_scale /= scale_factor  # normalize also the scale
            scale_center = np.mean(pc_scale, axis=0)
            pc_scale -= scale_center
            scale_magnitude = np.max(np.linalg.norm(pc_scale, axis=1))  # 计算最大范数
            pc_scale /= scale_magnitude
        else:
            scale_center = np.zeros(3)
            scale_magnitude = 1

        if "sh" in attribute:
            sh = pc[..., 11:14]
            sh = sh * 0.28209479177387814
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)
            pc[..., 11:14] = sh

        return pc, centroid, scale_factor, scale_center, scale_magnitude

    def __getitem__(self, idx):
        """__getitem__ function for ShapeNetGaussian
        加载一个模型的GS数据,正则化,随机采样sample_points_num个点.

        Returns:
            类别ID, 模型ID, GS数据, 中心, 缩放值
        """
        sample = self.file_list[idx]
        try:
            gs = PlyData.read(os.path.join(self.gs_path, sample["file_path"]))
        except Exception:
            print("Error in loading", os.path.join(self.gs_path, sample["file_path"]))

        vertex = gs["vertex"]

        data = read_gaussian_attribute(vertex, self.attribute)
        data_original = data.copy()
        data, centroid, scale_factor, scale_c, scale_m = self.norm_gs(data, self.norm_attribute)

        # choice_gs = np.random.choice(len(data), self.sample_points_num, replace=True)
        # data = data[choice_gs, :]
        data = data[:self.sample_points_num ,:]

        data = torch.from_numpy(data).float()
        scale_c = torch.from_numpy(scale_c).float()
        scale_m = torch.tensor(scale_m).float()
        data_original = torch.from_numpy(data_original).float()

        return sample["taxonomy_id"], sample["model_id"], data, centroid, scale_factor, scale_c, scale_m

    def __len__(self):
        return len(self.file_list)