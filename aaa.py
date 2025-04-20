import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os
import torch
import random
import numpy as np
import traceback
from multiprocessing import Pool
from fnmatch import fnmatch
from scipy.spatial import cKDTree
import multiprocessing as mp
import json

def process(paths):
    path, save_path = paths
    ply_name = os.path.basename(path)
    name_no_ext = os.path.splitext(ply_name)[0]

    print(name_no_ext)

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    centroid = np.mean(xyz, axis=0)
    xyz  -= centroid
    scale_factor = np.max(np.linalg.norm(xyz, axis=1))  # 计算最大欧几里得距离
    xyz /= scale_factor

    _xyz = torch.tensor(xyz, dtype=torch.float, device="cpu").requires_grad_(True)
    xyz = _xyz.detach().cpu().numpy()
    occ = np.ones((1_048_576, 4))


    zero_num = min(xyz.shape[0] * 2, 1048576)
    occ[:xyz.shape[0],:3] = xyz
    occ[xyz.shape[0]:zero_num,:3] = xyz

    sample_near_num = max(int((1_048_576 - zero_num) * 0.8), 0)
    sample_far_num = max(1_048_576 - sample_near_num - zero_num, 0)

    sample = []
    sample_far = []
    ptree = cKDTree(xyz)
    sigmas = []
    for p in np.array_split(xyz, 100, axis=0):
        d = ptree.query(p, 151)
        sigmas.append(d[0][:,-1])
    sigmas = np.concatenate(sigmas)


    POINT_NUM = xyz.shape[0] // 60
    POINT_NUM_GT = xyz.shape[0] // 60 * 60
    QUERY_EACH = sample_near_num // xyz.shape[0] + 5
    for i in range(QUERY_EACH):
        # scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = xyz + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=xyz.shape)
        # tt = xyz + np.random.normal(0.0, 0.1, size = xyz.shape)
        for j in range(tt.shape[0]):
            dis, _ = ptree.query(tt[j], k = 1)
            if dis < 0.2 :
                tmp = np.zeros(4)
                tmp[:3] = tt[j]
                tmp[3] = (0.2 - dis) / 0.2 
                sample.append(tmp)
                if len(sample) > sample_near_num:
                    break
        if len(sample) > sample_near_num:
            break
    sample = np.asarray(sample[:sample_near_num])
    occ[zero_num:zero_num + sample_near_num] = sample


    bbox_min = -1  # 在批次和点的维度上找全局最小
    bbox_max = 1  # 在批次和点的维度上找全局最大
    space_samples = np.random.uniform(bbox_min, bbox_max, size=(sample_far_num * 3, 3))

    for j in range(space_samples.shape[0]):
        dis, _ = ptree.query(space_samples[j], k = 1)
        if dis > 0.2:
            tmp = np.zeros(4)
            tmp[:3] = space_samples[j] 
            sample_far.append(tmp)
            if len(sample_far) > sample_far_num:
                break
        if len(sample_far) > sample_far_num:
            break
    sample_far = np.asarray(sample_far[:sample_far_num])
    occ[zero_num + sample_near_num:] = sample_far
    occ = occ.reshape(1_048_576, 4)

    save_file_path = os.path.join(save_path, f"{name_no_ext}_occ.npy")
    np.save(save_file_path, occ)


if __name__ == '__main__':
    save_gaussian_folder = r'dataset/ShapeSplatsV1_chair'
    save_path = r'dataset/ShapeSplatsV1_chair/occ'

    pattern = "*.ply"
    paths = []
    prossess_num = 0

    with open(os.path.join(save_gaussian_folder, "split","train.txt"), "w") as f:
        for file in os.listdir(save_gaussian_folder):
            if fnmatch(file, pattern):
                f.write(file + "\n")
                ply_file_path = os.path.join(save_gaussian_folder, file)
                paths.append((ply_file_path, save_path))
            prossess_num += 1
            if prossess_num == 30:
                break

    print(f"{len(paths)} .ply files found for processing!")
    workers = 10
    pool = mp.Pool(workers)
    pool.map(process, paths)
# process((save_gaussian_folder, save_path))