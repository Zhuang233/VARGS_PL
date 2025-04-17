import numpy as np
import time
from .mesh_io import write_off
from .file_utils import make_dir_for_file


# def model_space_to_volume_space(pts_ms, vol_res):
#     pts_pos_octant = (pts_ms + 1.0) / 2.0
#     return np.floor(pts_pos_octant * vol_res).astype(int)
def model_space_to_volume_space(pts_ms, vol_res):
    pts_pos_octant = (pts_ms + 1.0) / 2.0
    pts_pos_octant = np.clip(pts_pos_octant, 0, 1 - 1e-6)  # 防止 * vol_res 后恰好等于 256
    return np.floor(pts_pos_octant * vol_res).astype(int)


def cartesian_dist(vec_x: np.array, vec_y: np.array, axis=1) -> np.ndarray:
    dist = np.linalg.norm(vec_x - vec_y, axis=axis)
    return dist

def add_samples_to_volume(vol, pos_ms, val):
    """
    add samples, average multiple values per voxel
    :param vol:
    :param pos_ms:
    :param val:
    :return:
    """

    # get distance between samples and their corresponding voxel centers
    pos_vs = model_space_to_volume_space(pos_ms, vol.shape[0])
    # grid_cell_centers_ms = volume_space_to_model_space(pos_vs, vol.shape[0])
    grid_cell_centers_ms = pos_ms
    dist_pos_cell_center = cartesian_dist(pos_ms, grid_cell_centers_ms)

    # cluster by voxel
    unique_grid_pos, unique_counts = np.unique(pos_vs, return_counts=True, axis=0)
    values_per_voxel = np.split(val, np.cumsum(unique_counts)[:-1])
    dist_pos_cell_center_per_voxel = np.split(dist_pos_cell_center, np.cumsum(unique_counts)[:-1])
    coordinates_per_voxel = np.split(pos_vs, np.cumsum(unique_counts)[:-1])
    coordinates_per_voxel_first = np.array([c[0] for c in coordinates_per_voxel])

    # get sample closest to voxel center
    dist_pos_cell_center_per_voxel_arg_min = [np.argmin(voxel_data) for voxel_data in dist_pos_cell_center_per_voxel]
    # values_per_voxel_mean = np.array([v.mean() for v in values_per_voxel])  # TODO: weight by distance to voxel center
    values_per_voxel_closest = np.array([v[dist_pos_cell_center_per_voxel_arg_min[vi]]
                                        for vi, v in enumerate(values_per_voxel)])
    vol[coordinates_per_voxel_first[:, 0], coordinates_per_voxel_first[:, 1], coordinates_per_voxel_first[:, 2]] = \
        values_per_voxel_closest  # values_per_voxel_mean
    return vol

def propagate_sign(vol, sigma=5, certainty_threshold=13):
    """
    iterative propagation of SDF signs from 'seed' voxels to get a dense, truncated volume
    :param vol:
    :param certainty_threshold: int in (0..5^3]
    :param sigma: neighborhood of propagation (kernel size)
    :return:
    """

    from scipy.ndimage.filters import convolve

    # # remove disconnected voxels as pre-processing
    # # e.g. a single outside voxel surrounded by inside voxels
    # sigma_disc = 3
    # kernel_neighbor_sum = np.ones((sigma_disc, sigma_disc, sigma_disc), dtype=np.float32)
    # kernel_center = int(sigma_disc / 2)
    # kernel_neighbor_sum[kernel_center, kernel_center, kernel_center] = 0.0
    # neighbor_sum = convolve(np.sign(vol), kernel_neighbor_sum, mode='nearest')
    # num_neighbors = sigma_disc**3.0 - 1.0
    # neighbors_have_same_sign = np.abs(neighbor_sum) == int(num_neighbors)
    # disconnected_voxels = np.logical_and(neighbors_have_same_sign, np.sign(neighbor_sum) != np.sign(vol))
    # vol[disconnected_voxels] = neighbor_sum[disconnected_voxels] / num_neighbors

    # smoothing as pre-processing helps with the reconstruction noise
    # quality should be ok as long as the surface stays at the same place
    # over-smoothing reduces the overall quality
    # sigma_smooth = 3
    # kernel_neighbor_sum = np.ones((sigma_smooth, sigma_smooth, sigma_smooth), dtype=np.float32)
    # vol = convolve(vol, kernel_neighbor_sum, mode='nearest') / sigma_smooth**3

    vol_sign_propagated = np.sign(vol)
    unknown_initially = vol_sign_propagated == 0
    kernel = np.ones((sigma, sigma, sigma), dtype=np.float32)

    # assume borders to be outside, reduce number of iterations
    vol[+0, :, :] = -1.0
    vol[-1, :, :] = -1.0
    vol[:, +0, :] = -1.0
    vol[:, -1, :] = -1.0
    vol[:, :, +0] = -1.0
    vol[:, :, -1] = -1.0

    while True:
        unknown_before = vol_sign_propagated == 0
        if unknown_before.sum() == 0:
            break

        # sum sign of neighboring voxels
        vol_sign_propagated_new = convolve(vol_sign_propagated, kernel, mode='nearest')

        # update only when the sign is certain
        vol_sign_propagated_new_unsure = np.abs(vol_sign_propagated_new) < certainty_threshold
        vol_sign_propagated_new[vol_sign_propagated_new_unsure] = 0.0

        # map again to [-1.0, 0.0, +1.0]
        vol_sign_propagated_new = np.sign(vol_sign_propagated_new)

        # stop when no changes happen
        unknown_after = vol_sign_propagated_new == 0
        if unknown_after.sum() >= unknown_before.sum():
            break  # no changes -> some values might be caught in a tie
        vol_sign_propagated[unknown_initially] = vol_sign_propagated_new[unknown_initially]  # add new values

    vol[vol == 0] = vol_sign_propagated[vol == 0]
    return vol

def implicit_surface_to_mesh(query_dist_ms, query_pts_ms,
                             volume_out_file, mc_out_file, grid_res, sigma, certainty_threshold=26):

    if query_dist_ms.max() == 0.0 and query_dist_ms.min() == 0.0:
        print('WARNING: implicit surface for {} contains only zeros'.format(volume_out_file))
        return

    # add known values and propagate their signs to unknown values
    volume = np.zeros((grid_res, grid_res, grid_res))
    volume = add_samples_to_volume(volume, query_pts_ms, query_dist_ms)

    start = time.time()
    volume = propagate_sign(volume, sigma, certainty_threshold)
    end = time.time()
    print('Sign propagation took: {}'.format(end - start))

    # clamp to -1..+1
    volume[volume < -1.0] = -1.0
    volume[volume > 1.0] = 1.0

    # green = inside; red = outside
    query_dist_ms_norm = query_dist_ms / np.max(np.abs(query_dist_ms))
    query_pts_color = np.zeros((query_dist_ms_norm.shape[0], 3))
    query_pts_color[query_dist_ms_norm < 0.0, 0] = np.abs(query_dist_ms_norm[query_dist_ms_norm < 0.0]) + 1.0 / 2.0
    query_pts_color[query_dist_ms_norm > 0.0, 1] = query_dist_ms_norm[query_dist_ms_norm > 0.0] + 1.0 / 2.0
    write_off(volume_out_file, query_pts_ms, np.array([]), colors_vertex=query_pts_color)

    if volume.min() < 0.0 and volume.max() > 0.0:
        # reconstruct mesh from volume using marching cubes
        from skimage import measure
        start = time.time()
        # v, f, normals, values = measure.marching_cubes_lewiner(volume, 0)
        v, f, normals, values = measure.marching_cubes(volume, level=0)
        end = time.time()
        print('Marching Cubes Lewiner took: {}'.format(end - start))

        if v.size == 0 and f.size == 0:
            print('Warning: marching cubes gives no result!')
        else:
            import trimesh
            import trimesh.repair
            v = (((v + 0.5) / float(grid_res)) - 0.5) * 2.0
            mesh = trimesh.Trimesh(vertices=v, faces=f)
            trimesh.repair.fix_inversion(mesh)
            make_dir_for_file(mc_out_file)
            mesh.export(mc_out_file)
    else:
        print('Warning: volume for marching cubes contains no 0-level set!')