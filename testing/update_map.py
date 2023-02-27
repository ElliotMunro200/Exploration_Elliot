import numpy as np

import depth_utils_update_map as du
from scipy.ndimage.morphology import binary_dilation
from skimage import morphology
from clustering import frontier_clustering

def update_map(depth, current_pose):  # depth image shape(256,256) values (95.0-250.0), max_value (at 111,81), [pose_x_cm, pose_y_cm, ori_deg] = 1280.0,1280.0,0.0
    with np.errstate(invalid="ignore"):
        depth[depth > 64 * 5] = np.NaN  # 64 blocks * 5cm/map_block = 320 cm.
    camera_matrix = du.get_camera_matrix(width=256, height=256, fov=90.0)
    mapp = np.zeros((2560 // 5, 2560 // 5, 3), dtype=np.float32)
    point_cloud = du.get_point_cloud_from_z(depth, camera_matrix, \
                                            scale=2)
    # (128,128,3)

    #  3D point cloud adjusted for camera view
    agent_view = du.transform_camera_view(point_cloud,
                                          sensor_height=1.25,  # 1.25m
                                          camera_elevation_degree=0)  # 0

    # (128,128,3)

    #  3D point cloud adjusted for position
    shift_loc = [64 * 5 // 2, 0, np.pi / 2.0]  # [160, 0, pi/2 = 90 deg]
    agent_view_centered = du.transform_pose(agent_view, shift_loc)

    # (128,128,3)

    #  bins 3D point cloud into xy-z bins of [above 25cm, between 25-150cm, above 150cm]
    agent_view_flat = du.bin_points(agent_view_centered, map_size=64, z_bins=[25, 150], xy_resolution=5)

    # (64,64,3)

    print(agent_view_flat[1])

    # simplifying/cropping point cloud into just the middle z range.
    agent_view_cropped = agent_view_flat[:, :, 1]

    # making the map binary. This is fp_proj.
    agent_view_cropped = agent_view_cropped / 1  # threshold = 1
    agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
    agent_view_cropped[agent_view_cropped < 0.5] = 0.0

    # explored
    agent_view_explored = agent_view_flat.sum(2)
    agent_view_explored[agent_view_explored > 0] = 1.0

    geocentric_pc = du.transform_pose(agent_view, current_pose)

    geocentric_flat = du.bin_points(
        geocentric_pc, mapp.shape[0], z_bins=[25,150], xy_resolution=5)

    mapp = mapp + geocentric_flat

    map_gt = mapp[:, :, 1] / 1  # (=1), map_gt = occupancy map
    map_gt[map_gt >= 0.5] = 1.0
    map_gt[map_gt < 0.5] = 0.0

    explored_gt = mapp.sum(2)  # sum along 2nd axis
    explored_gt[explored_gt > 1] = 1.0
    new_explored = geocentric_flat.sum(2)
    new_explored[new_explored > 1] = 1.0

    k = np.zeros((3, 3), dtype=int)
    k[1] = 1
    k[:, 1] = 1
    contour = binary_dilation(explored_gt == 0, k) & (explored_gt == 1)
    contour = contour & (binary_dilation(map_gt, k) == 0)
    contour = morphology.remove_small_objects(contour, 2)

    # num_frontier_points = len(np.nonzero(contour)[0])
    # if self.num_maps == 5:
    #     frontier_clusters = np.zeros(np.shape(contour))
    # elif num_frontier_points < 5:
    #     print(f"ONLY {num_frontier_points} FRONTIER POINTS, SO NOT ENOUGH FOR CLUSTERING")
    #     frontier_clusters = np.zeros(np.shape(contour))
    # else:
    #     frontier_clusters = frontier_clustering(contour, step=0, algo="AGNES", metric=None, save_freq=None)

    return agent_view_cropped, map_gt, agent_view_explored, explored_gt, new_explored, \
        contour.astype(float) #, frontier_clusters

if __name__ == "__main__":
    depth = np.random.rand(256, 256)
    depth = (depth * 155) + 95
    current_pose = [1280.0, 1280.0, 0.0]
    update_map(depth, current_pose)
