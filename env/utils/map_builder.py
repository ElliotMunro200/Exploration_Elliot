import numpy as np

import env.utils.depth_utils as du
from scipy.ndimage.morphology import binary_dilation
from skimage import morphology
from clustering import frontier_clustering

class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution'] #5
        agent_min_z = params['agent_min_z'] #25
        agent_max_z = params['agent_max_z'] #150
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        self.num_maps = params['num_maps']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, depth, current_pose):  # depth image, [pose_x_cm, pose_y_cm, ori_deg]
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN  # 64 blocks * 5cm/map_block = 320 cm.
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)

        #  3D point cloud adjusted for camera view
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,  # 1.25m
                                              self.agent_view_angle)  # 0

        #  3D point cloud adjusted for position
        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]  # [160, 0, pi/2 = 90 deg]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        #  bins 3D point cloud into xy-z bins of [above 25cm, between 25-150cm, above 150cm]
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)

        # simplifying/cropping point cloud into just the middle z range.
        agent_view_cropped = agent_view_flat[:, :, 1]

        # making the map binary. This is fp_proj.
        agent_view_cropped = agent_view_cropped / self.obs_threshold  # threshold = 1
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        # explored
        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        self.map = self.map + geocentric_flat 

        map_gt = self.map[:, :, 1] / self.obs_threshold  # (=1), map_gt = occupancy map
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)  # sum along 2nd axis
        explored_gt[explored_gt > 1] = 1.0
        new_explored = geocentric_flat.sum(2)
        new_explored[new_explored > 1] = 1.0

        k = np.zeros((3,3),dtype=int)
        k[1] = 1
        k[:,1] = 1
        contour = binary_dilation(explored_gt==0, k) & (explored_gt==1)
        contour = contour & (binary_dilation(map_gt, k)==0)
        contour = morphology.remove_small_objects(contour, 2)
        
        num_frontier_points = len(np.nonzero(contour)[0])
        if self.num_maps == 5:
            frontier_clusters = np.zeros(np.shape(contour))
        elif num_frontier_points < 5:
            print(f"ONLY {num_frontier_points} FRONTIER POINTS, SO NOT ENOUGH FOR CLUSTERING")
            frontier_clusters = np.zeros(np.shape(contour))
        else:
            frontier_clusters = frontier_clustering(contour, step=0, algo="AGNES", metric=None, save_freq=None)

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, new_explored, contour.astype(float), frontier_clusters

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
