import os
import math
from multiprocessing import Manager
from weakref import ref

import numpy as np
from coperception.utils.obj_util import *
from coperception.datasets.NuscenesDataset import NuscenesDataset
from torch.utils.data import Dataset
import torch
from concurrent import futures
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce
from coperception.models.det.base.DetModelBase import *

class LA_V2XSimDet(Dataset):
    def __init__(
        self,
        dataset_roots=None,
        config=None,
        config_global=None,
        split=None,
        cache_size=10000,
        val=False,
        bound=None,
        kd_flag=False,
        rsu=False,
        k=3,
        tau=[1,1,1,1,1,1],
    ):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size

        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis

        self.bound = bound
        self.kd_flag = kd_flag
        self.rsu = rsu
        self.k = k
        self.tau = tau # Latency time for each agent, list length: agent_num
        # dataset_root = dataset_root + '/'+split
        if dataset_roots is None:
            raise ValueError(
                "The {} dataset root is None. Should specify its value.".format(
                    self.split
                )
            )
        self.dataset_roots = dataset_roots
        self.num_agent = len(dataset_roots)
        self.seq_files = []
        self.seq_scenes = []
        # for dataset_root in self.dataset_roots:
        #     # sort directories
        #     dir_list = [d.split("_") for d in os.listdir(dataset_root)]
        #     dir_list.sort(key=lambda x: (int(x[0]), int(x[1])))
        #     self.seq_scenes.append(
        #         [int(s[0]) for s in dir_list]
        #     )  # which scene this frame belongs to (required for visualization)
        #     dir_list = ["_".join(x) for x in dir_list]

        #     seq_dirs = [
        #         os.path.join(dataset_root, d)
        #         for d in dir_list
        #         if os.path.isdir(os.path.join(dataset_root, d))
        #     ]

        #     self.seq_files.append(
        #         [
        #             os.path.join(seq_dir, f)
        #             for seq_dir in seq_dirs
        #             for f in os.listdir(seq_dir)
        #             if os.path.isfile(os.path.join(seq_dir, f))
        #         ]
        #     )
        self.get_historical_data_dict()
        self.num_sample_seqs = len(self.seq_files[0])
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))
        # object information
        self.anchors_map = init_anchors_no_check(
            self.area_extents, self.voxel_size, self.box_code_size, self.anchor_size
        )
        self.map_dims = [
            int(
                (self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]
            ),
            int(
                (self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1]
            ),
        ]
        self.reg_target_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.pred_len,
            self.box_code_size,
        )
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (
            self.map_dims[0],
            self.map_dims[1],
            len(self.anchor_size),
            self.category_num,
        )
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = [manager.dict() for _ in range(self.num_agent)]
        self.cache_size = cache_size if split == "train" else 0

        if self.val:
            self.voxel_size_global = config_global.voxel_size
            self.area_extents_global = config_global.area_extents
            self.pred_len_global = config_global.pred_len
            self.box_code_size_global = config_global.box_code_size
            self.anchor_size_global = config_global.anchor_size
            # object information
            self.anchors_map_global = init_anchors_no_check(
                self.area_extents_global,
                self.voxel_size_global,
                self.box_code_size_global,
                self.anchor_size_global,
            )
            self.map_dims_global = [
                int(
                    (self.area_extents_global[0][1] - self.area_extents_global[0][0])
                    / self.voxel_size_global[0]
                ),
                int(
                    (self.area_extents_global[1][1] - self.area_extents_global[1][0])
                    / self.voxel_size_global[1]
                ),
            ]
            self.reg_target_shape_global = (
                self.map_dims_global[0],
                self.map_dims_global[1],
                len(self.anchor_size_global),
                self.pred_len_global,
                self.box_code_size_global,
            )
            self.dims_global = config_global.map_dims
        self.get_meta()

    def get_meta(self):
        meta = NuscenesDataset(
            dataset_root=self.dataset_roots[0],
            split=self.split,
            config=self.config,
            val=self.val,
        )
        if not self.val:
            (
                self.padded_voxel_points_meta,
                self.label_one_hot_meta,
                self.reg_target_meta,
                self.reg_loss_mask_meta,
                self.anchors_map_meta,
                _,
                _,
                self.vis_maps_meta,
            ) = meta[0]
        else:
            (
                self.padded_voxel_points_meta,
                self.label_one_hot_meta,
                self.reg_target_meta,
                self.reg_loss_mask_meta,
                self.anchors_map_meta,
                _,
                _,
                self.vis_maps_meta,
                _,
                _,
            ) = meta[0]
        del meta

    def __len__(self):
        return self.num_sample_seqs

    def get_historical_data_dict(self, central_agent = 1):
        self.seq_files = []
        for agent in range(len(self.dataset_roots)):
            agent_files = []
            dataset_root = self.dataset_roots[agent]
            dir_list = [
                d.split("_") 
                for d in os.listdir(dataset_root)
                # if int(d.split("_")[1]) >= (max(self.tau) + self.k - 1)]
                if int(d.split("_")[1]) >= 12]
            dir_list.sort(key=lambda x: (int(x[0]), int(x[1])))

            # dir_list = ["_".join([x[0], str(int(x[1]) - self.tau[agent])]) for x in dir_list]
            
            for scene, seq in dir_list:
                seq_list = []
                for t in range(self.k):
                    file_name = "_".join([scene, str(int(seq) - self.tau[agent] - self.k + (t + 1))])
                    path_name = os.path.join(dataset_root, file_name)
                    file_path = os.path.join(path_name, os.listdir(path_name)[0])
                    seq_list.append(file_path)
                agent_files.append(seq_list)

            self.seq_files.append(agent_files)

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def pick_newest_agent(self, para):
        agent_id, idx, res_his = para
        empty_flag = False
        seq_file = self.seq_files[agent_id][idx][-1]
        scene_index = seq_file.split('/')[-2]
        if scene_index in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][scene_index]
        else:
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                padded_voxel_points = []
                padded_voxel_points_teacher = []
                label_one_hot = np.zeros_like(self.label_one_hot_meta)
                reg_target = np.zeros_like(self.reg_target_meta)
                anchors_map = np.zeros_like(self.anchors_map_meta)
                vis_maps = np.zeros_like(self.vis_maps_meta)
                reg_loss_mask = np.zeros_like(self.reg_loss_mask_meta)

                if self.bound == "lowerbound":
                    padded_voxel_points = np.zeros_like(self.padded_voxel_points_meta)

                if self.kd_flag or self.bound == "upperbound":
                    padded_voxel_points_teacher = np.zeros_like(
                        self.padded_voxel_points_meta
                    )
                trans_matrices = np.zeros((self.num_agent, 4, 4))
                if self.k > 1:
                    padded_voxel_points_his = np.concatenate([res_his[t][0] for t in range(self.k-1)])
                    padded_voxel_points = np.concatenate([padded_voxel_points_his, padded_voxel_points])
                # trans_matrices_his = np.stack([res_his[t][-1] for t in range(self.k-1)])
                # trans_matrices = np.concatenate([trans_matrices_his, np.expand_dims(trans_matrices, 0)])

                current_pose_rec = {} # ego pose
                current_pose_rec['rotation'] = np.zeros(4)
                current_pose_rec['translation'] = np.zeros(3)
                current_cs_rec = {} # calibrated_sensor
                current_cs_rec['rotation'] = np.zeros(4)
                current_cs_rec['translation'] = np.zeros(3)
                
                if self.k > 1:
                    current_pose_rec_rotation_list = [res_his[t][1]['rotation'] for t in range(self.k - 1)]
                    current_pose_rec_translation_list = [res_his[t][1]['translation'] for t in range(self.k - 1)]
                    current_cs_rec_rotation_list = [res_his[t][2]['rotation'] for t in range(self.k - 1)]
                    current_cs_rec_translation_list = [res_his[t][2]['translation'] for t in range(self.k - 1)]
                    current_pose_rec_rotation_list.append(current_pose_rec['rotation'])
                    current_pose_rec_translation_list.append(current_pose_rec['translation'])
                    current_cs_rec_rotation_list.append(current_pose_rec['rotation'])
                    current_cs_rec_translation_list.append(current_pose_rec['translation'])
                    
                    current_pose_rec_rotation = np.stack(current_pose_rec_rotation_list)
                    current_pose_rec_translation = np.stack(current_pose_rec_translation_list)
                    current_cs_rec_rotation = np.stack(current_cs_rec_rotation_list)
                    current_cs_rec_translation = np.stack(current_cs_rec_translation_list)
                
                if self.val:
                    return [
                        padded_voxel_points,
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        [{"gt_box": []}],
                        [seq_file],
                        0,
                        0,
                        trans_matrices,
                        current_pose_rec_rotation,
                        current_pose_rec_translation,
                        current_cs_rec_rotation,
                        current_cs_rec_translation
                    ]
                else:
                    return [
                        padded_voxel_points,
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        0,
                        0,
                        trans_matrices,
                        current_pose_rec_rotation,
                        current_pose_rec_translation,
                        current_cs_rec_rotation,
                        current_cs_rec_translation
                    ]
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][scene_index] = gt_dict

        if not empty_flag:
            allocation_mask = gt_dict["allocation_mask"].astype(bool)
            reg_loss_mask = gt_dict["reg_loss_mask"].astype(bool)
            gt_max_iou = gt_dict["gt_max_iou"]

            # load regression target
            reg_target_sparse = gt_dict["reg_target_sparse"]
            # need to be modified Yiqi , only use reg_target and allocation_map
            reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

            reg_target[allocation_mask] = reg_target_sparse
            reg_target[np.bitwise_not(reg_loss_mask)] = 0
            label_sparse = gt_dict["label_sparse"]

            one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
            label_one_hot = np.zeros(self.label_one_hot_shape)
            label_one_hot[:, :, :, 0] = 1
            label_one_hot[allocation_mask] = one_hot_label_sparse

            if self.only_det:
                reg_target = reg_target[:, :, :, :1]
                reg_loss_mask = reg_loss_mask[:, :, :, :1]

            # only center for pred
            elif self.config.pred_type in ["motion", "center"]:
                reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
                reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
                reg_loss_mask[:, :, :, 1:, 2:] = False

            # Prepare padded_voxel_points
            padded_voxel_points = []
            if self.bound == "lowerbound" or self.bound == "both":
                for i in range(self.num_past_pcs):
                    indices = gt_dict["voxel_indices_" + str(i)]
                    curr_voxels = np.zeros(self.dims, dtype=bool)
                    curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    curr_voxels = np.rot90(curr_voxels, 3)
                    padded_voxel_points.append(curr_voxels)
                padded_voxel_points = np.stack(padded_voxel_points, 0).astype(
                    np.float32
                )
                padded_voxel_points = padded_voxel_points.astype(np.float32)
            
            anchors_map = self.anchors_map

            if self.config.use_vis:
                vis_maps = np.zeros(
                    (
                        self.num_past_pcs,
                        self.config.map_dims[-1],
                        self.config.map_dims[0],
                        self.config.map_dims[1],
                    )
                )
                vis_free_indices = gt_dict["vis_free_indices"]
                vis_occupy_indices = gt_dict["vis_occupy_indices"]
                vis_maps[
                    vis_occupy_indices[0, :],
                    vis_occupy_indices[1, :],
                    vis_occupy_indices[2, :],
                    vis_occupy_indices[3, :],
                ] = math.log(0.7 / (1 - 0.7))
                vis_maps[
                    vis_free_indices[0, :],
                    vis_free_indices[1, :],
                    vis_free_indices[2, :],
                    vis_free_indices[3, :],
                ] = math.log(0.4 / (1 - 0.4))
                vis_maps = np.swapaxes(vis_maps, 2, 3)
                vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
                for v_id in range(vis_maps.shape[0]):
                    vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
                vis_maps = vis_maps[-1]
            else:
                vis_maps = np.zeros(0)

            if self.rsu:
                trans_matrices = gt_dict["trans_matrices"]
            else:
                trans_matrices = gt_dict["trans_matrices_no_cross_road"]
            
            current_pose_rec = {} # ego pose
            # current_pose_rec_rotation = np.array(gt_dict['current_pose_rec']['rotation'])
            # current_pose_rec_translation = np.array(gt_dict['current_pose_rec']['translation'])
            current_pose_rec['rotation'] = np.array(gt_dict['current_pose_rec']['rotation'])
            current_pose_rec['translation'] = np.array(gt_dict['current_pose_rec']['translation'])
            current_cs_rec = {} # calibrated_sensor
            # current_cs_rec_rotation = np.array(gt_dict['current_cs_rec']['rotation'])
            # current_cs_rec_translation = np.array(gt_dict['current_cs_rec']['translation'])
            current_cs_rec['rotation'] = np.array(gt_dict['current_cs_rec']['rotation'])
            current_cs_rec['translation'] = np.array(gt_dict['current_cs_rec']['translation'])

            if self.k > 1:
                current_pose_rec_rotation_list = [res_his[t][1]['rotation'] for t in range(self.k - 1)]
                current_pose_rec_translation_list = [res_his[t][1]['translation'] for t in range(self.k - 1)]
                current_cs_rec_rotation_list = [res_his[t][2]['rotation'] for t in range(self.k - 1)]
                current_cs_rec_translation_list = [res_his[t][2]['translation'] for t in range(self.k - 1)]
                current_pose_rec_rotation_list.append(current_pose_rec['rotation'])
                current_pose_rec_translation_list.append(current_pose_rec['translation'])
                current_cs_rec_rotation_list.append(current_cs_rec['rotation'])
                current_cs_rec_translation_list.append(current_cs_rec['translation'])
                
                current_pose_rec_rotation = np.stack(current_pose_rec_rotation_list)
                current_pose_rec_translation = np.stack(current_pose_rec_translation_list)
                current_cs_rec_rotation = np.stack(current_cs_rec_rotation_list)
                current_cs_rec_translation = np.stack(current_cs_rec_translation_list)

            label_one_hot = label_one_hot.astype(np.float32)
            reg_target = reg_target.astype(np.float32)
            anchors_map = anchors_map.astype(np.float32)
            vis_maps = vis_maps.astype(np.float32)

            target_agent_id = gt_dict["target_agent_id"]
            num_sensor = gt_dict["num_sensor"]

            # Prepare padded_voxel_points_teacher
            padded_voxel_points_teacher = []
            if "voxel_indices_teacher" in gt_dict and (
                self.kd_flag or self.bound == "upperbound" or self.bound == "both"
            ):
                if self.rsu:
                    indices_teacher = gt_dict["voxel_indices_teacher"]
                else:
                    indices_teacher = gt_dict["voxel_indices_teacher_no_cross_road"]

                curr_voxels_teacher = np.zeros(self.dims, dtype=bool)
                curr_voxels_teacher[
                    indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]
                ] = 1
                curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
                padded_voxel_points_teacher.append(curr_voxels_teacher)
                padded_voxel_points_teacher = np.stack(
                    padded_voxel_points_teacher, 0
                ).astype(np.float32)
                padded_voxel_points_teacher = padded_voxel_points_teacher.astype(
                    np.float32
                )

            if self.k > 1:
                padded_voxel_points_his = np.concatenate([res_his[t][0] for t in range(self.k-1)])
                padded_voxel_points = np.concatenate([padded_voxel_points_his, padded_voxel_points])
            # trans_matrices_his = np.stack([res_his[t][-1] for t in range(self.k-1)])
            # trans_matrices = np.concatenate([trans_matrices_his, np.expand_dims(trans_matrices, 0)])
            # print(agent_id, ' is ready to return:')
            if self.val:
                return [
                    padded_voxel_points,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    [{"gt_box": gt_max_iou}],
                    [seq_file],
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                    current_pose_rec_rotation,
                    current_pose_rec_translation,
                    current_cs_rec_rotation,
                    current_cs_rec_translation
                ]

            else:
                return [
                    padded_voxel_points,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                    current_pose_rec_rotation,
                    current_pose_rec_translation,
                    current_cs_rec_rotation,
                    current_cs_rec_translation
                ]

    def pick_historical_frame(self, para, center_agent = 1):
        agent_id, idx, t = para
        empty_flag = False
        if t >= 0:     
            seq_file = self.seq_files[agent_id][idx][t]
        else:
            seq_file = self.seq_files[center_agent][idx][t]
            seq_file_list = seq_file.split('/')
            seq_file_list[-3] = 'agent' + str(agent_id)
            seq_file = '/'.join(seq_file_list)
        scene_index = seq_file.split('/')[-2]
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][scene_index]
        else:
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                padded_voxel_points = []

                if self.bound == "lowerbound":
                    padded_voxel_points = np.zeros_like(self.padded_voxel_points_meta)

                current_pose_rec = {} # ego pose
                current_pose_rec['rotation'] = np.zeros(4)
                current_pose_rec['translation'] = np.zeros(3)
                current_cs_rec = {} # calibrated_sensor
                current_cs_rec['rotation'] = np.zeros(4)
                current_cs_rec['translation'] = np.zeros(3)

                if t >= 0:     
                    return (
                            padded_voxel_points,
                            current_pose_rec,
                            current_cs_rec
                        )

                if t < 0:
                    return (
                            padded_voxel_points
                        )

            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][scene_index] = gt_dict

        if not empty_flag: 
            padded_voxel_points = []
            if self.bound == "lowerbound" or self.bound == "both":
                for i in range(self.num_past_pcs):
                    indices = gt_dict["voxel_indices_" + str(i)]
                    curr_voxels = np.zeros(self.dims, dtype=bool)
                    curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    curr_voxels = np.rot90(curr_voxels, 3)
                    padded_voxel_points.append(curr_voxels)
                padded_voxel_points = np.stack(padded_voxel_points, 0).astype(
                    np.float32
                )
                padded_voxel_points = padded_voxel_points.astype(np.float32)

            current_pose_rec = {} # ego pose
            current_pose_rec['rotation'] = np.array(gt_dict['current_pose_rec']['rotation'])
            current_pose_rec['translation'] = np.array(gt_dict['current_pose_rec']['translation'])
            current_cs_rec = {} # calibrated_sensor
            current_cs_rec['rotation'] = np.array(gt_dict['current_cs_rec']['rotation'])
            current_cs_rec['translation'] = np.array(gt_dict['current_cs_rec']['translation'])
        
        if t >= 0:     
            return (
                    padded_voxel_points,
                    current_pose_rec,
                    current_cs_rec
                )

        if t < 0:
            return (
                    padded_voxel_points
                )

    def __getitem__(self, idx):
        # res = []
        # res = [[] for i in range(self.num_agent)]
        workers = self.num_agent
        his_workers = (self.k - 1) * self.num_agent
        load_list_his_frames = [
                                    [agent, idx, t]
                                    for agent in range(self.num_agent)
                                    for t in range(self.k-1)
                                ]

        load_list_supervision_frames = [
                                    [agent, idx, -1]
                                    for agent in range(self.num_agent)
                                ]
        self.pick_historical_frame([2, idx, 1])
        with futures.ThreadPoolExecutor(None) as executor:
            res_his = executor.map(self.pick_historical_frame, sorted(load_list_his_frames), chunksize=his_workers)
            res_his = list(res_his)

        with futures.ThreadPoolExecutor(None) as executor:
            res_supervision = executor.map(self.pick_historical_frame, sorted(load_list_supervision_frames), chunksize=his_workers)
            res_supervision = list(res_supervision)

        load_list_new_frames = [[agent, idx, [res_his[agent*(self.k-1)+t] for t in range(self.k-1)]] for agent in range(self.num_agent)]
        
        a = self.pick_newest_agent([1,idx,[res_his[0],res_his[1]]])
        # b = self.pick_newest_agent([3,idx,[res_his[6],res_his[7]]])
        with futures.ThreadPoolExecutor(None) as executor:
            res = executor.map(self.pick_newest_agent, sorted(load_list_new_frames), chunksize=workers)
            res = list(res)
        # res = []
        # for para in load_list_new_frames:
        #     res.append(self.pick_newest_agent(para))
        center_agent = 1
        current_pose_rec_rotation = np.stack([res[i][-4] for i in range(len(res))], 0)
        current_pose_rec_translation = np.stack([res[i][-3] for i in range(len(res))], 0)
        current_cs_rec_rotation = np.stack([res[i][-2] for i in range(len(res))], 0)
        current_cs_rec_translation = np.stack([res[i][-1] for i in range(len(res))], 0)
        transform_matrices_center, transform_matrices_newest = self.get_transmatrices([current_pose_rec_rotation, current_pose_rec_translation, current_cs_rec_rotation, current_cs_rec_translation, center_agent])
        
        for agent_id in range(len(res)):
            del res[agent_id][-4:]
            res[agent_id].append(transform_matrices_center[agent_id])
            res[agent_id].append(transform_matrices_newest[agent_id])
            res[agent_id].append(res_supervision[agent_id])

        return res

    def trans2newest(self, para):
        return 0


    def vector2mat(self, pose_para):
       current_pose_rec_rotation, current_pose_rec_translation, current_cs_rec_rotation, current_cs_rec_translation, ref_from_car, car_from_global = pose_para
       car_from_ref = transform_matrix(current_cs_rec_translation, Quaternion(np.array(current_cs_rec_rotation)), inverse=False)
       global_from_car = transform_matrix(current_pose_rec_translation, Quaternion(np.array(current_pose_rec_rotation)), inverse=False)
       trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_ref])
       try:
        trans_matrix=np.linalg.inv(trans_matrix)
       except:
        trans_matrix = np.zeros(trans_matrix.shape)
       return trans_matrix


    def vector2mat_newest(self, pose_para):
       current_pose_rec_rotation, current_pose_rec_translation, current_cs_rec_rotation, current_cs_rec_translation, ref_from_car, car_from_global = pose_para
       car_from_ref = transform_matrix(current_cs_rec_translation, Quaternion(np.array(current_cs_rec_rotation)), inverse=False)
       global_from_car = transform_matrix(current_pose_rec_translation, Quaternion(np.array(current_pose_rec_rotation)), inverse=False)
       trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_ref])
    #    trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_ref])

    #    try:
    #     trans_matrix=np.linalg.inv(trans_matrix)
    #    except:
    #     trans_matrix = np.zeros(trans_matrix.shape)
       return trans_matrix

    def get_transmatrices(self, pose_para):
        current_pose_rec_rotation, current_pose_rec_translation, current_cs_rec_rotation, current_cs_rec_translation, center_agent = pose_para
        num_agent, k, _ = current_pose_rec_rotation.shape

        ref_from_center_car = transform_matrix(current_cs_rec_translation[center_agent][-1], Quaternion(np.array(current_cs_rec_rotation[center_agent][-1])), inverse=True)
        center_car_from_global = transform_matrix(current_pose_rec_translation[center_agent][-1], Quaternion(np.array(current_pose_rec_rotation[center_agent][-1])), inverse=True)

        # per_agent_pose_para_newest = [[current_pose_rec_rotation[agent_id], current_pose_rec_translation[agent_id], current_cs_rec_rotation[agent_id],current_cs_rec_translation[agent_id]] for agent_id in range(num_agent)]

        per_agent_pose_para_newest = [[ref_from_center_car, center_car_from_global, current_pose_rec_rotation[agent_id], current_pose_rec_translation[agent_id], current_cs_rec_rotation[agent_id],current_cs_rec_translation[agent_id]] for agent_id in range(num_agent)]
        
        with futures.ThreadPoolExecutor(None) as executor:
            res2newest = executor.map(self.per_agent_get2newest_transmatrices, per_agent_pose_para_newest, chunksize=num_agent)
            res2newest = list(res2newest)

        trans_matrices_newest = np.stack(res2newest)

        per_agent_pose_para_center = [[current_pose_rec_rotation[agent_id][-1], current_pose_rec_translation[agent_id][-1], current_cs_rec_rotation[agent_id][-1], current_cs_rec_translation[agent_id][-1], ref_from_center_car , center_car_from_global] for agent_id in range(num_agent)]

        with futures.ThreadPoolExecutor(None) as executor:
            res2center = executor.map(self.vector2mat, per_agent_pose_para_center, chunksize=num_agent)
            res2center = list(res2center)

        trans_matrices_center = np.stack(res2center)

        return trans_matrices_center, trans_matrices_newest

    def per_agent_get2newest_transmatrices(self, pose_para):
        ref_from_center_car, center_car_from_global, current_pose_rec_rotation, current_pose_rec_translation, current_cs_rec_rotation, current_cs_rec_translation = pose_para
        k, _ = current_pose_rec_rotation.shape
        # ref_from_car = transform_matrix(current_cs_rec_translation[-1], Quaternion(current_cs_rec_rotation[-1]), inverse=True)
        # car_from_global = transform_matrix(current_pose_rec_translation[-1], Quaternion(current_pose_rec_rotation[-1]), inverse=True)
        ref_from_car = ref_from_center_car
        car_from_global = center_car_from_global
        pose_para_newest_para = [[current_pose_rec_rotation[i], current_pose_rec_translation[i], current_cs_rec_rotation[i], current_cs_rec_translation[i], ref_from_car, car_from_global] for i in range(k)]

        with futures.ThreadPoolExecutor(None) as executor:
            res = executor.map(self.vector2mat, pose_para_newest_para, chunksize=k-1)
            res = list(res)
        trans_matrices = np.stack(res)

        
        return trans_matrices


