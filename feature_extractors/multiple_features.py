import torch
import numpy as np
import os
import math

from feature_extractors.features import Features
from utils.mvtec3d_util import organized_pc_to_unorganized_pc


def organized_pc_to_unorganized_pc_no_zeros(sample):
    # sample img=b 3 224 224, resized_organized_pc=b 3 224 224, resized_depth_map_3channel
    organized_pc = sample[1]  # b 3 224 224
    # maybe change to batch calculation
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 224 224 3
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)  # 224*224 3
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]  # 224*224
    # organized_pc_np = organized_pc.permute(0, 2, 3, 1).numpy()  # B 224 224 3
    # unorganized_pc = organized_pc_np.reshape(organized_pc_np.shape[0],
    #                                          organized_pc_np.shape[1] * organized_pc_np.shape[2],
    #                                          organized_pc_np.shape[3])  # B 224*224 3
    # nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=2))  # to B 224*224 to ()
    ############## if we want to do that we need to group and interpolar first
    # to 1 224*224-zero 3 to 1 3 224*224-zero
    unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
    return unorganized_pc_no_zeros, nonzero_indices


class RGBFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)
        self.patch_rgb_lib.append(rgb_patch)

    def run_coreset(self):
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)  # allsamples*N784 T768
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_rgb_lib',
                                                            coreset_dtype = self.coreset_dtype)
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch, _ = self.get_rgb_patch(rgb_feature_maps)
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 50176, 2

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(rgb_patch, mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx, rgb_path)

    def compute_s_s_map(self, rgb_patch, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, rgb_path):
        """
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        """

        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 2 224 224 to 224 448 to 448 224

        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

        self.img_name.append(rgb_path)

        if self.args.save_seg_results:
            print(rgb_path)
            seg_save_path = rgb_path[0]
            seg_save_path = seg_save_path.replace('mvtec_3d','segmentation')
            seg_save_path = seg_save_path.replace('png', 'pt')

            slash = seg_save_path.rfind('/')
            dic = seg_save_path[:slash]
            if not os.path.exists(dic):
                os.makedirs(dic)
            torch.save(s_map, seg_save_path)


class DepthFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        depth_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        rgb_patch, rgb_patch2 = self.get_rgb_patch(depth_feature_maps)
        self.patch_rgb_lib.append(rgb_patch)


    def run_coreset(self):
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)  # allsamples*N784 T768
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_rgb_lib',
                                                            coreset_dtype = self.coreset_dtype)
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        depth_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch, _ = self.get_rgb_patch(depth_feature_maps)
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 50176, 2

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        depth_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[2],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        rgb_patch, rgb_patch2 = self.get_rgb_patch(depth_feature_maps)

        self.compute_s_s_map(rgb_patch, mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, rgb_patch, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        """
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        """

        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 2 224 224 to 224 448 to 448 224

        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())


class PointFeatures(Features):
    # img, resized_organized_pc, resized_depth_map_3channel, gt[:1], label, rgb_path
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        # __call__ rgb, xyz
        # rgb_features = B 768 28 28 xyz_features= B 1152 128 center= B G xyz oidx= B G M 1 cidx B G 1 inter B 1152 N
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        self.patch_xyz_lib.append(xyz_patch)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)  # allsamples*N3136, T1152
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_xyz_lib',
                                                            coreset_dtype = self.coreset_dtype)
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

        # 2D dist, normalization because patch_xyz/rgb_lib has been concat and normalized in run coreset
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        # s=1 smap=1 1 224 224
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 50176, 2

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        # __call__ rgb, xyz
        # rgb_features = B 768 28 28 xyz_features= B 1152 128 center= B G xyz oidx= B G M 1 cidx B G 1 inter B 1152 N
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

        self.compute_s_s_map(xyz_patch, mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx, rgb_path)

    def compute_s_s_map(self, xyz_patch, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx, rgb_path):
        """
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        """

        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std  # normalization
        dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz],
                          dim=0).squeeze().reshape(1, -1).permute(1, 0)  # 2 224 224 to 224 448 to 448 224

        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

        self.img_name.append(rgb_path)

        if self.args.save_seg_results:
            print(rgb_path)
            seg_save_path = rgb_path[0]
            seg_save_path = seg_save_path.replace('mvtec_3d','segmentation')
            seg_save_path = seg_save_path.replace('png', 'pt')

            slash = seg_save_path.rfind('/')
            dic = seg_save_path[:slash]
            if not os.path.exists(dic):
                os.makedirs(dic)
            torch.save(s_map, seg_save_path)


class RGBorXYZWithOneHallucination(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)

        with torch.no_grad():
            if self.args.main_modality == 'rgb':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'xyz')
                elif self.args.use_hrnet:
                    rgb = sample[0].cuda()
                    hallucination = self.fusion.hallucination_generation(rgb)
                    # hallucination = self.fusion.hallucination_generation(sample[0])
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    frgb = rgb_patch2.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(rgb_feature=frgb, out_type='xyz').cpu()
                    # hallucination = self.fusion.feature_fusion(rgb_feature=rgb_patch2.unsqueeze(0), out_type='xyz')
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'xyz')
            elif self.args.main_modality == 'xyz':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'rgb')
                elif self.args.use_hrnet:
                    xyz = sample[1].cuda()
                    hallucination = self.fusion.hallucination_generation(xyz)
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    fxyz = xyz_patch.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(xyz_feature=fxyz, out_type='rgb').cpu()
                    # hallucination = self.fusion.feature_fusion(xyz_feature=xyz_patch.unsqueeze(0), out_type='rgb')
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'rgb')

            else:
                raise Exception('Unknown modality')

            assert len(hallucination.shape) == 3
        hallucination = hallucination.reshape(-1, hallucination.shape[2]).detach()  # 3136 1920

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_xyz_lib.append(xyz_patch)
        self.patch_fusion_lib.append(hallucination)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        if self.args.main_modality == 'rgb':
            self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        elif self.args.main_modality == 'xyz':
            self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std

        if self.f_coreset < 1:
            if self.args.main_modality == 'rgb':
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                                n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                                eps=self.coreset_eps, lib='patch_rgb_lib',
                                                                coreset_dtype=self.coreset_dtype)
                self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            elif self.args.main_modality == 'xyz':
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                                n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                                eps=self.coreset_eps, lib='patch_xyz_lib',
                                                                coreset_dtype=self.coreset_dtype)
                self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_fusion_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_fusion_lib',
                                                            coreset_dtype=self.coreset_dtype)
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)

        with torch.no_grad():
            if self.args.main_modality == 'rgb':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'xyz')
                elif self.args.use_hrnet:
                    rgb = sample[0].cuda()
                    hallucination = self.fusion.hallucination_generation(rgb)
                    # hallucination = self.fusion.hallucination_generation(sample[0])
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    frgb = rgb_patch2.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(rgb_feature=frgb, out_type='xyz').cpu()
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'xyz')
            elif self.args.main_modality == 'xyz':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'rgb')
                elif self.args.use_hrnet:
                    xyz = sample[1].cuda()
                    hallucination = self.fusion.hallucination_generation(xyz)
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    fxyz = xyz_patch.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(xyz_feature=fxyz, out_type='rgb').cpu()
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'rgb')
        hallucination = hallucination.reshape(-1, hallucination.shape[2]).detach()

        hallucination = (hallucination - self.fusion_mean) / self.fusion_std
        dist_fusion = self.calculate_dist(hallucination, self.patch_fusion_lib)
        fusion_feat_size = (int(math.sqrt(hallucination.shape[0])), int(math.sqrt(hallucination.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(hallucination, dist_fusion, fusion_feat_size,
                                                             modal='fusion')
        # print('hallucination', hallucination, hallucination.dtype)
        # print('dist_fusion', dist_fusion, dist_fusion.dtype)
        # print('fusion_feat_size', fusion_feat_size)
        if self.args.main_modality == 'rgb':
            rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
            dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)
            rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
            s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
            s = torch.tensor([[self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        elif self.args.main_modality == 'xyz':
            xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
            dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
            xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
            s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
            s = torch.tensor([[self.args.xyz_s_lambda * s_xyz,self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)
        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

        with torch.no_grad():
            if self.args.main_modality == 'rgb':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'xyz')
                elif self.args.use_hrnet:
                    rgb = sample[0].cuda()
                    hallucination = self.fusion.hallucination_generation(rgb)
                    # hallucination = self.fusion.hallucination_generation(sample[0])
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    frgb = rgb_patch2.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(rgb_feature=frgb, out_type='xyz').cpu()
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'xyz')
            elif self.args.main_modality == 'xyz':
                if self.args.use_uff:
                    hallucination = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0), 'rgb')
                elif self.args.use_hrnet:
                    xyz = sample[1].cuda()
                    hallucination = self.fusion.hallucination_generation(xyz)
                    assert tuple(hallucination.shape[1:]) == (768, 56, 56)
                    hallucination = hallucination.reshape(hallucination.shape[0], hallucination.shape[1], -1)
                    hallucination = hallucination.transpose(-1, -2).cpu()  # b 3136 768
                elif self.args.use_hn:
                    fxyz = xyz_patch.unsqueeze(0).cuda()
                    hallucination = self.fusion.hallucination_generation(xyz_feature=fxyz, out_type='rgb').cpu()
                    # hallucination = self.fusion.hallucination_generation(xyz_patch.unsqueeze(0),
                    #                                                      rgb_patch2.unsqueeze(0),
                    #                                                      'rgb')
        hallucination = hallucination.reshape(-1, hallucination.shape[2]).detach()

        self.compute_s_s_map(xyz_patch, rgb_patch, hallucination, mask, label,
                             center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx, rgb_path)

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx, rgb_path):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std
        dist_fusion = self.calculate_dist(fusion_patch, self.patch_fusion_lib)
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        if self.args.main_modality == 'rgb':
            rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
            dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)
            rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
            s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
            s = torch.tensor([[self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        elif self.args.main_modality == 'xyz':
            xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
            dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
            xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
            s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
            s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        # print(s)
        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, self.gt_size, self.gt_size)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

        self.img_name.append(rgb_path)

        if self.args.save_seg_results:
            print(rgb_path)
            seg_save_path = rgb_path[0]
            seg_save_path = seg_save_path.replace('mvtec_3d','segmentation')
            seg_save_path = seg_save_path.replace('png', 'pt')

            slash = seg_save_path.rfind('/')
            dic = seg_save_path[:slash]
            if not os.path.exists(dic):
                os.makedirs(dic)
            torch.save(s_map, seg_save_path)


class RGBorXYZWithOneHallucinationFromFeature(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)
        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        if self.args.main_modality == 'rgb':
            with torch.no_grad():
                frgb = rgb_patch2.unsqueeze(0).cuda()
                xyz_hallucination = self.fusion.hallucination_generation(frgb).cpu()

            organized_pc_np = xyz_hallucination.squeeze().permute(1, 2, 0).numpy()  # 224 224 3
            unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)  # 224*224 3
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]  # 224*224
            unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)

            xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(xyz=unorganized_pc_no_zeros.contiguous(), out_type='xyz')

            hallucination = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        elif self.args.main_modality == 'xyz':
            with torch.no_grad():
                fxyz = xyz_patch.unsqueeze(0).cuda()
                rgb_hallucination = self.fusion.hallucination_generation(fxyz).cpu()
                assert tuple(rgb_hallucination.shape) == tuple(sample[0].shape)
                rgb_feature_maps = self(rgb=rgb_hallucination, out_type='rgb')
            hallucination,_ = self.get_rgb_patch(rgb_feature_maps)
        else:
            raise NotImplementedError

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_xyz_lib.append(xyz_patch)
        self.patch_fusion_lib.append(hallucination)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        if self.args.main_modality == 'rgb':
            self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        elif self.args.main_modality == 'xyz':
            self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std

        if self.f_coreset < 1:
            if self.args.main_modality == 'rgb':
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                                n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                                eps=self.coreset_eps, lib='patch_rgb_lib',
                                                                coreset_dtype=self.coreset_dtype)
                self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            elif self.args.main_modality == 'xyz':
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                                n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                                eps=self.coreset_eps, lib='patch_xyz_lib',
                                                                coreset_dtype=self.coreset_dtype)
                self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_fusion_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_fusion_lib',
                                                            coreset_dtype=self.coreset_dtype)
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        if self.args.main_modality == 'rgb':
            rgb_feature_maps = self(rgb=sample[0], out_type='rgb')
            rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)
            with torch.no_grad():
                frgb = rgb_patch2.unsqueeze(0).cuda()
                xyz_hallucination = self.fusion.hallucination_generation(frgb).cpu()

            organized_pc_np = xyz_hallucination.squeeze().permute(1, 2, 0).numpy()  # 224 224 3
            unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)  # 224*224 3
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]  # 224*224
            unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)

            xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(
                xyz=unorganized_pc_no_zeros.contiguous(), out_type='xyz')

            hallucination = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        elif self.args.main_modality == 'xyz':
            unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
            xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(
                xyz=unorganized_pc_no_zeros.contiguous(), out_type='xyz')
            xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
            with torch.no_grad():
                fxyz = xyz_patch.unsqueeze(0).cuda()
                rgb_hallucination = self.fusion.hallucination_generation(fxyz).cpu()
                assert tuple(rgb_hallucination.shape) == tuple(sample[0].shape)
                rgb_feature_maps = self(rgb=rgb_hallucination, out_type='rgb')
            hallucination, _ = self.get_rgb_patch(rgb_feature_maps)

        hallucination = (hallucination - self.fusion_mean) / self.fusion_std
        dist_fusion = self.calculate_dist(hallucination, self.patch_fusion_lib)
        fusion_feat_size = (int(math.sqrt(hallucination.shape[0])), int(math.sqrt(hallucination.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(hallucination, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        if self.args.main_modality == 'rgb':
            rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
            dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)
            rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
            s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
            s = torch.tensor([[self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        elif self.args.main_modality == 'xyz':
            xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
            dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
            xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
            s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
            s = torch.tensor([[self.args.xyz_s_lambda * s_xyz,self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        if self.args.main_modality == 'rgb':
            rgb_feature_maps = self(rgb=sample[0], out_type='rgb')
            rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)
            with torch.no_grad():
                frgb = rgb_patch2.unsqueeze(0).cuda()
                xyz_hallucination = self.fusion.hallucination_generation(frgb).cpu()

            organized_pc_np = xyz_hallucination.squeeze().permute(1, 2, 0).numpy()  # 224 224 3
            unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)  # 224*224 3
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]  # 224*224
            unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)

            xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(
                xyz=unorganized_pc_no_zeros.contiguous(), out_type='xyz')

            hallucination = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

            xyz_patch = None
            self.compute_s_s_map(xyz_patch, rgb_patch, hallucination, mask, label,
                                 center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(),
                                 center_idx, rgb_path)

        elif self.args.main_modality == 'xyz':
            unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
            xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(
                xyz=unorganized_pc_no_zeros.contiguous(), out_type='xyz')
            xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
            with torch.no_grad():
                fxyz = xyz_patch.unsqueeze(0).cuda()
                rgb_hallucination = self.fusion.hallucination_generation(fxyz).cpu()
                assert tuple(rgb_hallucination.shape) == tuple(sample[0].shape)
                rgb_feature_maps = self(rgb=rgb_hallucination, out_type='rgb')
            hallucination, _ = self.get_rgb_patch(rgb_feature_maps)
            rgb_patch = None

        self.compute_s_s_map(xyz_patch, rgb_patch, hallucination, mask, label,
                             center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(),
                             center_idx, rgb_path)

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx, rgb_path):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std
        dist_fusion = self.calculate_dist(fusion_patch, self.patch_fusion_lib)
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        if self.args.main_modality == 'rgb':
            rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
            dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)
            rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
            s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
            s = torch.tensor([[self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.rgb_smap_lambda * s_map_rgb, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        elif self.args.main_modality == 'xyz':
            xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
            dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
            xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
            s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
            s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.fusion_s_lambda * s_fusion]])
            s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.fusion_smap_lambda * s_map_fusion],
                              dim=0).squeeze().reshape(2, -1).permute(1, 0)
        # print(s)
        s = torch.tensor(self.detect_fuser.score_samples(s))
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, self.gt_size, self.gt_size)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

        self.img_name.append(rgb_path)

        if self.args.save_seg_results:
            print(rgb_path)
            seg_save_path = rgb_path[0]
            seg_save_path = seg_save_path.replace('mvtec_3d','segmentation')
            seg_save_path = seg_save_path.replace('png', 'pt')

            slash = seg_save_path.rfind('/')
            dic = seg_save_path[:slash]
            if not os.path.exists(dic):
                os.makedirs(dic)
            torch.save(s_map, seg_save_path)


class DoubleRGBPointFeatures(Features):
    # img, resized_organized_pc, resized_depth_map_3channel, gt[:1], label, rgb_path
    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        # if self.args.use_depth:
        #     sample[0] = sample[2]
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        # __call__ rgb, xyz
        # rgb_features = B 768 28 28 xyz_features= B 1152 128 center= B G xyz oidx= B G M 1 cidx B G 1 inter B 1152 N
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)
        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)

        if self.args.save_feature_for_fusion:
            self.class_name = class_name
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)
            if not os.path.exists(self.args.save_path+'/train'):
                os.makedirs(self.args.save_path+'/train')
            if not os.path.exists(self.args.save_path + '/test'):
                os.makedirs(self.args.save_path + '/test')
            patch = torch.cat([xyz_patch, rgb_patch2], dim=1)  # 3136 768+1152
            torch.save(patch, os.path.join(self.args.save_path, 'train', class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

        if self.args.save_frgb_xyz:
            self.class_name = class_name
            if not os.path.exists(self.args.save_path_frgb_xyz):
                os.makedirs(self.args.save_path_frgb_xyz)
            if not os.path.exists(self.args.save_path_frgb_xyz+'/train'):
                os.makedirs(self.args.save_path_frgb_xyz+'/train')
                os.makedirs(self.args.save_path_frgb_xyz + '/train/frgb')
                os.makedirs(self.args.save_path_frgb_xyz + '/train/xyz')
            if not os.path.exists(self.args.save_path_frgb_xyz + '/test'):
                os.makedirs(self.args.save_path_frgb_xyz + '/test')
                os.makedirs(self.args.save_path_frgb_xyz + '/test/frgb')
                os.makedirs(self.args.save_path_frgb_xyz + '/test/xyz')
            organized_pc_to_save = sample[1].squeeze()  # 3 224 224
            assert tuple(organized_pc_to_save.shape) == (3, 224, 224)
            torch.save(rgb_patch2, os.path.join(self.args.save_path_frgb_xyz, 'train/frgb', class_name + str(self.ins_id2) + '_frgb.pt'))
            torch.save(organized_pc_to_save, os.path.join(self.args.save_path_frgb_xyz, 'train/xyz', class_name + str(self.ins_id2) + '_xyz.pt'))
            self.ins_id2 += 1

        if self.args.save_rgb_fxyz:
            if not os.path.exists(self.args.save_path_rgb_fxyz):
                os.makedirs(self.args.save_path_rgb_fxyz)
            if not os.path.exists(self.args.save_path_rgb_fxyz+'/train'):
                os.makedirs(self.args.save_path_rgb_fxyz+'/train')
                os.makedirs(self.args.save_path_rgb_fxyz + '/train/rgb')
                os.makedirs(self.args.save_path_rgb_fxyz + '/train/fxyz')
            if not os.path.exists(self.args.save_path_rgb_fxyz + '/test'):
                os.makedirs(self.args.save_path_rgb_fxyz + '/test')
                os.makedirs(self.args.save_path_rgb_fxyz + '/test/rgb')
                os.makedirs(self.args.save_path_rgb_fxyz + '/test/fxyz')

            xyz_patch2828 = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices, get_2828=True)
            rgb_to_save = sample[0].squeeze()
            assert tuple(xyz_patch2828.shape) == (784, 768)
            assert tuple(xyz_patch.shape) == (3136, 768)
            assert tuple(rgb_to_save.shape) == (3, 224, 224)

            torch.save(xyz_patch,
                       os.path.join(self.args.save_path_rgb_fxyz, 'train/fxyz', class_name + str(self.ins_id3) + '_hfxyz.pt'))
            torch.save(xyz_patch2828,
                       os.path.join(self.args.save_path_rgb_fxyz, 'train/fxyz', class_name + str(self.ins_id3) + '_lfxyz.pt'))
            torch.save(rgb_to_save,
                       os.path.join(self.args.save_path_rgb_fxyz, 'train/rgb', class_name + str(self.ins_id3) + '_rgb.pt'))
            self.ins_id3 += 1
        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)  # allsamples*N3136, T1152
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)  # allsamples*N784 T768

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_xyz_lib',
                                                            coreset_dtype=self.coreset_dtype)
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, lib='patch_rgb_lib',
                                                            coreset_dtype=self.coreset_dtype)
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def add_sample_to_late_fusion_mem_bank(self, sample):
        if self.args.use_depth:
            sample[0] = sample[1]
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T  # N784 T768

        # 2D dist, normalization because patch_xyz/rgb_lib has been concat and normalized in run coreset
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        # s=1 smap=1 1 224 224
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])

        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)  # 50176, 2

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def predict(self, sample, mask, label, rgb_path):
        unorganized_pc_no_zeros, nonzero_indices = organized_pc_to_unorganized_pc_no_zeros(sample)
        if self.args.use_depth:
            sample[0] = sample[1]
        # __call__ rgb, xyz
        # rgb_features = B 768 28 28 xyz_features= B 1152 128 center= B G xyz oidx= B G M 1 cidx B G 1 inter B 1152 N
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices)

        rgb_patch, rgb_patch2 = self.get_rgb_patch(rgb_feature_maps)

        if self.args.save_feature_for_fusion:
            patch_save = torch.cat([xyz_patch, rgb_patch2], dim=1)
            torch.save(patch_save, os.path.join(self.args.save_path, 'test', self.class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

        if self.args.save_frgb_xyz:
            organized_pc_to_save = sample[1].squeeze()  # 3 224 224
            torch.save(rgb_patch2, os.path.join(self.args.save_path_frgb_xyz, 'test/frgb', self.class_name + str(self.ins_id2) + '_frgb.pt'))
            torch.save(organized_pc_to_save, os.path.join(self.args.save_path_frgb_xyz, 'test/xyz', self.class_name + str(self.ins_id2) + '_xyz.pt'))
            self.ins_id2 += 1

        if self.args.save_rgb_fxyz:
            xyz_patch2828 = self.get_xyz_patch(xyz_feature_maps, interpolated_pc, nonzero_indices, get_2828=True)
            rgb_to_save = sample[0].squeeze()
            torch.save(xyz_patch,
                       os.path.join(self.args.save_path_rgb_fxyz, 'test/fxyz', self.class_name + str(self.ins_id3) + '_hfxyz.pt'))
            torch.save(xyz_patch2828,
                       os.path.join(self.args.save_path_rgb_fxyz, 'test/fxyz', self.class_name + str(self.ins_id3) + '_lfxyz.pt'))
            torch.save(rgb_to_save,
                       os.path.join(self.args.save_path_rgb_fxyz, 'test/rgb', self.class_name + str(self.ins_id3) + '_rgb.pt'))
            self.ins_id3 += 1

        self.compute_s_s_map(xyz_patch, rgb_patch, mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx, rgb_path)

    def compute_s_s_map(self, xyz_patch, rgb_patch, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, rgb_path):
        """
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        """

        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std  # normalization
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = self.calculate_dist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = self.calculate_dist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)  # 2 224 224 to 224 448 to 448 224
        # print(s)
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))

        s_map = s_map.view(1, 224, 224)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

        self.img_name.append(rgb_path)

        if self.args.save_seg_results:
            print(rgb_path)
            seg_save_path = rgb_path[0]
            seg_save_path = seg_save_path.replace('mvtec_3d','segmentation')
            seg_save_path = seg_save_path.replace('png', 'pt')

            slash = seg_save_path.rfind('/')
            dic = seg_save_path[:slash]
            if not os.path.exists(dic):
                os.makedirs(dic)
            torch.save(s_map, seg_save_path)

