import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import random_projection, linear_model
from sklearn.metrics import roc_auc_score
import math
import cupy as cp
from cupyx.scipy.spatial import distance


from models.models import Model
from models.pointnet2_utils import interpolating_points
from utils.utils import set_seeds, KNNGaussianBlur
from utils.au_pro_util import calculate_au_pro
from models.hallucination_network import HallucinationCrossModalityNetwork, HallucinationRGBFeatureToXYZInputMLP, HallucinationFeatureToInputConv, HallucinationCrossModalityConv
from models.hrnet import HRNet


class Features(torch.nn.Module):
    def __init__(self, args, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
            device=self.device,
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group
        )
        self.deep_feature_extractor.to(self.device)

        self.args = args
        self.class_name = None
        self.rgb_size = args.rgb_size
        self.xyz_size = args.xyz_size
        self.gt_size = args.gt_size

        self.f_coreset = args.f_coreset
        self.coreset_eps = args.coreset_eps

        self.coreset_dtype = args.coreset_dtype
        self.class_name = None

        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_xyz_lib = []
        self.patch_rgb_lib = []
        self.patch_fusion_lib = []
        self.patch_lib = []
        self.patch_share_lib = []
        self.patch_non_share_lib = []
        self.random_state = args.random_state

        self.xyz_dim = 0
        self.rgb_dim = 0

        self.xyz_mean = 0
        self.xyz_std = 0
        self.rgb_mean = 0
        self.rgb_std = 0
        self.fusion_mean = 0
        self.fusion_std = 0

        self.share_mean = 0
        self.share_std = 0
        self.non_share_mean = 0
        self.non_share_std = 0

        self.average = torch.nn.AvgPool2d(3, stride=1)  # torch.nn.AvgPool2d(1, stride=1) #
        self.resize28 = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.resize56 = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.au_pro_001 = 0
        self.ins_id = 0
        self.ins_id2 = 0
        self.ins_id3 = 0
        self.rgb_layernorm = torch.nn.LayerNorm(768, elementwise_affine=False)

        if self.args.use_hn:
            self.fusion = HallucinationCrossModalityNetwork(args, 768, 768, hidden_ratio=2.5)
            self.fusion.cuda()
        if self.args.use_hn_conv:
            self.fusion = HallucinationCrossModalityConv(args, 768, 768)
        if self.args.use_hn_from_rgb_mlp:
            self.fusion = HallucinationRGBFeatureToXYZInputMLP(args, 768)
        if self.args.use_hn_from_rgb_conv:
            self.fusion = HallucinationFeatureToInputConv(args, 768)
            self.fusion.cuda()
        if self.args.use_hrnet:
            self.fusion = HRNet(args.c_hrnet, 768, 0.1)
            self.fusion.cuda()

        if args.fusion_module_path != '':
            ckpt = torch.load(args.fusion_module_path)['model']
            incompatible = self.fusion.load_state_dict(ckpt)
            print('[Fusion Block]', incompatible)
            self.fusion.eval()

        self.detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu, max_iter=args.ocsvm_maxiter)
        self.seg_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu, max_iter=args.ocsvm_maxiter)

        self.s_lib = []
        self.s_map_lib = []

        self.img_name = []
        self.save_num = 0

    def __call__(self, rgb=None, xyz=None, out_type="rgb+xyz"):
        if out_type == "rgb+xyz":
            # Extract the desired feature maps using the backbone model.
            # __call__ rgb, xyz in 1 3 224 224, 1 3 224*224-zero
            rgb = rgb.to(self.device)
            xyz = xyz.to(self.device)
            with torch.no_grad():
                # rgb_features = B 768 28 28 xyz_features= B 1152 128 center= B G xyz oriidx= B G M 1 center_idx B G 1
                rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)

            interpolate = True
            if interpolate:
                # upsampled points data, [B, D', N] B T1152 N224*224-zero out
                # xyz = 1 3 N224*224-zero center = B G 3 to B 3 G xyz_feature_maps=B T1152 G128 in B T1152 N224*224-zero out
                interpolated_feature_maps = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feature_maps).to("cpu")

            xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]
            rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]

            if interpolate:
                return rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps
            else:
                return rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx
        elif out_type == 'rgb':
            rgb = rgb.to(self.device)
            with torch.no_grad():
                rgb_feature_maps = self.deep_feature_extractor(rgb=rgb, out_type=out_type)
            rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]
            return rgb_feature_maps
        elif out_type == 'xyz':
            xyz = xyz.to(self.device)
            with torch.no_grad():
                xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb=None, xyz=xyz, out_type=out_type)
            interpolated_feature_maps = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feature_maps).to("cpu")
            xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]
            return xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps

    def get_rgb_patch(self, rgb_feature_maps):
        rgb_patch = torch.cat(rgb_feature_maps, 1)  # 1 T768 28 28
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T  # N784 T768
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))  # sqrt(B*N)
        # B*N784 T768 in to T768 B*N784 to B =1 T768 28 28 to T768 56 56
        rgb_patch2 = self.resize56(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))  # 768 56 56
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T  # 3136 768
        return rgb_patch, rgb_patch2

    def get_xyz_patch(self, xyz_feature_maps, interpolated_pc, nonzero_indices, get_2828=False):
        xyz_patch = torch.cat(xyz_feature_maps, 1)  # xyz_patch= 1 1152 128
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.xyz_size * self.xyz_size),
                                     dtype=xyz_patch.dtype)  # 1 1152 N224*224
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc  # only pick non-zero points
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.xyz_size, self.xyz_size)
        # 1 1152 800 800 in 1, 1152, 798, 798 to 1, 1152, 56, 56 out
        xyz_patch_full_resized = self.resize56(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T  # N3136, T1152

        if get_2828:
            xyz_patch_full_resized2 = self.resize28(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T  # 3136 1152
            return xyz_patch2
        else:
            return xyz_patch

    def calculate_dist(self, single_patch, patch_lib):
        assert len(single_patch.shape) == 2
        assert len(patch_lib.shape) == 2
        if self.args.dist_method_s == 'l2':
            dist = torch.cdist(single_patch, patch_lib)
        # elif self.args.dist_method_s == 'l1':
        #     dist = torch.cdist(single_patch, patch_lib, p=1)
        elif self.args.dist_method_s == 'l1':
            single_patch_gpu = cp.asarray(single_patch)
            patch_lib_gpu = cp.asarray(patch_lib)
            dist = distance.cdist(single_patch, patch_lib, 'minkowski',p=1.).toDlpack()
            dist = torch.utils.dlpack.from_dlpack(dist).cpu()
        elif self.args.dist_method_s == 'cos_dist':
            single_patch_gpu = cp.asarray(single_patch)
            patch_lib_gpu = cp.asarray(patch_lib)
            dist = distance.cdist(single_patch, patch_lib, 'cosine').toDlpack()
            dist = torch.utils.dlpack.from_dlpack(dist).cpu()
        else:
            raise NotImplementedError
        return dist

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label, rgb_path):
        raise NotImplementedError

    def add_sample_to_late_fusion_mem_bank(self, sample):
        raise NotImplementedError

    def interpolate_points(self, rgb, xyz):
        with torch.no_grad():
            rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)
        return xyz_feature_maps, center, xyz

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        raise NotImplementedError

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        min_val, min_idx = torch.min(dist, dim=1)  # find the min distance of each point between point in patch and bank
        s_idx = torch.argmax(min_val)  # find the point with the largest distance in patch: most abnormal
        # print('dist',dist)
        # print('min_val', min_val)
        s_star = torch.max(min_val)


        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch 1 1 token

        if modal == 'xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = self.calculate_dist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal == 'rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = self.calculate_dist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        elif modal == 'share':
            m_star = self.patch_share_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = self.calculate_dist(m_star, self.patch_share_lib)  # find knn to m_star pt.1
        elif modal == 'non_share':
            m_star = self.patch_non_share_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = self.calculate_dist(m_star, self.patch_non_share_lib)  # find knn to m_star pt.1
        elif modal == 'fusion':
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = self.calculate_dist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1

        # pick top 3 similar features in M (itself included)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2 1 3

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1)
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.

        # equation 7 from the paper
        if modal == 'xyz':
            # distance of abnormal point in patch and the other neighbors
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'share':
            m_star_knn = torch.linalg.norm(m_test - self.patch_share_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'non_share':
            m_star_knn = torch.linalg.norm(m_test - self.patch_non_share_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'fusion':
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))  # sqrt of Token
        # kind like softmax of distance between the abnormal point & it's related feature in M, and neighbors' distance
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        # print('w', w)
        # print('s_star', s_star)
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)  # 1, 1 dim dim
        s_map = torch.nn.functional.interpolate(s_map, size=(self.gt_size, self.gt_size), mode='bilinear')  # 1 1 224 224
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        raise NotImplementedError

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.img_name = np.stack(self.img_name)
        # print('\nimage_preds is:\n')
        # print(self.image_preds)
        # print('\nimage_labels is:\n')
        # print(self.image_labels)
        # print('\nimage_name is:\n')
        # print(self.img_name)
        # print(self.image_preds.shape)
        # assert len(self.image_preds.shape) == 2
        if self.args.save_raw_results:
            txt_to_save = np.concatenate((self.image_preds, self.image_labels, self.img_name), axis=1)
            np.savetxt(f'./visualization/{self.args.experiment_note}/{self.class_name}_raw_results.csv', txt_to_save, delimiter=',', fmt="%s")


        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        self.au_pro_001, _ = calculate_au_pro(self.gts, self.predictions, 0.01)

    # def save_prediction_maps(self, output_path, rgb_path, save_num=5):
    #     for i in range(max(save_num, len(self.predictions))):
    #         # fig = plt.figure(dpi=300)
    #         fig = plt.figure()
    #
    #         ax3 = fig.add_subplot(1, 3, 1)
    #         gt = plt.imread(rgb_path[i][0])
    #         ax3.imshow(gt)
    #
    #         ax2 = fig.add_subplot(1, 3, 2)
    #         im2 = ax2.imshow(self.gts[i], cmap=plt.cm.gray)
    #
    #         ax = fig.add_subplot(1, 3, 3)
    #         im = ax.imshow(self.predictions[i], cmap=plt.cm.jet)
    #
    #         class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
    #         if not os.path.exists(class_dir):
    #             os.mkdir(class_dir)
    #
    #         ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
    #         if not os.path.exists(ad_dir):
    #             os.mkdir(ad_dir)
    #
    #         plt.savefig(
    #             os.path.join(ad_dir, str(self.image_preds[i]) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))

    def run_late_fusion(self):
        # if mem=2 s ls B (1, mem)  s_lib ls B (50176, mem)
        self.s_lib = torch.cat(self.s_lib, 0)  # number of sample, 3
        self.s_map_lib = torch.cat(self.s_map_lib, 0)

        self.detect_fuser.fit(self.s_lib)
        self.seg_fuser.fit(self.s_map_lib)

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, coreset_dtype='FP16', force_cpu=False, lib=''):
        # maybe need to improve
        # xyz B*N3136, T1152 rgb B*N784 T768
        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps, random_state=self.random_state)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))  # B*N3136, T(1152-unknown)

            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        # add initial point to coreset, zlib = N T-unknown
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        # calculate distance between each point in zlib and initial point N 1
        if self.args.dist_method_coreset == 'l2':
            min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        elif self.args.dist_method_coreset == 'l1':
            min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True, ord=1)
        elif self.args.dist_method_coreset == 'dot':
            min_distances = torch.sum(torch.mul(z_lib, last_item), dim=1, keepdim=True)
        elif self.args.dist_method_coreset == 'cos_dist':
            min_distances = 1 - torch.cosine_similarity(z_lib, last_item)
        else:
            raise NotImplementedError

        if coreset_dtype == 'FP16':
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        elif coreset_dtype == 'TF32':
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            raise NotImplementedError

        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1), desc=f'Extracting coreset for: {lib}', mininterval=2):
            # to find n points in Mc (n-1 left)
            # calculate distance between each point in M and picked point in Mc
            if self.args.dist_method_coreset == 'l2':
                distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            # elif self.args.dist_method_coreset == 'dot':
            #     distances = torch.sum(torch.mul(z_lib, last_item), dim=1, keepdim=True)
            elif self.args.dist_method_coreset == 'l1':
                distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True, ord=1)
            elif self.args.dist_method_coreset == 'cos_dist':
                distances = 1 - torch.cosine_similarity(z_lib, last_item)
            # find the smaller one of distance between each point and current point or last point
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            # find the furthest point
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]  # update last point
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))

        if coreset_dtype == 'TF32':
            torch.backends.cuda.matmul.allow_tf32 = False

        return torch.stack(coreset_idx)
