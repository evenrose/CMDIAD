import torch
import torch.nn as nn
from utils.utils import MlpModule


def feature_reshape(feature):
    feature = feature.permute(0, 2, 1)
    feature = feature.reshape(feature.shape[0], feature.shape[1], 56, 56)
    return feature


def feature_reshape_back(feature):
    feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
    feature = feature.permute(0, 2, 1)
    return feature


class HallucinationCrossModalityNetwork(nn.Module):
    def __init__(self, args, xyz_dim, rgb_dim, hidden_ratio=2.5, mlp_depth=1):
        super().__init__()
        self.args = args
        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim

        self.xyz_norm = nn.LayerNorm(xyz_dim)
        self.xyz_mlp = MlpModule(in_features=xyz_dim, hidden_features=int(xyz_dim * hidden_ratio), out_features=self.rgb_dim, act_layer=nn.GELU, mlp_depth=mlp_depth)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = MlpModule(in_features=rgb_dim, hidden_features=int(rgb_dim * hidden_ratio), out_features=self.xyz_dim, act_layer=nn.GELU, mlp_depth=mlp_depth)

        self.sig = nn.Sigmoid()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def hallucination_generation(self, xyz_feature=None, rgb_feature=None, out_type='Train'):
        # 1 3136 768, 1 3136 768 in
        if out_type == 'train':
            xyz_feature_hallucination = self.rgb_mlp(self.rgb_norm(rgb_feature))
            rgb_feature_hallucination = self.xyz_mlp(self.xyz_norm(xyz_feature))
            return xyz_feature_hallucination, rgb_feature_hallucination
        elif out_type == 'xyz':
            xyz_feature_hallucination = self.rgb_mlp(self.rgb_norm(rgb_feature))
            return xyz_feature_hallucination
        elif out_type == 'rgb':
            rgb_feature_hallucination = self.xyz_mlp(self.xyz_norm(xyz_feature))
            return rgb_feature_hallucination

    def forward(self, xyz_feature, rgb_feature, sigmoid, dist_method='cos_dist'):
        xyz_feature_hallucination, rgb_feature_hallucination = self.hallucination_generation(xyz_feature, rgb_feature, 'train')
        assert len(xyz_feature_hallucination.shape) == 3
        assert len(xyz_feature.shape) == 3
        assert xyz_feature.shape[2] == xyz_feature_hallucination.shape[2]
        if dist_method == 'cos_dist':
            cosine_distance_to_xyz_real = (1 - torch.cosine_similarity(xyz_feature_hallucination, xyz_feature, dim=2))
            cosine_distance_to_xyz_real = torch.sum(cosine_distance_to_xyz_real) / cosine_distance_to_xyz_real.shape[0]
            cosine_distance_to_rgb_real = (1 - torch.cosine_similarity(rgb_feature_hallucination, rgb_feature, dim=2))
            cosine_distance_to_rgb_real = torch.sum(cosine_distance_to_rgb_real) / cosine_distance_to_rgb_real.shape[0]
            return cosine_distance_to_xyz_real, cosine_distance_to_rgb_real
        elif dist_method == 'l2':
            l2_distance_to_xyz_real = torch.linalg.norm(xyz_feature_hallucination - xyz_feature, dim=2)
            l2_distance_to_xyz_real = torch.sum(l2_distance_to_xyz_real) / l2_distance_to_xyz_real.shape[0]
            l2_distance_to_rgb_real = torch.linalg.norm(rgb_feature_hallucination - rgb_feature, dim=2)
            l2_distance_to_rgb_real = torch.sum(l2_distance_to_rgb_real) / l2_distance_to_rgb_real.shape[0]
            return l2_distance_to_xyz_real, l2_distance_to_rgb_real
        elif dist_method == 'smooth_l1':
            l1_distance_to_xyz_real = self.smooth_l1(xyz_feature_hallucination, xyz_feature)
            l1_distance_to_xyz_real = torch.sum(l1_distance_to_xyz_real) / l1_distance_to_xyz_real.shape[0]
            l1_distance_to_rgb_real = self.smooth_l1(rgb_feature_hallucination, rgb_feature)
            l1_distance_to_rgb_real = torch.sum(l1_distance_to_rgb_real) / l1_distance_to_rgb_real.shape[0]
            return l1_distance_to_xyz_real, l1_distance_to_rgb_real


class HallucinationCrossModalityConv(nn.Module):
    def __init__(self, args, xyz_dim, rgb_dim):
        super().__init__()
        self.args = args
        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim

        self.xyz_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.xyz_dim, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )

        self.rgb_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.rgb_dim, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )

        self.sig = nn.Sigmoid()

    def hallucination_generation(self, xyz_feature, rgb_feature, out_type):
        # 1 3136 768, 1 3136 768 in
        if out_type == 'train':
            xyz_feature = feature_reshape(xyz_feature)  # 1 768 56 56
            rgb_feature = feature_reshape(rgb_feature)
            xyz_feature_hallucination = self.rgb_conv(rgb_feature)  # 1 768 56 56
            rgb_feature_hallucination = self.xyz_conv(xyz_feature)
            xyz_feature_hallucination = feature_reshape_back(xyz_feature_hallucination)
            rgb_feature_hallucination = feature_reshape_back(rgb_feature_hallucination)
            return xyz_feature_hallucination, rgb_feature_hallucination
        elif out_type == 'xyz':
            rgb_feature = feature_reshape(rgb_feature)
            xyz_feature_hallucination = self.rgb_conv(rgb_feature)
            xyz_feature_hallucination = feature_reshape_back(xyz_feature_hallucination)
            return xyz_feature_hallucination
        elif out_type == 'rgb':
            xyz_feature = feature_reshape(xyz_feature)
            rgb_feature_hallucination = self.xyz_conv(xyz_feature)
            rgb_feature_hallucination = feature_reshape_back(rgb_feature_hallucination)
            return rgb_feature_hallucination

    def forward(self, xyz_feature, rgb_feature, sigmoid, dist_method):
        xyz_feature_hallucination, rgb_feature_hallucination = self.hallucination_generation(xyz_feature, rgb_feature, 'train')
        assert tuple(xyz_feature_hallucination.shape[1:]) == (3136, 768)
        if sigmoid is True:
            distance_to_xyz_real = torch.linalg.norm(self.sig(xyz_feature_hallucination) - self.sig(xyz_feature), dim=2)
            distance_to_xyz_real = torch.sum(distance_to_xyz_real) / distance_to_xyz_real.shape[0]
            distance_to_rgb_real = torch.linalg.norm(self.sig(rgb_feature_hallucination) - self.sig(rgb_feature), dim=2)
            distance_to_rgb_real = torch.sum(distance_to_rgb_real) / distance_to_rgb_real.shape[0]
            return distance_to_xyz_real, distance_to_rgb_real
        else:
            l2_distance_to_xyz_real = torch.linalg.norm(xyz_feature_hallucination - xyz_feature, dim=2)
            l2_distance_to_xyz_real = torch.sum(l2_distance_to_xyz_real) / l2_distance_to_xyz_real.shape[0]
            l2_distance_to_rgb_real = torch.linalg.norm(rgb_feature_hallucination - rgb_feature, dim=2)
            l2_distance_to_rgb_real = torch.sum(l2_distance_to_rgb_real) / l2_distance_to_rgb_real.shape[0]
            return l2_distance_to_xyz_real, l2_distance_to_rgb_real


class HallucinationRGBFeatureToXYZInputMLP(nn.Module):
    def __init__(self, args, rgb_dim):
        super().__init__()
        self.args = args
        if args.estimate_depth:
            out_dim = 1
        else:
            out_dim = 3
        self.rgb_dim = rgb_dim
        self.rgb_norm = nn.LayerNorm(rgb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.rgb_dim, 1152),
            nn.GELU(),
            nn.Linear(1152, 384),
            nn.GELU(),
            nn.Linear(384, 96),
            nn.GELU(),
            nn.Linear(96, out_dim)
        )

    def hallucination_generation(self, x):
        x = self.rgb_norm(x)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], 56, 56)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic')
        return x

    def forward(self, rgb_feature, xyz):
        # frgb B 3136 768 xyz B 3 224 224 xyz_hallucination B 3136 3/1
        rgb_feature = rgb_feature.reshape(rgb_feature.shape[0], rgb_feature.shape[1], -1)
        xyz_hallucination = self.hallucination_generation(rgb_feature)

        l2_distance_to_xyz_real = torch.linalg.norm(xyz_hallucination - xyz, dim=1)
        l2_distance_to_xyz_real = torch.sum(l2_distance_to_xyz_real) / l2_distance_to_xyz_real.shape[0]
        return l2_distance_to_xyz_real


class HallucinationFeatureToInputConv(nn.Module):
    def __init__(self, args=None, dim=768):
        super().__init__()
        self.args = args

        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.act = nn.ReLU(inplace=True)

    def hallucination_generation(self, feature):
        feature = feature.transpose(1, 2)
        feature = feature.reshape(feature.shape[0], feature.shape[1], 56, 56)
        hallucination = self.conv1(feature)
        hallucination = nn.functional.interpolate(hallucination, size=(224, 224), mode='bicubic')
        hallucination = self.conv2(hallucination)
        hallucination = self.act(hallucination)
        hallucination = self.conv3(hallucination)
        hallucination = self.act(hallucination)
        hallucination = self.conv4(hallucination)
        return hallucination

    def forward(self, feature, img):
        # frgb B 3136 768 xyz B 3 224 224 xyz_hallucination B 3136 3/1
        hallucination = self.hallucination_generation(feature)
        # assert xyz_hallucination.shape[1:] == (1, 224, 224)
        assert hallucination.shape[1:] == (3, 224, 224)
        assert img.shape[1:] == (3, 224, 224)
        distance = torch.linalg.norm(hallucination - img, dim=1)
        distance = torch.sum(distance) / distance.shape[0]
        return distance
