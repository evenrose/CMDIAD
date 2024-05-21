import numpy as np
import random
import torch
from torch import nn
from torchvision import transforms
from PIL import ImageFilter
import os
from pathlib import Path


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def set_multithreading(cpu_num: int = 8) -> None:
    # dynamic = 'True'
    # os.environ['OMP_DYNAMIC'] = dynamic
    # os.environ['MKL_DYNAMIC'] = dynamic
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def load_model(args, model, optimizer, loss_scaler=None):
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if args.train_stage != 'second':
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, without_opt=True):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        if loss_scaler is not None:
            to_save = {'model': model_without_ddp.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': epoch,
                       'scaler': loss_scaler.state_dict(),
                       'args': args}
        elif without_opt:
            to_save = {'model': model_without_ddp.state_dict(),
                       'epoch': epoch,
                       'args': args}
        else:
            to_save = {'model': model_without_ddp.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': epoch,
                       'args': args}
        torch.save(to_save, checkpoint_path)


class KNNGaussianBlur(torch.nn.Module):
    # maybe change to CUDA processing
    def __init__(self, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=radius)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map


class MlpBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        return x


class MlpModule(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, mlp_depth=1,):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.mlp_module = nn.ModuleList()
        for _ in range(mlp_depth):
            self.mlp_module.append(MlpBlock(in_features, hidden_features, out_features, act_layer))

    def forward(self, x):
        for module in self.mlp_module:
            x = module(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
