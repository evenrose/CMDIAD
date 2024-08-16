import argparse
import torch
from torch.utils.data import DataLoader
import os
import time
import math
import sys
from pathlib import Path
import datetime
import multiprocessing
from torch.utils.tensorboard import SummaryWriter

from utils.utils import set_seeds, set_multithreading, load_model, save_model
import dataset
from models.hallucination_network import HallucinationCrossModalityNetwork, HallucinationRGBFeatureToXYZInputMLP, HallucinationFeatureToInputConv, HallucinationCrossModalityConv
from utils.misc import MetricLogger, SmoothedValue, all_reduce_mean
import utils.lr_sched as lr_sched
from models.hrnet import HRNet


def get_args_parser():
    parser = argparse.ArgumentParser('fdc_pre-training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--weight_decay', type=float, default=1.5e-6,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.002, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=False, action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--cpu_core_num', default=6)
    parser.add_argument('--AMP', default=False)
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--method_name', default=None)
    parser.add_argument('--dist_method', default='l2', type=str)
    parser.add_argument('--compared_with_norm_feature', default=False, action='store_true')
    parser.add_argument('--with_norm', default=True, type=bool)
    parser.add_argument('--mlp_depth', default=1, type=int)
    parser.add_argument('--train_method', choices=['HallucinationCrossModality',
                                                   'HallucinationCrossModalityConv',
                                                   'RGBFeatureToXYZInputMLP',
                                                   'XYZFeatureToRGBInputMLP',
                                                   'RGBFeatureToXYZInputConv',
                                                   'XYZFeatureToRGBInputConv',
                                                   'RGBInputToXYZFeatureHRNET',
                                                   'XYZInputToRGBFeatureHRNET'])
    parser.add_argument('--estimate_depth', default=False, action='store_true')

    parser.add_argument('--tensorboard_save', default='./tensorboard_logs')
    parser.add_argument('--sigmoid_loss', default=False, action='store_true')
    parser.add_argument('--c_hrnet', default=48, type=int)
    parser.add_argument('--rgb_backbone', default='dino', type=str)
    return parser


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.train_method == 'HallucinationCrossModality' or args.train_method == 'HallucinationCrossModalityConv':
            if args.rgb_backbone == 'dino_small':
                assert samples.shape[2] == 1152
            else:
                assert samples.shape[2] == 1536

            xyz_samples = samples[:, :, :768].to(device)
            rgb_samples = samples[:, :, 768:].to(device)
            distance_to_xyz_real, distance_to_rgb_real = model(xyz_samples, rgb_samples, args.sigmoid_loss, args.dist_method)
            loss_total = distance_to_xyz_real + distance_to_rgb_real
            loss_xyz = distance_to_xyz_real.item()
            loss_rgb = distance_to_rgb_real.item()
            loss_total_value = loss_total.item()

            loss_total = loss_total/accum_iter
            metric_logger.update(loss_xyz=loss_xyz)
            metric_logger.update(loss_rgb=loss_rgb)
            metric_logger.update(loss_total_value=loss_total_value)
            writer.add_scalar('train_loss/xyz', loss_xyz, epoch)
            writer.add_scalar('train_loss/rgb', loss_rgb, epoch)
            writer.add_scalar('train_loss/total', loss_total_value, epoch)
        elif args.train_method == 'RGBFeatureToXYZInputMLP' or args.train_method == 'RGBFeatureToXYZInputConv' or args.train_method == 'XYZFeatureToRGBInputMLP' or args.train_method == 'XYZFeatureToRGBInputConv':
            feature = samples.to(device)
            img = labels.to(device)
            loss_total = model(feature, img)
            loss_total_value = loss_total.item()
            loss_total = loss_total / accum_iter
            metric_logger.update(loss_total_value=loss_total_value)
            writer.add_scalar('train_loss/total', loss_total_value, epoch)
        elif args.train_method == 'RGBInputToXYZFeatureHRNET' or args.train_method == 'XYZInputToRGBFeatureHRNET':
            i = samples.to(device)
            f = labels.to(device)
            loss_total = model(i, f)
            loss_total_value = loss_total.item()
            loss_total = loss_total / accum_iter
            metric_logger.update(loss_total_value=loss_total_value)
            writer.add_scalar('train_loss/total', loss_total_value, epoch)

        if not math.isfinite(loss_total_value):
            print("Loss is {}, stopping training".format(loss_total_value))
            sys.exit(1)

        loss_total.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = all_reduce_mean(loss_total_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    multiprocessing.set_start_method('spawn')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed
    set_seeds(seed)

    # torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.train_method == 'RGBFeatureToXYZInputMLP' or args.train_method == 'RGBFeatureToXYZInputConv':
        dataset_train = dataset.FeatureToInputPreTrainTensorDataset(args.data_path+'/train', data_type='xyz_frgb')
        dataset_test = dataset.FeatureToInputPreTrainTensorDataset(args.data_path + '/test', data_type='xyz_frgb')

        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       multiprocessing_context='forkserver', num_workers=args.num_workers,
                                       pin_memory=args.pin_mem, prefetch_factor=6, drop_last=True)
        data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size,
                                      multiprocessing_context='forkserver', num_workers=args.num_workers,
                                      pin_memory=args.pin_mem, prefetch_factor=6, drop_last=False)
    elif args.train_method == 'XYZFeatureToRGBInputMLP' or args.train_method == 'XYZFeatureToRGBInputConv':
        dataset_train = dataset.FeatureToInputPreTrainTensorDataset(args.data_path + '/train', data_type='rgb_fxyz')
        dataset_test = dataset.FeatureToInputPreTrainTensorDataset(args.data_path + '/test', data_type='rgb_fxyz')

        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       multiprocessing_context='forkserver', num_workers=args.num_workers,
                                       pin_memory=args.pin_mem, prefetch_factor=6, drop_last=True)
        data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size,
                                      multiprocessing_context='forkserver', num_workers=args.num_workers,
                                      pin_memory=args.pin_mem, prefetch_factor=6, drop_last=False)

    elif args.train_method == 'RGBInputToXYZFeatureHRNET':
        dataset_train = dataset.InputToFeaturePreTrainTensorDataset(args.data_path + '/train', 'rgb_fxyz')
        dataset_test = dataset.InputToFeaturePreTrainTensorDataset(args.data_path + '/test', 'rgb_fxyz')
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size,
                                      num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    elif args.train_method == 'XYZInputToRGBFeatureHRNET':
        dataset_train = dataset.InputToFeaturePreTrainTensorDataset(args.data_path + '/train', 'xyz_frgb')
        dataset_test = dataset.InputToFeaturePreTrainTensorDataset(args.data_path + '/test', 'xyz_frgb')
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size,
                                      num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        dataset_train = dataset.PreTrainTensorDataset(args.data_path+'/train')
        dataset_test = dataset.PreTrainTensorDataset(args.data_path+'/test')

        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       multiprocessing_context='forkserver',num_workers=args.num_workers,
                                       pin_memory=args.pin_mem,prefetch_factor=1,drop_last=True)

        data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size,
                                      multiprocessing_context='forkserver', num_workers=args.num_workers,
                                      pin_memory=args.pin_mem, prefetch_factor=1, drop_last=False)

    test_loss_xyz = 0
    test_loss_rgb = 0
    test_loss_total = 0

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.train_method == 'HallucinationCrossModality':
        if args.rgb_backbone == 'dino_small':
            rgb_dim = 384
        else:
            rgb_dim = 768
        model = HallucinationCrossModalityNetwork(args, 768, rgb_dim=rgb_dim, mlp_depth=args.mlp_depth)
    elif args.train_method == 'RGBFeatureToXYZInputMLP':
        model = HallucinationRGBFeatureToXYZInputMLP(args, 768)
    elif args.train_method == 'RGBFeatureToXYZInputConv' or args.train_method == 'XYZFeatureToRGBInputConv':
        model = HallucinationFeatureToInputConv(args, 768)
    elif args.train_method == 'HallucinationCrossModalityConv':
        model = HallucinationCrossModalityConv(args, 768, 768)
    elif args.train_method == 'RGBInputToXYZFeatureHRNET' or args.train_method == 'XYZInputToRGBFeatureHRNET':
        model = HRNet(args.c_hrnet, 768, 0.1)
    else:
        raise NotImplementedError

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)

    load_model(args=args, model=model, optimizer=optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch,
            args=args)
        if args.output_dir and ((epoch+1) % 5 == 0):
            save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=None, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                for samples, labels in data_loader_test:
                    if args.train_method == 'HallucinationCrossModality' or args.train_method == 'HallucinationCrossModalityConv':
                        xyz_samples = samples[:, :, :768].to(device)
                        rgb_samples = samples[:, :, 768:].to(device)
                        distance_to_xyz_real, distance_to_rgb_real = model(xyz_samples, rgb_samples, args.sigmoid_loss, args.dist_method)
                        loss_total = distance_to_xyz_real + distance_to_rgb_real
                        test_loss_xyz += distance_to_xyz_real.item()
                        test_loss_rgb += distance_to_rgb_real.item()
                        test_loss_total = test_loss_xyz + test_loss_rgb
                    elif args.train_method == 'RGBFeatureToXYZInputMLP' or args.train_method == 'RGBFeatureToXYZInputConv' or args.train_method == 'XYZFeatureToRGBInputMLP' or args.train_method == 'XYZFeatureToRGBInputConv':
                        feature = samples.to(device)
                        img = labels.to(device)
                        loss_total = model(feature, img)
                        test_loss_total += loss_total.item()
                    elif args.train_method == 'RGBInputToXYZFeatureHRNET' or args.train_method == 'XYZInputToRGBFeatureHRNET':
                        i = samples.to(device)
                        f = labels.to(device)
                        loss_total = model(i, f)
                        test_loss_total += loss_total.item()
                    else:
                        raise NotImplementedError

                if args.train_method == 'HallucinationCrossModality' or args.train_method == 'HallucinationCrossModalityConv':
                    print('\ntest_loss_xyz: {:.4f}\n'.format(test_loss_xyz / len(data_loader_test)))
                    print('test_loss_rgb: {:.4f}\n'.format(test_loss_rgb / len(data_loader_test)))
                    writer.add_scalar('test_loss/xyz', test_loss_xyz / len(data_loader_test), epoch)
                    writer.add_scalar('test_loss/rgb', test_loss_rgb / len(data_loader_test), epoch)
                    test_loss_xyz = 0
                    test_loss_rgb = 0

                print('test_loss_total: {:.4f}\n'.format(test_loss_total / len(data_loader_test)))
                writer.add_scalar('test_loss/total', test_loss_total / len(data_loader_test), epoch)
                test_loss_total = 0

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    writer = SummaryWriter(args.tensorboard_save)

    cpu_num = int(args.cpu_core_num)
    set_multithreading(cpu_num)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)