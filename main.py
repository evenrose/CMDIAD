import argparse
from cmdiad_runner import CMDIAD
from dataset import eyecandies_classes, mvtec3d_classes
from utils.utils import set_multithreading
import pandas as pd
import os
import torch


def run_3d_ads(args):
    if args.dataset_type == 'eyecandies':
        classes = eyecandies_classes()
    elif args.dataset_type == 'mvtec3d':
        classes = mvtec3d_classes()

    METHOD_NAMES = [args.method_name]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_001_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        model = CMDIAD(args)
        model.fit(cls)
        image_rocaucs, pixel_rocaucs, au_pros, au_pros_001 = model.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)
        au_pros_001_df[cls.title()] = au_pros_001_df['Method'].map(au_pros_001)

        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1), 3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1), 3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1), 3)
    au_pros_001_df['Mean'] = round(au_pros_001_df.iloc[:, 1:].mean(axis=1), 3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))

    # print("\n\n##########################################################################")
    # print("############################ AU PRO 0.01 Results #########################")
    # print("##########################################################################\n")
    # print(au_pros_001_df.to_markdown(index=False))

    if args.save_results:
        with open("results/image_rocauc_results.md", "a") as tf:
            tf.write('\n\n'+args.experiment_note+'\n')
            tf.write(image_rocaucs_df.to_markdown(index=False))
        with open("results/pixel_rocauc_results.md", "a") as tf:
            tf.write('\n\n'+args.experiment_note+'\n')
            tf.write(pixel_rocaucs_df.to_markdown(index=False))
        with open("results/aupro_results.md", "a") as tf:
            tf.write('\n\n'+args.experiment_note+'\n')
            tf.write(au_pros_df.to_markdown(index=False))
        # with open("results/aupro_001_results.md", "a") as tf:
        #     tf.write('\n\n'+args.experiment_note+'\n')
        #     tf.write(au_pros_001_df.to_markdown(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--method_name', default='DINO+Point_MAE+Fusion', type=str,
                        choices=['DINO', 'Point_MAE', 'DINO+Point_MAE',
                                 'WithHallucination', 'WithHallucinationFromFeature'],
                        help='Anomaly detection modal name.')
    parser.add_argument('--max_sample', default=500, type=int,
                        help='Max sample number.')
    parser.add_argument('--memory_bank', default='multiple', type=str,
                        choices=["multiple"])
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str,
                        choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k',
                                 'vit_small_patch8_224_dino','vit_base_patch14_dinov2.lvd142m'],
                        help='Timm checkpoints name of RGB backbone.')
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str)
    parser.add_argument('--fusion_module_path', default='', type=str)

    parser.add_argument('--save_preds', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=1024, type=int,
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    parser.add_argument('--dataset_type', default='mvtec3d', type=str, choices=['mvtec3d', 'eyecandies'],
                        help='Dataset type for training or testing')
    parser.add_argument('--dataset_path', default='datasets/mvtec_3d', type=str,
                        help='Dataset store path')
    parser.add_argument('--xyz_s_lambda', default=1.0, type=float,
                        help='xyz_s_lambda')
    parser.add_argument('--xyz_smap_lambda', default=1.0, type=float,
                        help='xyz_smap_lambda')
    parser.add_argument('--rgb_s_lambda', default=0.1, type=float,
                        help='rgb_s_lambda')
    parser.add_argument('--rgb_smap_lambda', default=0.1, type=float,
                        help='rgb_smap_lambda')
    parser.add_argument('--fusion_s_lambda', default=1.0, type=float,
                        help='fusion_s_lambda')
    parser.add_argument('--fusion_smap_lambda', default=1.0, type=float,
                        help='fusion_smap_lambda')
    parser.add_argument('--share_s_lambda', default=1.0, type=float,
                        help='share_s_lambda')
    parser.add_argument('--share_smap_lambda', default=1.0, type=float,
                        help='non_share_smap_lambda')
    parser.add_argument('--non_share_s_lambda', default=1.0, type=float,
                        help='share_s_lambda')
    parser.add_argument('--non_share_smap_lambda', default=1.0, type=float,
                        help='non_share_smap_lambda')

    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--asy_memory_bank', default=None, type=int,
                        help='build an asymmetric memory bank for point clouds')
    parser.add_argument('--ocsvm_nu', default=0.5, type=float,
                        help='ocsvm nu')
    parser.add_argument('--ocsvm_maxiter', default=1000, type=int,
                        help='ocsvm maxiter')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')

    parser.add_argument('--img_process_method', default='cpu_v1', type=str)
    parser.add_argument('--cpu_core_num', default=6, type=int)
    parser.add_argument('--experiment_note', default='', type=str)
    parser.add_argument('--coreset_dtype', default='FP16', type=str)
    parser.add_argument('--similarity_only', default=False, action='store_true')
    parser.add_argument('--difference_only', default=False, action='store_true')
    parser.add_argument('--concat_only', default=False, action='store_true')
    parser.add_argument('--need_detection_head', default=False, type=bool)
    parser.add_argument('--train_with_validation', default=False, action='store_true')
    parser.add_argument('--dist_method_s', default='l2', type=str, choices=['l1', 'l2', 'cos_dist'])
    parser.add_argument('--dist_method_coreset', default='l2', type=str, choices=['l1', 'l2', 'cos_dist'])
    parser.add_argument('--main_modality', default='', type=str)

    parser.add_argument('--use_hn', default=False, action='store_true')
    parser.add_argument('--use_hn_conv', default=False, action='store_true')
    parser.add_argument('--use_hn_from_rgb_mlp', default=False, action='store_true')
    parser.add_argument('--use_hn_from_rgb_conv', default=False, action='store_true')
    parser.add_argument('--use_depth', default=False, action='store_true')
    parser.add_argument('--use_hrnet', default=False, action='store_true')

    parser.add_argument('--with_norm', default=True, type=bool)
    parser.add_argument('--rgb_size', default=224, type=int,
                        help='Images size for model')
    parser.add_argument('--xyz_size', default=224, type=int,
                        help='XYZ size for model')
    parser.add_argument('--gt_size', default=224, type=int,
                        help='gt size for model')

    parser.add_argument('--save_feature_for_fusion', default=False, action='store_true')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_path_frgb_xyz', type=str)
    parser.add_argument('--save_path_rgb_fxyz', type=str)
    parser.add_argument('--save_frgb_xyz', default=False, action='store_true')
    parser.add_argument('--save_rgb_fxyz', default=False, action='store_true')
    parser.add_argument('--save_results', default=True, type=bool)
    parser.add_argument('--c_hrnet', default=48, type=int)

    parser.add_argument('--save_raw_results', default=False, action='store_true')
    parser.add_argument('--save_seg_results', default=False, action='store_true')

    args = parser.parse_args()
    cpu_num = args.cpu_core_num
    set_multithreading(cpu_num)

    run_3d_ads(args)