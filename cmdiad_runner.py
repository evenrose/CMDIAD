import torch
from tqdm import tqdm
import os

from feature_extractors import multiple_features
from dataset import get_data_loader


class CMDIAD():
    def __init__(self, args):
        self.args = args
        self.rgb_size = args.rgb_size
        self.xyz_size = args.xyz_size
        self.gt_size = args.gt_size
        self.count = args.max_sample
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": multiple_features.RGBFeatures(args),
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'WithHallucination':
            self.methods = {"WithHallucination": multiple_features.RGBorXYZWithOneHallucination(args)}
        elif args.method_name == 'WithHallucinationFromFeature':
            self.methods = {"WithHallucinationFromFeature": multiple_features.RGBorXYZWithOneHallucinationFromFeature(args)}

    def fit(self, class_name):
        # (img, resized_organized_pc, resized_depth_map_3channel), label
        # img = B RGB3 224 224 resized_organized_pc= B XYZ3 224 224
        if self.args.train_with_validation:
            train_loader = get_data_loader("train_validation", class_name=class_name, rgb_size=self.rgb_size,
                                           xyz_size=self.xyz_size, gt_size=self.gt_size, args=self.args)
        else:
            train_loader = get_data_loader("train", class_name=class_name, rgb_size=self.rgb_size,
                                           xyz_size=self.xyz_size, gt_size=self.gt_size, args=self.args)

        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}', mininterval=2):
            for method in self.methods.values():
                method.add_sample_to_mem_bank(sample, class_name=class_name)
                flag += 1
            if flag > self.count:
                flag = 0
                break

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()

        if self.args.memory_bank == 'multiple':
            flag = 0
            for sample, _ in tqdm(train_loader, desc=f'Running late fusion for {method_name} on class {class_name}..',
                                  mininterval=2):
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break

            for method_name, method in self.methods.items():
                print(f'\n\nTraining Dicision Layer Fusion for {method_name} on class {class_name}...')
                method.run_late_fusion()

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        au_pros_001 = dict()
        # (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path
        test_loader = get_data_loader("test", class_name=class_name, rgb_size=self.rgb_size,
                                           xyz_size=self.xyz_size, gt_size=self.gt_size, args=self.args)
        path_list = []
        with torch.no_grad():
            for sample, mask, label, rgb_path in tqdm(test_loader, mininterval=1,
                                                      desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, mask, label, rgb_path)
                    path_list.append(rgb_path)

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            image_rocaucs[method_name] = round(method.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
            au_pros[method_name] = round(method.au_pro, 3)
            au_pros_001[method_name] = round(method.au_pro_001, 3)
            print(f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, '
                  f'{method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}, '
                  f'{method_name} AU-PRO-0.01: {method.au_pro_001:.3f}')
            # if self.args.save_preds:
            #     method.save_prediction_maps('./pred_maps', path_list)
        return image_rocaucs, pixel_rocaucs, au_pros, au_pros_001
