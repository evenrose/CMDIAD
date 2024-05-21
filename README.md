# Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD

This repository is the official implementation of 
[Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD](). 

## Visualization of Some Prediction Results
![fig1](./figures/fig1.png)
## Requirements
We implement this repo with the following environment:

* Ubuntu 22.04
* CUDA 12.1
* Python 3.11

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Sometimes conda's version control will cause the installation failure. We recommend using venv or conda to create 
> a virtual environment and then use pip to install all packages.

## Dataset and Pre-trained Models
### Dataset
The `MVTec 3D-AD` dataset can be downloaded from  [MVTec3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). 
It should be unzipped and placed under the `datasets` folder.

### Data Pre-processing
```Pre-processing
python utils/preprocessing.py --dataset_path datasets/mvtec_3d/ 
```
>📋  It is recommended to use the default value for the path to the dataset to prevent problems in subsequent training and evaluation, but you can change the number of threads used according to your configuration. Please note that the pre-processing is performed in place.
### Checkpoints
| Purpose                               | Checkpoint                                                                                          |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| Point Clouds (PCs) feature extractor  | [Point-MAE](https://drive.google.com/file/d/1CCiYO9MazSIRpA4Q5_lGWuAYPAkj7X7w/view?usp=sharing)     |
| RGB Images feature extractor          | [DINO](https://drive.google.com/file/d/1fOEhASxuygcP-vnnrn1AlzFXbnEC-0CK/view?usp=sharing)          |
| Feature-to-Feature network (main PCs) | [MTFI_FtoF_PCs](https://drive.google.com/file/d/1SzQgPsLLxEYtzYOCs0YYh1GYhZG776iF/view?usp=sharing) |
| Feature-to-Input network (main PCs)   | [MTFI_FtoI_PCs](https://drive.google.com/file/d/1LPl6bAHrJiLdY-w0vwNWeiyai4L7amMr/view?usp=sharing) |
| Input-to-Feature network (main PCs)   | [MTFI_ItoF_PCs](https://drive.google.com/file/d/1hD1J8XMlpelRYbOFwgkGHtJ05XsFG1Qk/view?usp=sharing) |
| Feature-to-Feature network (main RGB) | [MTFI_FtoF_RGB](https://drive.google.com/file/d/1N6QHaD4KhUy04C98jbg9m9hDjOu3YVJk/view?usp=sharing) |
| Feature-to-Input network (main RGB)   | [MTFI_FtoI_RGB](https://drive.google.com/file/d/1Xkpn7sISaoz4I63DimvRBc2kknw2dZRr/view?usp=sharing) |
| Input-to-Feature network (main RGB)   | [MTFI_ItoF_RGB](https://drive.google.com/file/d/1QqQccrk_whV0shnphSSJyMVPijlpKjcz/view?usp=sharing) |
>📋  Please put all checkpoints in folder `checkpoints`. 

## Training

To train the models in the paper, run these commands:
### MTFI pipeline with Feature-to-Feature distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--experiment_note <your_note> \
--save_feature_for_fusion \
--save_path datasets/patch_lib \
```
To train MTFI pipeline with Feature-to-Feature distillation network:
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/patch_lib \
--output_dir <your_output_dir_path> \
--train_method HallucinationCrossModality \
--num_workers 2 \
```
>📋 For MTFI pipeline with Feature-to-Feature distillation network, PCs or RGB images as the main modality are trained simultaneously.
> You can define the maximum number of threads with `--cpu_core_num` and leave your note through `--experiment_note`.
> The results are saved in the `results` folder.
> If you need to output the raw anomaly scores at image or pixel level to a file, add `--save_raw_results` or `--save_seg_results`.

### MTFI pipeline with Feature-to-Input distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```
For PCs as main modality.
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--train_method XYZFeatureToRGBInputConv \
```
For RGB images as main modality.
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--train_method RGBFeatureToXYZInputConv \
```
### MTFI pipeline with Input-to-Feature distillation network:
Similarly, you need to store the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```

For PCs as main modality.
```
python -u hallucination_network_pretrain.py \
--lr 0.0003 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--train_method XYZInputToRGBFeatureHRNET \
--c_hrnet 128 \
--pin_mem \
```
For RGB images as main modality.
```
python -u hallucination_network_pretrain.py \
--lr 0.0002 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--train_method XYZInputToRGBFeatureHRNET \
--c_hrnet 192 \
--pin_mem \
```

## Evaluation

### Evaluate the model on MVTec 3D-AD with single and dual memory bank method
For single PCs memory bank:
```single PCs memory bank
python main.py \
--method_name Point_MAE \
--experiment_note <your_note> \
```

>📋 For single RGB memory bank and dual memory bank, please replace `Point_MAE` with `DINO` and `DINO+Point_MAE`, respectively.

### MTFI pipeline with Feature-to-Feature distillation network:
For PCs as main modality.
```MTFI PCs
python main.py \
--method_name WithHallucination \
--use_hn \
--main_modality xyz \
--fusion_module_path checkpoints/MTFI_FtoF_PCs.pth \
--experiment_note <your_note> \
```

>📋 For RGB images as main modality, please replace `xyz` with `rgb` for `--main_modality` and give the new checkpoint path `checkpoints/MTFI_FtoF_RGB.pth` to the model.

### MTFI pipeline with Feature-to-Input distillation network:
For PCs as main modality.
```
python main.py \
--method_name WithHallucinationFromFeature \
--use_hn_from_rgb_conv \
--main_modality xyz \
--fusion_module_path checkpoints/MTFI_FtoI_PCs.pth \
--experiment_note <your_note> \
```

>📋 For RGB images as main modality, replace `xyz` with `rgb` and give model the new checkpoint path.

### MTFI pipeline with Input-to-Feature distillation network:
For PCs as main modality.
```
python main.py \
--method_name WithHallucination \
--use_hrnet \
--main_modality xyz \
--c_hrnet 128 \
--fusion_module_path checkpoints/MTFI_ItoF_PCs.pth \
--experiment_note <your_note> \
```

For RGB images as main modality.
```
python main.py \
--method_name WithHallucination \
--use_hrnet \
--main_modality rgb \
--c_hrnet 192 \
--fusion_module_path checkpoints/MTFI_ItoF_RGB.pth \
--experiment_note <your_note> \
```

## Citation
If you think this repository is helpful for your project, please use the following.
```

```
## Acknowledgement
We appreciate the following github repos for their valuable code:
- [M3DM](https://github.com/nomewang/M3DM/)
- [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS)
- [Shape-Guided](https://github.com/jayliu0313/Shape-Guided)