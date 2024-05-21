# Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD

This repository is the official implementation of 
[Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD](). 


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
It should be placed under the `datasets` folder.

### Data Pre-processing
```Pre-processing
python utils/preprocessing.py --dataset_path datasets/mvtec_3d/ --num_process 6
```
>📋  It is recommended to use the default value for the path to the dataset to prevent problems in subsequent training and evaluation, but you can change the number of threads used according to your configuration. Please note that the pre-processing is performed in place.
### Checkpoints
| Purpose                               | Checkpoint                                                                                          |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| Point Clouds (PCs) feature extractor  | [Point-MAE](https://drive.google.com/file/d/1-wlRIz0GM8o6BuPTJz4kTt6c_z1Gh6LX/view?usp=sharing)     |
| RGB Images feature extractor          | [DINO](https://drive.google.com/file/d/17s6lwfxwG_nf1td6LXunL-LjRaX67iyK/view?usp=sharing)          |
| Feature-to-Feature network (main PCs) | [FtoF main PCs](https://drive.google.com/file/d/1Z2AkfPqenJEv-IdWhVdRcvVQAsJC4DxW/view?usp=sharing) |
| Feature-to-Input network (main PCs)   | [FtoI main PCs](https://drive.google.com/file/d/1Z2AkfPqenJEv-IdWhVdRcvVQAsJC4DxW/view?usp=sharing) |
| Input-to-Feature network (main PCs)   | [ItoF main PCs](https://drive.google.com/file/d/1Z2AkfPqenJEv-IdWhVdRcvVQAsJC4DxW/view?usp=sharing) |
| Feature-to-Feature network (main RGB) | [FtoF main RGB]()                                                                                   |
| Feature-to-Input network (main RGB)   | [FtoI main RGB]()                                                                                   |
| Input-to-Feature network (main RGB)   | [ItoF main RGB]()                                                                                   |
>📋  Please put all checkpoints in folder `checkpoints`. 

## Training

To train the models in the paper, run these commands:
### MTFI pipeline with Feature-to-Feature distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--cpu_core_num 6 \
--experiment_note <your_note> \
--save_feature_for_fusion \
--save_path datasets/patch_lib \
```
To train MTFI pipeline with Feature-to-Feature distillation network:
```
python hallucination_network_pretrain.py \
--warmup_epochs 10 \
--epochs 100 \
--accum_iter 1 \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/patch_lib \
--output_dir <your_output_dir_path> \
--cpu_core_num 6 \
--train_method HallucinationCrossModality \
--num_workers 2 \
```
>📋 For MTFI pipeline with Feature-to-Feature distillation network, PCs or RGB images as the main modality are trained simultaneously.

### MTFI pipeline with Feature-to-Input distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--cpu_core_num 6 \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```
For PCs as main modality.
```
python hallucination_network_pretrain.py \
--warmup_epochs 10 \
--epochs 100 \
--accum_iter 1 \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--cpu_core_num 6 \
--train_method XYZFeatureToRGBInputConv \
--num_workers 4 \
```
For RGB images as main modality.
```
python hallucination_network_pretrain.py \
--warmup_epochs 10 \
--epochs 100 \
--accum_iter 1 \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--cpu_core_num 6 \
--train_method RGBFeatureToXYZInputConv \
--num_workers 4 \
```
### MTFI pipeline with Input-to-Feature distillation network:
Similarly, you need to store the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--cpu_core_num 6 \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```

For PCs as main modality.
```
python -u hallucination_network_pretrain.py \
--warmup_epochs 10 \
--epochs 100 \
--accum_iter 1 \
--lr 0.0003 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--cpu_core_num 6 \
--train_method XYZInputToRGBFeatureHRNET \
--num_workers 4 \
--c_hrnet 128 \
--pin_mem \
```
For RGB images as main modality.
```
python -u hallucination_network_pretrain.py \
--warmup_epochs 10 \
--epochs 100 \
--accum_iter 1 \
--lr 0.0002 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--cpu_core_num 6 \
--train_method XYZInputToRGBFeatureHRNET \
--num_workers 4 \
--c_hrnet 192 \
--pin_mem \
```

## Evaluation

### Evaluate the model on MVTec 3D-AD with single and dual memory bank method
For single PCs memory bank:
```single PCs memory bank
python main.py \
--method_name Point_MAE \
--cpu_core_num 6 \
--experiment_note <your_note> \
```

>📋 For single RGB memory bank and dual memory bank, please replace `Point_MAE` with `DINO` and `DINO+Point_MAE`, respectively. 
> You can define the maximum number of threads with `--cpu_core_num` and leave your note through `--experiment_note`.
> The results are saved in the `results` folder.
> If you need to output the raw anomaly scores of sample classification to a file, add --save_raw_results.

### MTFI pipeline with Feature-to-Feature distillation network:
For PCs as main modality.
```MTFI PCs
python main.py \
--method_name WithHallucination \
--use_hn \
--main_modality xyz \
--fusion_module_path checkpoints/MTFI_FtoF_PCs.pth \
--cpu_core_num 6 \
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
--cpu_core_num 6 \
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
--cpu_core_num 6 \
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
--cpu_core_num 6 \
--experiment_note <your_note> \
```




## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 