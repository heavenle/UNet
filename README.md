# Unet model(include Unet, Unet_2plus, Unet_3plus)

## Requirements
- PyTorch 1.7以上
- scikit-learn
- python-opencv
- tqdm
- yaml 

## 0. 前言
本代码主要实现Unet网络，Unet++网络和Unet+++网络。<br>

整体的代码框架为3个部分，分别是：
1. 参数配置部分（存储在config文件中）
2. 网络配置部分（存储在NetWorkRegistry部分）
3. 辅助工具部分（存储在utils文件中）
4. 启动脚本文件train.py和test.py

注意：
- 整体的代码只需要按照规则修改config.yml中的内容就可以进行训练，无需修改源码。<br>
- 配置代码采用了注册机制的方法进行整合，如果读者朋友想要自定义可以自行注册。

## 1. 环境配置参数简介
参数|介绍
---|---
cuda|是否使用cuda，填写示例：True或者False
gpu_ids|使用GPU的索引号，目前不支持多卡训练，填写示例：0
load_from|网络的与预训练权重，填写示例：'xxx/xxx.pth'或者false
work_dir|工作目录，后续结果都会保存在这个文件中，填写示例：'output'
epoch|训练周期，填写示例：20
batch_size|batch大小，填写示例：2
checkpoint_iter|每多少个周期保存权重，填写示例20
labels|英文标签，","前后不要有空格。填写示例：background,build
num_classes|标签数量，填写示例：2
arch|选择模型，填写示例：'UNet_3Plus'
arch_params|模型的输入参数，用字典的形式填写，填写示例：{in_channels: 3, n_classes: 2, feature_scale: 4, is_deconv: True, is_batchnorm: True}
deep_supervision|针对Unet3+设置的参数，是否启动深度监督，填写示例：False
resize_w|输入图像的宽，填写示例512
resize_h|输入图像的高，填写示例512
test_size|验证集的占有比例，填写示例：0.2
loss|loss方法，可以写多个，","前后不要有空格，填写示例：'FocalLoss,IOULoss'
loss_params|loss的配置参数，采用字典的形式，填写示例：{FocalLoss:{gamma: 2, alpha: [0.5, 0.5]}, IOULoss:{}}
optimizer|优化器方法，填写示例'sgd_optim'
optimizer_params|优化器的参数，采用字典的形式，填写示例{lr: 0.001, momentum: 0.9, weight_decay: 0.0001, nesterov: False}
scheduler|学习率迭代器，填写示例：'CosineAnnealingLR'
scheduler_params|学习率迭代器的参数，填写示例：{T_max: 200, eta_min: 0.00001}
data_type|数据加载的方法，填写示例："NormalDataLoader"
images_path|数据的图像位置，填写示例：'/xxx/xxx/images'
masks_path|数据的标签位置，填写示例：'/xxx/xxx/masks'
img_ext|图像数据的后缀：'.tif'
mask_ext|标签数据的后缀：'.tif'
num_workers|数据迭代器的工作数量，默认0，填写示例：0
test_path|测试集的路径，填写示例：'./demo/test_data'
test_img_ext|测试集的后缀，填写示例：'.tif'
load_from_to_test|测试时的网络权重路径，填写示例：'/xxx/xxx.pth'

## 2. 数据
- 输入数据的存储形式：
```
├── data_path
│   ├── images
│   │   └── 00ae65...
│   └── masks
│       └── 00ae65...            
├── ...

```
其中，图像数据可以为RGB数据，RGBA数据，灰度数据等，只需和网络输入通道一致。
标签数据目前仅支持单通道灰度数据，类别标签从0开始到类别数。
可以参考demo文件夹中train_data_format的存储形式。

## 3. 网路配置介绍
本代码的所有网络相关的内容都放入了NetWorkRegistry文件夹中。<br>
NetWorkRegistry文件夹包含5个子文件夹分别对应5个模块，分别是：
1. loader 数据加载模块。
2. loss 损失函数模块。
3. models 网路模型模块。
4. optimizer 优化器模块。
5. scheduler 学习率优化器模块。

### 3.1 loader数据加载模块
目前包含： = ["NormalDataLoader"] <br>

### 3.2 loss损失函数模块：
目前包含： = ["BCELoss",
           "CrossEntropyLoss",
           "FocalLoss",
           "IOULoss",
           "LovaszHingeLoss",
           "BCEDiceLoss",
           "MSSSIMLoss"]<br>
           
### 3.3 models网络模型模块
目前包含： = ["UNet", 
            "UNet_2Plus", 
            "UNet_3Plus", 
            "UNet_3Plus_DeepSup_CGM", 
            "UNet_3Plus_DeepSup"]<br>
            
### 3.4 optimizer优化器模块
目前包含： = ["sgd_optim",
            "adam_optim"]<br>
### 3.5 scheduler学习率优化模块
目前包含：= ["CosineAnnealingLR",
           "ReduceLROnPlateau",
           "MultiStepLR"]<br>

