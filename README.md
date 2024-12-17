# Enhancing UAV Object Detection through Multi-Scale Deformable Convolutions and Adaptive Fusion Attention

## 1. Introduction

MSDC-DETR is an innovative model designed specifically for UAV image object detection. This model combines Multi-Scale Deformable Convolutions (MSDC) and Adaptive Fusion Attention Module (AFAM) to improve the detection performance of small objects in complex backgrounds. By dynamically adjusting the shape and size of convolution kernels, MSDC-DETR can better adapt to the geometric variations of different targets, aiming to enhance detection accuracy in UAV image object detection and unlock its potential for practical applications in the UAV field.

## 2. Overall Structure Diagram

![绘图1改](https://github.com/user-attachments/assets/5c89b9b7-48d6-46ab-a26d-ec26b49691cb)


## 3. Usage Guide

### 3.1 Clone the code repository

Clone the code repository:

```python
git clone https://github.com/13849446299/MSDC-DETR
```
You can also download the MSDC-DETR compressed package directly by clicking on "Code" and then selecting "Download ZIP."

### 3.2 Install Dependencies

To install the required dependencies, run:

```python
pip install -r requirements.txt 
```

### 3.3 Enter the code directory

```python
cd MSDC-DETR
```
### 3.4 Training Example
Run the following command to start training the model:
```python
python train.py --config ultralytics/cfg/models/my_models/r18_MSDC_AFAM.yaml --data dataset/VisDrone.yaml
```
### 3.5 Evaluate Example
```python
python val.py --config ultralytics/cfg/models/my_models/r18_MSDC_AFAM.yaml --data dataset/VisDrone.yaml
```
### 3.6 Detec Example
```python
python Detec.py --config Your trained weight file address  --source Your picture address
```
## 4. Key Algorithms
### 4.1 Multi-Scale Deformable Convolutions (MSDC)

![绘图2](https://github.com/user-attachments/assets/d924a03d-bb4e-4919-8eb6-db4e352cdcae)

The Multi-Scale Deformable Convolution (MSDC) module enhances the multi-scale perception and spatial adaptability of convolutional networks through a multi-step feature processing approach. First, the input feature map is subjected to depth-wise convolution, which aggregates local features effectively. Next, multi-branch depth-wise strip convolutions are employed to capture contextual information across different scales and directions, facilitating a more comprehensive multi-scale feature representation.Following this, a 1×1 convolution is used to model inter-channel relationships. The outputs from this step serve as attention weights that reweight the input features, thereby emphasizing critical regions of interest. Finally, these attention weights adjust offsets and sampling point weights in conjunction with deformable convolution layers. This enables flexible adaptation to spatial variations, allowing the model to better handle diverse object shapes and sizes.

Here is the PyTorch implementation of the MSDC module:
```python
class DC_MSCA(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()

        padding = autopad(kernel_size, None, 1)
        self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)
        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        self.attention =MSCAAttention(self.out_channel)  # 原始

    def forward(self, x):
        conv_offset_mask = self.conv_offset_mask(x)
        conv_offset_mask = self.attention(conv_offset_mask)
        return conv_offset_mask
class MSDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2_Dynamic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()
    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()
class MSCA(BaseModule):
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        # Convolutional Attention
        x = attn * u
        return x
```
### 4.2 Adaptive Fusion Attention Module (AFAM)
![AFAM](https://github.com/user-attachments/assets/4cfbc8cd-0e71-49cf-8d0d-155a893cc112)

High-level features are downsampled (ADOWN) and processed through convolutional layers (Conv), adapting high-semantic features to enhance deep feature representation for small object detection. Simultaneously, low-level features are upsampled (EMA) and reprocessed. EMA uses parallel substructures and multi-scale convolutions for aggregating spatial structures in low-semantic features.Consequently, the Adaptive Fusion Attention Module (AFAM) effectively combines low-level and high-level features, emphasizing small object regions while suppressing irrelevant background information, ultimately improving detection accuracy for small targets.

Adaptive Downsampling (AODWN) module structure diagram

![ADOWN](https://github.com/user-attachments/assets/45ab4503-6ecc-4e22-91b0-0f07f4c1223c)

The PyTorch implementation of the AODWN module is as follows:

```python
class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)
```

Exponential Moving Average (EMA) module diagram

![EMA](https://github.com/user-attachments/assets/9f42e4c9-d6d8-4c0e-951d-3aa0767642fa)

The PyTorch implementation of the EMA module is as follows:

```python
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
```

## 5. Datasets

### 5.1 Download Links
- **VisDrone**: [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- **UAVDT**: [https://github.com/dataset-ninja/uavdt](https://github.com/dataset-ninja/uavdt)

### 5.2 Dataset Introduction

**VisDrone2019** is a comprehensive dataset meticulously designed for UAV vision tasks, containing 10,209 images divided into training (6,471 images), validation (548 images), and test sets (3,190 images). The images are collected from diverse environments and altitudes, covering various scenarios such as urban and rural areas, and accounting for different weather and lighting conditions. The objects in the dataset are categorized into 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. As the official test set is not publicly available, this study follows previous experimental setups, reporting evaluation results on the validation set to ensure reliability and comparability.

The **UAVDT** dataset provides 40,735 high-quality image resources, with the training set containing 24,143 images and the test set containing 16,592 images. The images in this dataset are primarily captured by UAVs in urban environments, with a resolution of approximately 1024×540 pixels. The annotated objects in the dataset are divided into three categories: car, bus, and truck, offering extensive data support for object detection and recognition research in UAV vision tasks.

## 6. Build the configuration file for your custom training set and the model configuration file

### 6.1 Dataset configuration file

```python
# Ultralytics YOLO 🚀, AGPL-3.0 license
# VisDrone2019-DET dataset https://github.com/VisDrone/VisDrone-Dataset by Tianjin University
# Example usage: yolo train data=VisDrone.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── VisDrone  ← downloads here (2.3 GB)
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /root/autodl-tmp/VisDrone # dataset root dir
train: VisDrone2019-DET-train  # train images (relative to 'path')  6471 images
val: VisDrone2019-DET-val  # val images (relative to 'path')  548 images
test: VisDrone2019-DET-test-dev
# number of classes
nc: 10

# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

### 6.2 Model configuration file

```python
# Parameters
nc: 10  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n-cls.yaml' will call yolov8-cls.yaml with scale 'n'
  # [depth, width, max_channels]
  l: [1.00, 1.00, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, ConvNormLayer, [32, 3, 2, 1, 'relu']] # 0-P1
  - [-1, 1, ConvNormLayer, [32, 3, 1, 1, 'relu']] # 1
  - [-1, 1, ConvNormLayer, [64, 3, 1, 1, 'relu']] # 2
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]] # 3-P2

  - [ -1, 2, Blocks, [ 64,  BottleNeck_DCNv2_attention,  2, False ] ] # 4
  - [ -1, 2, Blocks, [ 128, BottleNeck_DCNv2_attention,  3, False ] ] # 5-P3/8   [1, 512, 80, 80]
  - [ -1, 2, Blocks, [ 256, BasicBlock,  4, False ] ] # 6-P4
  - [ -1, 2, Blocks, [ 512, BasicBlock,  5, False ] ] # 7-P5

head:
  - [-1, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 8 input_proj.2
  - [-1, 1, EMA, [256]]
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, Conv, [256, 1, 1]]  # 11, Y5, lateral_convs.0

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 12
  - [6, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 13 input_proj.1
  - [-1, 1, EMA, [256]]
  - [[-3, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256, 0.5]]   # 16, fpn_blocks.0
  - [-1, 1, Conv, [256, 1, 1]]  # 17, Y4, lateral_convs.1

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 18
  - [5, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 19 input_proj.0
  - [-1, 1, EMA, [256]]
  - [[-3, -1], 1, Concat, [1]]  # 21 cat backbone P4
  - [-1, 3, RepC3, [256, 0.5]]   # X3 (22), fpn_blocks.1

  - [-1, 1, ADown, [256]]    # 23, downsample_convs.0  [1, 256, 40, 40]
  - [[-1, 17], 1, Concat, [1]]  # 24 cat Y4
  - [-1, 3, RepC3, [256, 0.5]]   # F4 (25), pan_blocks.0

  - [-1, 1, ADown, [256]]   # 26, downsample_convs.1  [1, 256, 20, 20]
  - [[-1, 11], 1, Concat, [1]]  # 27 cat Y5
  - [-1, 3, RepC3, [256, 0.5]]  # F5 (28), pan_blocks.1

  - [[22, 25, 28], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]  # Detect(P3, P4, P5)
```
## 7. Additional relevant details

The model configuration file for the ablation experiment is located in the following file:

```bash
MSDC-DETR/ultralytics/cfg/models/my_models
```

Training hyperparameter settings

```bash
epochs：   150        # (int) number of epochs to train for
batch：    6          # (int) number of images per batch (-1 for AutoBatch)
workers:   4          # (int) number of worker threads for data loading (per RANK if DDP)
lr0:       0.0001     # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf:       1          # (float) final learning rate (lr0 * lrf)
weight_decay:      0.0001     # (float) optimizer weight decay 5e-4
warmup_epochs:     2000       # (float) warmup epochs (fractions ok)
warmup_momentum:   0.8        # (float) warmup initial momentum
warmup_bias_lr:    0.1        # (float) warmup initial bias lr
optimizer:         AdamW      # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
```
The DOI link for the code is:

```bash
https://zenodo.org/records/14358387
```


