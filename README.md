# Enhancing UAV Object Detection through Multi-Scale Deformable Convolutions and Adaptive Fusion Attention

## 1. Introduction

MSDC-DETR is an innovative model designed specifically for UAV image object detection. This model combines Multi-Scale Deformable Convolutions (MSDC) and Adaptive Fusion Attention Module (AFAM) to improve the detection performance of small objects in complex backgrounds. By dynamically adjusting the shape and size of convolution kernels, MSDC-DETR can better adapt to the geometric variations of different targets, aiming to enhance detection accuracy in UAV image object detection and unlock its potential for practical applications in the UAV field.

## 2. Overall Structure Diagram

![绘图1改](https://github.com/user-attachments/assets/5c89b9b7-48d6-46ab-a26d-ec26b49691cb)


## 3. Usage Guide

### 3.1 Install Dependencies

To install the required dependencies, run:

```python
pip install -r requirements.txt 
```

### 3.2 Install Dependencies

Clone the code repository:

```python
git clone https://github.com/yourusername/yourrepository.git
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

