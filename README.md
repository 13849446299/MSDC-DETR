MSDC-DETR: UAV Object Detection Enhanced by Multi-Scale Deformable Convolution
Project Overview
MSDC-DETR is an innovative model designed for UAV (Unmanned Aerial Vehicle) image object detection. This model combines Multi-Scale Deformable Convolution (MSDC) and Adaptive Fusion Attention Module (AFAM) to improve the detection performance of small objects in complex backgrounds. By dynamically adjusting the shape and size of convolution kernels, MSDC-DETR can better adapt to the geometric variations of different targets, effectively enhancing the accuracy and robustness of object detection.

Key Features
Multi-Scale Deformable Convolution: The MSDC module dynamically adjusts convolution kernels to enhance adaptability to targets of varying sizes and shapes.
Adaptive Fusion Attention Module: AFAM improves the model's ability to detect small objects by integrating multi-scale features.
High Performance: MSDC-DETR outperforms existing state-of-the-art methods in mAP and other evaluation metrics on the VisDrone and UAVDT datasets.
Installation and Usage
Please refer to the requirements.txt file to install the necessary dependencies. You can create a virtual environment and install the dependencies using the following command:

pip install -r requirements.txt
Datasets
This model has been trained and evaluated on the following datasets:

VisDrone: VisDrone Dataset    https://github.com/VisDrone/VisDrone-Dataset
UAVDT: UAVDT Dataset          https://github.com/dataset-ninja/uavdt
Please ensure that you comply with the usage terms of the respective datasets.

Contribution
Contributions of any kind are welcome! If you have suggestions, issues, or wish to submit code, please feel free to open an issue or submit a pull request.
