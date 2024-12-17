import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/my_rtdetr/r18_dcnv2_msa_ema3(640)/weights/best.pt')
    model.val(data=r'dataset/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='/root/autodl-tmp/val',
              name='r18_dcnv2_msa_ema3',
              )