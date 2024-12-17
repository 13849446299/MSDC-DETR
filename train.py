import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/my_models/r18_MSDC_AFAM.yaml')
    #model.load('/root/autodl-tmp/my_rtdetr/r18_dcnv2_msa_ema3_adown_Foceeiou0.75_inner(640)3/weights/best.pt') # loading pretrain weights
    model.train(data=r'dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=6,
                workers=2,
                device='0',
                resume= '',
                project='',
                name='',
                # amp=True
                )