import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('E:\\训练数据\\r18_dcnv2_msa_ema3_adown_Foceeiou0.75(640)\\weights\\best.pt') # select your model.pt path
    model.predict(source='E:\\daima\\RTDETR_model\\RT-DETR\dataset\\22222\\1\\1\\0000006_00159_d_0000001.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  classes=[0,1,2,3,4,5,6,7,8,9],
                )