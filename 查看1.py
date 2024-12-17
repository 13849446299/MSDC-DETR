import warnings
warnings.filterwarnings('ignore')
from torchinfo import summary
import torch.jit
device = torch.device("cuda")
from ultralytics.nn.tasks import RTDETRDetectionModel
model = RTDETRDetectionModel()
print(model)
from torchinfo import  summary
from torch.utils.tensorboard import SummaryWriter


print("********************************")
print("********************************")
print("********************************")
input_size = (1, 3, 640, 640)

summary(model,input_size=input_size)

# # 创建输入张量
input = torch.ones((1, 3,608, 1024)).to(device=device, dtype=torch.float32)
#
#
# # 创建 SummaryWriter
writer = SummaryWriter("logs")
#
# # 将模型的图形添加到日志文件
with torch.no_grad():
     writer.add_graph(model, input)
#
# # 关闭 SummaryWriter
writer.close()


