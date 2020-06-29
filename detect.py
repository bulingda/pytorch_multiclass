"""图片和模型传进来就可以用了"""
import torch
import numpy as np
import time
import os
from DCModel import pmodel
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

TEST_IMAGE_PATH = r"A:\picture\blackeye\dark_circle\3\left1_004638_China_female_34_0_0_1_0_0.jpg"
# LABELPATH = r'D:\E\document\datas\megaage_asian\list'


class Detector:
    def __init__(self, net, image=TEST_IMAGE_PATH, net_param=r"model_Adam_Focalloss_steplr_bs128_lr0.001_acc0.478698.pt", isCuda=False):
        self.net = net
        if os.path.exists(net_param):
            self.net.load_state_dict(torch.load(net_param)['state_dict'])
            self.net.eval()
        else:
            print("没找到模型！！！")
        self.image_path = image
        self.isCuda = isCuda
        if self.isCuda:
            self.net.to(device)

    def detect(self):
        # for filename in os.listdir(self.image_path):
        img = Image.open(self.image_path).convert('RGB')
        img = transformers(img)
        img = torch.unsqueeze(img, 0)
        if self.isCuda:
            inputs = img.to(device)
            outputs = self.net(inputs)
        else:
            inputs = img
            outputs = self.net(inputs)
        outputs = torch.argmax(outputs, axis=1)
        return outputs.item()


def netMain():
    pre_model = pmodel()
    detects1 = Detector(pre_model, net_param=r"model_Adam_Focalloss_steplr_bs128_lr0.001_acc0.478698.pt")
    output = detects1.detect()
    # x = np.median(age1)
    print("预测结果：", output)


if __name__ == '__main__':
    start_time = time.time()
    netMain()
    end_time = time.time()
    usingtime = end_time - start_time
    print("time of per picture in {:.4f}:".format(usingtime))