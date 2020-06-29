from torch.utils.data.dataset import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms

"""读数据"""

transformers = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20, expand=True),  # 正负20度之间旋转，填满整张图片
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomGrayscale(0.1),
    transforms.RandomAffine(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class DCDataset(Dataset):  # 读数据集总路径
    def __init__(self, img_path, transform = None):
        self.img_path = img_path
        self.transform = transform
        self.images = []
        for clslist in os.listdir(self.img_path):
            for image in os.listdir(os.path.join(self.img_path, clslist)):
                self.images.append(os.path.join(self.img_path, clslist, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        strs = str(self.images[index]).split('\\')[-2]
        img = Image.open(os.path.join(self.images[index])).convert('RGB')
        label = torch.tensor(int(strs))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    img_path = r"A:\picture\blackeye\dark_circle"
    data = DCDataset(img_path, transformers)
    print(len(data))