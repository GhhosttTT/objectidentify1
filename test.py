import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

image_path = "./imgs/dog.png"
image = Image.open(image_path)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])  # 数据变换
image = transform(image)

print(image.shape)


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, padding=2),  # 这个padding要打出来不然会变成其他的属性值
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("./testpth/test_199.pth")
print(model)
image = torch.reshape(image, [1,3, 32, 32])
image=image.to("cuda")
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))

test_data = torchvision.datasets.CIFAR10("./dataset", False, torchvision.transforms.ToTensor(), download=False)
print(test_data.classes[output.argmax(1).item()])