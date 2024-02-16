import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image


class Models():
    def __init__(self,
                 path_to_model: str,
                 photo_path: str) -> None:
        super(Models, self).__init__()
        self.path_to_model = path_to_model
        self.photo_path = photo_path
        self.device = torch.device("cpu")
    
    def preprocess_image(self, src):
        imsize = 256

        loader = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])])
        
        image = Image.open(src)
        size = image.size
        image = loader(image).unsqueeze(0).to(self.device, torch.float)
        
        return image, size
    
    def run(self):
        model = torch.load(self.path_to_model, map_location=torch.device('cpu')).to(self.device).eval()
        photo, size = self.preprocess_image(self.photo_path)
        w, h = size
        output = model(photo).cpu().detach()[0]

        loader = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
            transforms.Resize((h, w))])

        res = loader(output).numpy().transpose(1, 2, 0)

        return res
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)