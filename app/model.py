import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms

import copy


class Model():
    def __init__(self,
                 style_img_src: str,
                 content_img_src: str,
                 model_path: str) -> None:
        super(Model, self).__init__()
        self.style_img_src = style_img_src
        self.content_img_src = content_img_src

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

    def image_loader(self, image_name, loader):
        image = Image.open(image_name)
        size = image.size
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float), size

    def preprocess_image(self):
        imsize = 128

        loader = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor()])

        style_img, _ = self.image_loader(self.style_img_src, loader)
        content_img, size = self.image_loader(self.content_img_src, loader)
        return style_img, content_img, size

    def gram_matrix(self, inp):
        a, b, c, d = inp.size()
        features = inp.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers, style_layers):
        normalization = Normalization(normalization_mean, normalization_std)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature, self.gram_matrix)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, content_layers, style_layers, num_steps=300,
                           style_weight=1000000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img, content_img,
                                                                         content_layers, style_layers)
        input_img.requires_grad_(True)
        model.eval()
        model.requires_grad_(False)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    def vgg(self):
        cnn = torch.load(self.model_path).to(self.device).eval()
        style_img, content_img, size = self.preprocess_image()
        w, h = size
        input_img = content_img.clone()
        output = self.run_style_transfer(cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                    content_img, style_img, input_img, self.content_layers_default, self.style_layers_default).cpu().detach()[0]
        transform = transforms.Resize((h, w))
        res = transform(output).numpy().transpose(1, 2, 0)

        return res

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature, gram_matrix):
        super(StyleLoss, self).__init__()
        self.gram_matrix = gram_matrix
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1) # self.mean = torch.tensor(mean).view(-1, 1, 1) 
        self.std = mean.clone().detach().view(-1, 1, 1) # self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
