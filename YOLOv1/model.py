# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes, **kwargs)
        self.darknet.apply(self.init_weights)
        self.fcs.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        # First block in picture
        layers.append(CNNBlock(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2,2), padding=(3, 3)))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # Second block
        layers.append(CNNBlock(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # Third block
        layers.append(CNNBlock(in_channels=192, out_channels=128, kernel_size=(1, 1), padding=(0, 0)))
        layers.append(CNNBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(CNNBlock(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=(0, 0)))
        layers.append(CNNBlock(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # 4th block
        for i in range(4):
            layers.append(CNNBlock(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=(0, 0)))
            layers.append(CNNBlock(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(CNNBlock(in_channels=512, out_channels=512, kernel_size=(1, 1), padding=(0, 0)))
        layers.append(CNNBlock(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # 5th block
        for i in range(2):
            layers.append(CNNBlock(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=(0, 0)))
            layers.append(CNNBlock(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(CNNBlock(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(CNNBlock(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        # 6th block
        layers.append(CNNBlock(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(CNNBlock(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)))

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

def build_model(args):
    return Yolov1(in_channels=3, split_size=args.S, num_boxes=args.B, num_classes=args.C)

if __name__ == '__main__':
    yolo = Yolov1(split_size=7, num_boxes=2, num_classes=20)

    input_tensor = torch.rand((1, 3, 448, 448), dtype=torch.float)
    output = yolo(input_tensor)
    print(output.shape)
