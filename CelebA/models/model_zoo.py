import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['Adversary', 'ResNets', 'resnet20s', 'resnet32s', 'resnet44s', 'resnet56s', 'resnet110s', 'resnet1202s']


class Adversary(torch.nn.Module):
    def __init__(self, input_dim, output_dim, with_y=False, with_logits=False, use_mlp=False):
        super(Adversary, self).__init__()
        self.c = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.input_dim = input_dim
        self.with_y = with_y
        self.with_logits = with_logits
        if self.with_logits:
            self.input_dim += input_dim
        if self.with_y:
            self.input_dim += input_dim

        self.use_mlp = use_mlp
        if self.use_mlp:
            hidden_dim = [512]
            hidden_dim = [self.input_dim] + list(hidden_dim) + [output_dim]
            self.seq = torch.nn.ModuleList()
            for i in range(1, len(hidden_dim)):
                self.seq.append(torch.nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                if i != (len(hidden_dim) - 1):
                    self.seq.append(torch.nn.ReLU())
        else:
            self.fc = torch.nn.Linear(self.input_dim, output_dim, bias=True)

    def forward(self, inputs, *, y=None):
        assert len(inputs.shape) == 2
        if inputs.shape[1] > 1:
            s = torch.softmax(inputs * (1 + torch.abs(self.c)), dim=-1)
        else:
            s = torch.sigmoid(inputs * (1 + torch.abs(self.c)))
        if self.with_y:
            assert y is not None
            _y = y.long()
            encoded_inputs = torch.cat([s, s * _y], dim=1)
        else:
            assert y is None
            encoded_inputs = s
        if self.with_logits:
            encoded_inputs = torch.cat([inputs, encoded_inputs], dim=1)

        if self.use_mlp:
            logits = encoded_inputs
            for i, l in enumerate(self.seq):
                logits = l(logits)
            res = logits

        else:
            res = self.fc(encoded_inputs)

        return res


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNets, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20s(num_classes=10):
    return ResNets(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32s(num_classes=10):
    return ResNets(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44s(num_classes=10):
    return ResNets(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56s(num_classes=10):
    return ResNets(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110s(num_classes=10):
    return ResNets(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202s(num_classes=10):
    return ResNets(BasicBlock, [200, 200, 200], num_classes=num_classes)
