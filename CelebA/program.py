import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class Program(nn.Module):
    def __init__(self, out_size):
        super(Program, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(3, *out_size))
        self.weight.data.uniform_(-1, 1)

    def forward(self, mask):
        x = self.weight * mask
        return x


class AdvProgram(nn.Module):
    def __init__(self, in_size, out_size, mask_size, device=torch.device('cuda')):
        super(AdvProgram, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.program = Program(out_size).to(device)

        l_pad = int((mask_size[0] - in_size[0] + 1) / 2)
        r_pad = int((mask_size[0] - in_size[0]) / 2)

        mask = torch.zeros(3, *in_size, device=device)
        self.mask = F.pad(mask, (l_pad, r_pad, l_pad, r_pad), value=1)

    def forward(self, x):
        x = x + torch.tanh(self.program(self.mask))
        return x


class PatchProgram(nn.Module):
    def __init__(self, patch_size, out_size, loc=0, device=torch.device('cuda')):
        super(PatchProgram, self).__init__()

        assert len(patch_size) == 2 and len(out_size) == 2 and patch_size[0] * 2 < out_size[0]

        self.trigger_size = patch_size
        self.out_size = out_size
        self.program = Program(out_size).to(device)

        mask = torch.ones(3, *patch_size, device=device)
        # background
        bg = torch.zeros(3, *out_size, device=device)

        # loc represents different corners:
        # 01
        # 23
        size = patch_size[0]
        large_size = out_size[0]
        if loc == 0:
            bg[:, 0:size, 0:size] = mask
        elif loc == 1:
            bg[:, large_size - 1 - size:large_size - 1, 0:size] = mask
        elif loc == 2:
            bg[:, 0:size, large_size - 1 - size:large_size - 1] = mask
        else:
            bg[:, large_size - 1 - size:large_size - 1, large_size - 1 - size:large_size - 1] = mask

        self.mask = bg

    def forward(self, x):
        x = x * (1 - self.mask) + torch.tanh(self.program(self.mask))
        return x


class OptimProgram(nn.Module):
    def __init__(self, size, k, device=torch.device('cuda')):
        super(OptimProgram, self).__init__()
        self.size = size
        self.k = k
        self.device = device
        self.program = Program(size).to(device)

        self.scores = torch.nn.Parameter(data=torch.Tensor(3, *size)).to(device)
        self.scores.data.uniform_(-1, 1)

    def forward(self, x):
        adj = GetSubnet.apply(self.scores, self.k)
        x = x * (1 - adj) + torch.tanh(self.program(adj))
        return x

    def set_k(self, k):
        self.k = k


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):  # binarization
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()  # same addr . change dim but
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

