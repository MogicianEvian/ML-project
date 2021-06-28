import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundReLU(nn.ReLU):
    def __init__(self):
        super(BoundReLU, self).__init__()
    def forward(self, x, lower=None, upper=None):
        y = super(BoundReLU, self).forward(x)
        if lower is None or upper is None:
            return y, None, None
        return y, torch.relu(lower), torch.relu(upper)

class BoundTanh(nn.Tanh):
    def __init__(self):
        super(BoundTanh, self).__init__()
    def forward(self, x, lower=None, upper=None):
        y = super(BoundTanh, self).forward(x)
        if lower is None or upper is None:
            return y, None, None
        return y, torch.tanh(lower), torch.tanh(upper)

def linear(input, weight, bias, w_scale, b_scale):
    if bias is None:
        return torch.mm(input, weight.T) * w_scale
    return torch.addmm(bias, input, weight.T, alpha=w_scale, beta=b_scale)

class BoundLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_scale=1.0, b_scale=1.0):
        super(BoundLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.zero_()
        self.w_scale = w_scale / math.sqrt(in_features)
        self.b_scale = b_scale
    def forward(self, x, lower=None, upper=None):
        y = linear(x, self.weight, self.bias, self.w_scale, self.b_scale)
        if lower is None or upper is None:
            return y, None, None
        x_mul_2 = lower + upper
        r_mul_2 = upper - lower
        x = linear(x_mul_2, self.weight, self.bias, self.w_scale / 2, self.b_scale)
        r_mul_2 = torch.mm(r_mul_2, self.weight.abs().T)
        lower = torch.add(x, r_mul_2, alpha=-self.w_scale / 2)
        upper = torch.add(x, r_mul_2, alpha=self.w_scale / 2)
        return y, lower, upper

class BoundFinalLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_scale=1.0, b_scale=1.0):
        super(BoundFinalLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight.data.normal_()
        self.bias.data.zero_()
        self.w_scale = w_scale / math.sqrt(in_features)
        self.b_scale = b_scale
    def forward(self, x, lower=None, upper=None, targets=None):
        y = linear(x, self.weight, self.bias, self.w_scale, self.b_scale)
        if lower is None or upper is None or targets is None:
             return y
        w = self.weight - self.weight.index_select(0, targets).unsqueeze(1) # B * CO * CI
        x_mul_2 = lower + upper
        r_mul_2 = upper - lower
        x = w.bmm(x_mul_2.unsqueeze(-1)) * (self.w_scale / 2)
        if self.bias is not None:
            b = self.bias - self.bias.index_select(0, targets).unsqueeze(1)
            x = torch.add(x, b.unsqueeze(-1), alpha=self.b_scale)
        r_mul_2 = w.abs().bmm(r_mul_2.unsqueeze(-1))
        res = torch.add(x, r_mul_2, alpha=self.w_scale / 2).squeeze(-1)
        return y, res


class BoundConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BoundConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=1, groups=1, bias=False)

    def forward(self, x, lower=None, upper=None):
        y = super(BoundConv2d, self).forward(x)
        if lower is None or upper is None:
            return y, None, None
        c = (lower + upper) / 2.0
        r = (upper - lower) / 2.0
        c = F.conv2d(c, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        r = F.conv2d(r, self.weight.abs(), bias=None, stride=self.stride, padding=self.padding)
        lower = c - r
        upper = c + r
        return y, lower, upper

# class BounConv2d(nn.Conv2d):
#     def __init__(self, in_features, out_features, bias=True, w_scale=1.0, b_scale=1.0):
#         super(BoundLinear, self).__init__(in_features, out_features, bias=bias)
#         self.weight.data.normal_()
#         if self.bias is not None:
#             self.bias.data.zero_()
#         self.w_scale = w_scale / math.sqrt(in_features)
#         self.b_scale = b_scale
#     def forward(self, x, lower=None, upper=None):
#         y = F.convolution_2d(x, self.weight, self.bias, self.w_scale, self.b_scale)
#         if lower is None or upper is None:
#             return y, None, None
#         x_mul_2 = lower + upper
#         r_mul_2 = upper - lower
#         x = F.convolution_2d(x_mul_2, self.weight, self.bias, self.w_scale / 2, self.b_scale)
#         r_mul_2 = torch.mm(r_mul_2, self.weight.abs().T)
#         lower = torch.add(x, r_mul_2, alpha=-self.w_scale / 2)
#         upper = torch.add(x, r_mul_2, alpha=self.w_scale / 2)
#         return y, lower, upper

from model.norm_dist import MeanNorm
class BoundMeanNorm(MeanNorm):
    def __init__(self, out_channels, momentum=0.1):
        super(BoundMeanNorm, self).__init__(out_channels, momentum)
    def forward(self, x, lower=None, upper=None):
        z = super(BoundMeanNorm, self).forward(x)
        if lower is None or upper is None:
             return z, None, None
        x = (lower + upper) / 2
        y = x.view(x.size(0), x.size(1), -1)
        y_lower = lower.view_as(y)
        y_upper = upper.view_as(y)
        if self.training:
            if x.dim() > 2:
                mean = y.mean(dim=-1).mean(dim=0)
            else:
                mean = x.mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        else:
            mean = self.running_mean
        lower = (y_lower - mean.unsqueeze(-1)).view_as(lower)
        upper = (y_upper - mean.unsqueeze(-1)).view_as(upper)
        return z, lower, upper

class BoundFinalIdentity(nn.Module):
    def __init__(self):
        super(BoundFinalIdentity, self).__init__()
    def forward(self, x, lower=None, upper=None, targets=None):
        if lower is None or upper is None or targets is None:
            return x
        res = upper - torch.gather(lower, dim=1, index=targets.unsqueeze(-1))
        res = res.scatter(dim=1, index=targets.unsqueeze(-1), value=0)
        return x, res

class Predictor(nn.Module):
    # def __init__(self, in_features, hidden, out_dim):
    #     super(Predictor, self).__init__()
    #     self.conv1 = BoundConv2d(in_features//256, 256, 3, stride = 1, padding = 1)
    #     self.fc1 = BoundReLU()
    #     self.norm1 = BoundMeanNorm(256)
    #     self.conv2 = BoundConv2d(256, 256, 3, stride = 1, padding = 1)
    #     self.fc2 = BoundReLU()
    #     self.norm2 = BoundMeanNorm(256)
    #     self.conv3 = BoundConv2d(256, 512, 4, stride = 2, padding = 1)
    #     self.fc3 = BoundReLU()
    #     self.norm3 = BoundMeanNorm(512)
    #     self.fc4 = BoundLinear(512*8*8, hidden, bias=True)
    #     self.tanh = BoundTanh()
    #     self.fc5 = BoundFinalLinear(hidden, out_dim)

    # def forward(self, x, lower=None, upper=None, targets=None):
    #     ret = x, lower, upper
    #     # ret = ret[0].view(ret[0].size(0), 3, 32, 32), None if ret[1] is None else ret[1].view(ret[1].size(0), 3, 32, 32), None if ret[2] is None else ret[2].view(ret[2].size(0), 3, 32, 32)
    #     ret = self.conv1(*ret)
    #     ret = self.fc1(*ret)
    #     ret = self.norm1(*ret)
    #     ret = self.conv2(*ret)
    #     ret = self.fc2(*ret)
    #     ret = self.norm2(*ret)
    #     ret = self.conv3(*ret)
    #     ret = self.fc3(*ret)
    #     ret = self.norm3(*ret)
    #     # ret[0] = ret[0].view(ret[0].size(0), -1)
    #     ret = ret[0].view(ret[0].size(0),-1), None if ret[1] is None else ret[1].view(ret[1].size(0),-1), None if ret[2] is None else ret[2].view(ret[2].size(0),-1)
    #     ret = self.fc4(*ret)
    #     ret = self.tanh(*ret)
    #     ret = self.fc5(*ret, targets=targets)
    #     return ret

    def __init__(self, in_features, hidden, out_dim):
        super(Predictor, self).__init__()
        self.fc1 = BoundLinear(in_features, hidden, bias=True)
        self.tanh = BoundTanh()
        self.fc2 = BoundFinalLinear(hidden, out_dim)
    def forward(self, x, lower=None, upper=None, targets=None):
        ret = x, lower, upper
        ret = self.fc1(*ret)
        ret = self.tanh(*ret)
        ret = self.fc2(*ret, targets=targets)
        return ret
