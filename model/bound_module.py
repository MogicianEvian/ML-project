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
                 padding=0, dilation=1, groups=1, bias=True):
        super(BoundConv2d, self).__init__()
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, lower=None, upper=None):
        y = super(BoundConv2d, self).forward(x)

        c = (lower + upper) / 2.
        r = (upper - lower) / 2.
        c = F.convolution_2d(c, self.weight, bias=self.bias, stride=self.stride, pad=self.pad)
        r = F.convolution_2d(r, abs(self.weight), bias=None, stride=self.stride, pad=self.pad)
        lower = c - r
        upper = c + r
        return y, lower, upper

# class RobustConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, non_negative = True):
#         super(RobustConv2d, self).__init__()
#         if non_negative:
#             self.weight = Parameter(torch.rand(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
#         else:
#             self.weight = Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
#         if bias:
#             self.bias = Parameter(torch.zeros(out_channels))
#         else:
#             self.bias = None
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = 1
#         self.non_negative = non_negative

#     def forward(self, input):
#         input_p = input[:input.shape[0]//2]
#         input_n = input[input.shape[0]//2:]
#         if self.non_negative:
#             out_p = F.conv2d(input_p, F.relu(self.weight), self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#             out_n = F.conv2d(input_n, F.relu(self.weight), self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#             return torch.cat([out_p, out_n],0)
            
#         u = (input_p + input_n)/2
#         r = (input_p - input_n)/2
#         out_u = F.conv2d(u, self.weight,self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#         out_r = F.conv2d(r, torch.abs(self.weight), None, self.stride,
#                         self.padding, self.dilation, self.groups)
#         return torch.cat([out_u + out_r, out_u - out_r], 0)
    

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
    def __init__(self, in_features, hidden, out_dim):
        super(Predictor, self).__init__()
        self.fc0 = BoundLinear(in_features, 3*32*32, bias = True)
        self.fc1 = BoundTanh()
        self.conv1 = BoundConv2d(3, 32, 3, stride = 1, padding = 1)
        self.fc2 = BoundReLU()
        self.conv2 = BoundConv2d(32, 32, 4, stride = 2, padding = 1)
        self.fc3 = BoundReLU()
        self.conv3 = BoundConv2d(32, 64, 3, stride = 1, padding = 1)
        self.fc4 = BoundReLU()
        self.conv4 = BoundConv2d(64, 64, 4, stride = 2, padding = 1)
        self.fc5 = BoundReLU()
        self.fc6 = BoundLinear(64*8*8, hidden, bias=True)
        self.tanh = BoundTanh()
        self.fc7 = BoundFinalLinear(hidden, out_dim)

    def forward(self, x, lower=None, upper=None, targets=None):
        ret = x, lower, upper
        ret = self.fc0(*ret)
        ret = self.fc1(*ret)
        # print(ret)
        ret[0].view(ret[0].size(0), 3, 32, 32)
        ret = self.conv1(*ret)
        ret = self.fc2(*ret)
        ret = self.conv2(*ret)
        ret = self.fc3(*ret)
        ret = self.conv3(*ret)
        ret = self.fc4(*ret)
        ret = self.conv4(*ret)
        ret = self.fc5(*ret)
        ret[0].view(ret[0].size(0), -1)
        ret = self.fc6(*ret)
        ret = self.tanh(*ret)
        ret = self.fc7(*ret, targets=targets)
        return ret

    # nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
    # nn.ReLU(),
    # Flatten(),
    # nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
    # nn.ReLU(),
    # nn.Linear(linear_size,linear_size),
    # nn.ReLU(),
    # nn.Linear(linear_size,10)

     # def __init__(self, in_features, hidden, out_dim):
    #     super(Predictor, self).__init__()
    #     self.fc1 = BoundLinear(in_features, hidden, bias=True)
    #     self.tanh = BoundTanh()
    #     self.fc2 = BoundFinalLinear(hidden, out_dim)
    # def forward(self, x, lower=None, upper=None, targets=None):
    #     ret = x, lower, upper
    #     ret = self.fc1(*ret)
    #     ret = self.tanh(*ret)
    #     ret = self.fc2(*ret, targets=targets)
    #     return ret
