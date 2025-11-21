import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import deform_conv2d


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, stride=1, padding=None, bias=True):
        super().__init__()
        self.k = k
        self.stride = stride
        self.padding = k // 2 if padding is None else padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, offset, mask=None, dyn_weight=None):
        if dyn_weight is None:
            return deform_conv2d(
                input=x, weight=self.weight, offset=offset, mask=mask,
                bias=self.bias, stride=self.stride, padding=self.padding, dilation=1
            )

        B, C, H, W = x.shape
        outs = []
        for b in range(B):
            w_b = dyn_weight[b].unsqueeze(1).contiguous()
            x_b = x[b:b+1]
            off_b = offset[b:b+1]
            m_b = None if mask is None else mask[b:b+1]

            y_b = deform_conv2d(
                input=x_b, weight=w_b, offset=off_b, mask=m_b,
                bias=None, stride=self.stride, padding=self.padding, dilation=1
            )
            outs.append(y_b)

        y = torch.cat(outs, dim=0)
        return y


class TargetRegionDecoupler(nn.Module):
    def __init__(self, channels, num_regions):
        super().__init__()
        self.num_regions = num_regions
        self.pw = nn.Conv2d(channels, num_regions, kernel_size=1, bias=True)

    def forward(self, F):
        logits = self.pw(F)
        M = torch.softmax(logits, dim=1)
        
        M_regions = [M[:, i:i+1] for i in range(self.num_regions)]
        F_regions = [F * M[:, i:i+1] + F for i in range(self.num_regions)]
        return F_regions, M_regions


class OffsetGenerator(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.k = k
        hidden = max(32, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * k * k, 3, padding=1, bias=True)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, F_stream):
        return self.net(F_stream)


class AdaptiveKernelGenerator(nn.Module):
    def __init__(self, channels, k=3, Kdescriptor=7, act="tanh"):
        super().__init__()
        self.k = k
        self.desc = nn.Sequential(
            nn.AdaptiveAvgPool2d(Kdescriptor),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Conv2d(channels, channels * k * k, kernel_size=1, bias=True)
        self.act = nn.Tanh() if act == "tanh" else nn.Sigmoid()

    def forward(self, F_stream):
        B, C, _, _ = F_stream.shape
        d = self.desc(F_stream)
        w = self.head(d).view(B, C, self.k, self.k)
        w = self.act(w)
        return w


class DyRABlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_regions=2, k=3, Kdescriptor=7, use_bn=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.num_regions = num_regions
        
        self.decouple = TargetRegionDecoupler(in_channels, num_regions)

        self.offs = nn.ModuleList([OffsetGenerator(in_channels, k=k) for _ in range(num_regions)])
        self.akgs = nn.ModuleList([AdaptiveKernelGenerator(in_channels, k=k, Kdescriptor=Kdescriptor) for _ in range(num_regions)])
        self.dconvs = nn.ModuleList([ModulatedDeformConv(in_channels, out_channels, k=k) for _ in range(num_regions)])

        self.pwconv = nn.Conv2d(in_channels=out_channels*self.num_regions, out_channels=out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, F_c):
        # 1) target region decouple
        F_regions, M_regions = self.decouple(F_c)

        outs = []
        dec_dict = {}
        for i in range(self.num_regions):
            # 2) offsets + AKG mask
            F_i = F_regions[i]
            p_i = self.offs[i](F_i)
            K_i = self.akgs[i](F_i)
            
            # 3) deformable conv
            O_i = self.dconvs[i](F_i, p_i, mask=None, dyn_weight=K_i)
            
            outs.append(O_i)
            dec_dict[f"M_{i}"] = M_regions[i]
            dec_dict[f"O_{i}"] = O_i

        # 4) region-wise aggregation; 
        O = torch.cat(outs, dim=1)
        O = self.norm(self.pwconv(O))
        
        return O, dec_dict
    
    
class ACNN_Branch(nn.Module):
    def __init__(self, checkpoint, cls=3, num_regions=4, k=3, Kdescriptor=7):
        super().__init__()
        base_model = models.resnet18(pretrained=checkpoint)
        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.dyra1 = DyRABlock(64, 64, num_regions=num_regions, k=k, Kdescriptor=Kdescriptor)
        self.dyra2 = DyRABlock(128, 128, num_regions=num_regions, k=k, Kdescriptor=Kdescriptor)
        self.dyra3 = DyRABlock(256, 256, num_regions=num_regions, k=k, Kdescriptor=Kdescriptor)
        self.dyra4 = DyRABlock(512, 512, num_regions=num_regions, k=k, Kdescriptor=Kdescriptor)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x, dec1 = self.dyra1(x)
        x = self.layer2(x)
        x, dec2 = self.dyra2(x)
        x = self.layer3(x)
        out, dec3 = self.dyra3(x)
        x = self.layer4(out)
        pred, dec4 = self.dyra4(x)
    
        pred = self.avgpool(pred).squeeze(2).squeeze(2)
        pred = self.fc(pred)
    
        return out, pred, [dec1, dec2, dec3, dec4]
