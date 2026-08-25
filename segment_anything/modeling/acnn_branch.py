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


class TargetConditionedKernelGenerator(nn.Module):
    """Target-conditioned kernel generator (Sec. III-A.2 of the paper).

    A single generator is instantiated per DyRA-Block and re-used for all N
    targets, so that ``Conv^shared_1x1`` is genuinely weight-shared across
    targets at stage l:

        d^l_n     = GAP(ReLU(Conv3x3(AvgPool_K(F_bar^l_n))))          in R^C
        K_spec    = Reshape(Conv^spec_1x1([d^l_n ; phi(c_n)]))        in R^{C x Kb x Kb}
        K_shared  = Reshape(Conv^shared_1x1(d^l_n))                   in R^{C x Kb x Kb}
        zeta^l_n  = Sigmoid(phi_zeta(d^l_n))                          in R
        K^l_n     = Tanh(K_shared + zeta^l_n * K_spec)

    ``{c_n}`` are learnable, location-agnostic target prototypes. The
    un-composed ``K_spec`` is returned as well because the kernel
    decorrelation loss (Eq. 11) is defined on the target-specific residual.
    """

    def __init__(self, channels, num_targets, k=3, Kdescriptor=7, proto_dim=None, act="tanh"):
        super().__init__()
        self.k = k
        self.channels = channels
        self.num_targets = num_targets
        self.proto_dim = proto_dim or channels

        # (i) local-context summarization -> channel-wise descriptor d^l_n
        self.desc = nn.Sequential(
            nn.AdaptiveAvgPool2d(Kdescriptor),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # (ii) learnable target prototypes {c_n} and their projection phi(.)
        # Orthonormal init: targets start maximally distinguishable (c_n^T c_m = 0,
        # i.e. omega_nm = 0.5 in Eq. (11)) and at a magnitude comparable to d^l_n,
        # so the target identity actually modulates the kernel from step 0.
        self.prototypes = nn.Parameter(torch.empty(num_targets, self.proto_dim))
        if num_targets <= self.proto_dim:
            nn.init.orthogonal_(self.prototypes)
        else:
            nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.proto_proj = nn.Conv2d(self.proto_dim, channels, kernel_size=1, bias=True)

        # (iii) shared-specific kernel decomposition heads
        self.head_spec = nn.Conv2d(2 * channels, channels * k * k, kernel_size=1, bias=True)
        self.head_shared = nn.Conv2d(channels, channels * k * k, kernel_size=1, bias=True)

        # (iv) scalar gate zeta^l_n modulating the target-specific contribution
        self.gate_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        self.act = nn.Tanh() if act == "tanh" else nn.Sigmoid()

    def forward(self, F_stream, target_idx):
        B, C, _, _ = F_stream.shape

        # channel-wise descriptor of the target-attended feature
        d = self.desc(F_stream)                                       # (B, C, 1, 1)

        # target identity injection: [d ; phi(c_n)]
        c = self.prototypes[target_idx].view(1, self.proto_dim, 1, 1)
        c = self.proto_proj(c).expand(B, -1, -1, -1)                  # (B, C, 1, 1)

        K_spec = self.head_spec(torch.cat([d, c], dim=1)).view(B, C, self.k, self.k)
        K_shared = self.head_shared(d).view(B, C, self.k, self.k)
        zeta = torch.sigmoid(self.gate_proj(d)).view(B, 1, 1, 1)

        K = self.act(K_shared + zeta * K_spec)
        return K, K_spec, zeta.view(B)


class DyRABlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_regions=2, k=3, Kdescriptor=7, use_bn=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.num_regions = num_regions
        
        self.decouple = TargetRegionDecoupler(in_channels, num_regions)

        self.offs = nn.ModuleList([OffsetGenerator(in_channels, k=k) for _ in range(num_regions)])
        # One generator for all targets: Conv^shared_1x1 is weight-shared across
        # targets, while the target identity enters through the prototypes {c_n}.
        self.akg = TargetConditionedKernelGenerator(
            in_channels, num_targets=num_regions, k=k, Kdescriptor=Kdescriptor
        )
        # The dynamic kernels are applied depth-wise, so a single deformable-conv
        # wrapper is shared by every target region.
        self.dconv = ModulatedDeformConv(in_channels, out_channels, k=k)

        self.pwconv = nn.Conv2d(in_channels=out_channels*self.num_regions, out_channels=out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, F_c):
        # 1) target region decouple
        F_regions, M_regions = self.decouple(F_c)

        outs = []
        dec_dict = {}
        K_specs, zetas = [], []
        for i in range(self.num_regions):
            # 2) offsets + target-conditioned kernel (shared + gated specific)
            F_i = F_regions[i]
            p_i = self.offs[i](F_i)
            K_i, K_spec_i, zeta_i = self.akg(F_i, target_idx=i)

            # 3) deformable conv
            O_i = self.dconv(F_i, p_i, mask=None, dyn_weight=K_i)

            outs.append(O_i)
            K_specs.append(K_spec_i)
            zetas.append(zeta_i)
            dec_dict[f"M_{i}"] = M_regions[i]
            dec_dict[f"O_{i}"] = O_i

        # target-specific residuals / gates / prototypes for the KD loss (Eq. 11)
        dec_dict["K_spec"] = torch.stack(K_specs, dim=1)      # (B, N, C, k, k)
        dec_dict["zeta"] = torch.stack(zetas, dim=1)          # (B, N)
        dec_dict["prototypes"] = self.akg.prototypes         # (N, C)

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
