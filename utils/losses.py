import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union

from utils.utils import soft_dice

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        target = target.long()
        if target.dim() == 4:
            target = target.squeeze(1)

        ce = self.cross_entropy(pred, target)
        dice = 1.0 - soft_dice(pred, target)

        total = self.dice_weight * dice + self.ce_weight * ce
        return total, dice, ce


def _gaussian_kernel2d(ks: int, sigma: float, device):
    ax = torch.arange(ks, device=device) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

def gaussian_blur(x: torch.Tensor, sigma: float, ksize: Optional[int] = None):
    if ksize is None:
        ksize = int(max(3, 2 * round(3 * sigma) + 1))
    k = _gaussian_kernel2d(ksize, sigma, device=x.device).view(1, 1, ksize, ksize)
    return F.conv2d(x, k, padding=ksize // 2)

def binary_dilate(x: torch.Tensor, radius: int):
    if radius <= 0:
        return x
    k = 2 * radius + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)

@torch.no_grad()
def signed_distance_transform_2d(mask: torch.Tensor):
    try:
        from scipy.ndimage import distance_transform_edt as edt
        outs = []
        for b in range(mask.shape[0]):
            m = (mask[b, 0] > 0.5).detach().cpu().numpy().astype(bool)
            dist_out = edt(~m)
            dist_in  = edt(m)
            sdt = dist_out - dist_in
            outs.append(torch.from_numpy(sdt)[None, None])
        return torch.cat(outs, 0).to(mask.device, dtype=mask.dtype)
    except Exception:
        B, _, H, W = mask.shape
        big = torch.full((B,1,H,W), 1e6, device=mask.device, dtype=mask.dtype)
        dist_in  = torch.where(mask > 0.5, torch.zeros_like(mask), big)
        dist_out = torch.where(mask <= 0.5, torch.zeros_like(mask), big)
        ker = torch.tensor([[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]],
                           device=mask.device, dtype=mask.dtype).view(1,1,3,3)
        iters = H + W
        for _ in range(iters):
            dist_in  = torch.min(dist_in,  F.conv2d(dist_in,  ker, padding=1)/8.0 + 1.0)
            dist_out = torch.min(dist_out, F.conv2d(dist_out, ker, padding=1)/8.0 + 1.0)
        return dist_out - dist_in

def make_soft_target(Y: torch.Tensor, r_d: int, blur_sigma: float):
    Y = Y.float().clamp(0, 1)
    Yd = binary_dilate(Y, radius=r_d)
    Ys = gaussian_blur(Yd, sigma=blur_sigma)
    mn = Ys.amin(dim=(-2,-1), keepdim=True)
    mx = Ys.amax(dim=(-2,-1), keepdim=True)
    return (Ys - mn) / (mx - mn + 1e-8)

def make_boundary_weight(Y: torch.Tensor, gamma: float, eps: float):
    sdt = signed_distance_transform_2d(Y)
    return torch.exp(-(sdt**2) / (2 * (gamma**2))) + eps


class KernelDecorrelationLoss(nn.Module):
    """Kernel decorrelation (KD) loss of Eq. (11).

        L_kd = 1/L sum_l 1/(N(N-1)) sum_{n != m} w_nm * |<K~_n,spec , K~_m,spec>|

    with ``K~_n,spec`` the l2-normalized target-specific residual produced by the
    DyRA-Block and ``w_nm = 1 - Sigmoid(c_n^T c_m / tau_c)`` an affinity-based
    softening term computed from the learnable target prototypes.

    Args:
        tau_c: affinity temperature ``tau_c`` (0.2 in the paper).
        target_indices: subset of region indices to penalize. ``None`` uses all
            regions produced by the DyRA-Block (index 0 is the background
            region under the ``num_regions = cls + 1`` convention).
        detach_affinity: if True, ``w_nm`` is treated as a constant weight. This
            blocks the degenerate shortcut where the prototypes are pushed
            towards each other to shrink ``w_nm`` instead of decorrelating the
            kernels themselves.
        cosine_affinity: if True, the prototype affinity uses cosine similarity
            instead of the raw inner product of Eq. (11). Useful when prototype
            norms drift during long schedules.
    """

    def __init__(self,
                 tau_c: float = 0.2,
                 target_indices: Optional[List[int]] = None,
                 detach_affinity: bool = True,
                 cosine_affinity: bool = False,
                 kernel_key: str = "K_spec",
                 proto_key: str = "prototypes"):
        super().__init__()
        self.tau_c = tau_c
        self.target_indices = target_indices
        self.detach_affinity = detach_affinity
        self.cosine_affinity = cosine_affinity
        self.kernel_key = kernel_key
        self.proto_key = proto_key

    def forward(self, preds: List[dict]) -> torch.Tensor:
        stage_losses = []
        ref = None

        for p in preds:
            if self.kernel_key not in p or self.proto_key not in p:
                continue

            K = p[self.kernel_key]                                    # (B, N, C, k, k)
            C_proto = p[self.proto_key]                               # (N, C_p)
            ref = K

            if self.target_indices is not None:
                idx = torch.as_tensor(self.target_indices, device=K.device, dtype=torch.long)
                K = K.index_select(1, idx)
                C_proto = C_proto.index_select(0, idx)

            N = K.shape[1]
            if N < 2:
                continue

            # |<K~_n , K~_m>| averaged over the batch
            K_norm = F.normalize(K.flatten(2), dim=-1)                # (B, N, C*k*k)
            gram = torch.matmul(K_norm, K_norm.transpose(1, 2)).abs() # (B, N, N)
            gram = gram.mean(dim=0)                                   # (N, N)

            # prototype affinity -> softening weights w_nm
            proto = F.normalize(C_proto, dim=-1) if self.cosine_affinity else C_proto
            affinity = torch.matmul(proto, proto.transpose(0, 1)) / self.tau_c
            omega = 1.0 - torch.sigmoid(affinity)
            if self.detach_affinity:
                omega = omega.detach()

            off_diag = ~torch.eye(N, dtype=torch.bool, device=K.device)
            stage_losses.append((omega * gram)[off_diag].sum() / (N * (N - 1)))

        if not stage_losses:
            device = ref.device if ref is not None else None
            return torch.zeros((), device=device)
        return torch.stack(stage_losses).mean()


class RegionDecouplingLoss(nn.Module):
    def __init__(self,
                 num_regions: int = 2,
                 target_classes: List[int] = None,
                 r_d: int = 2,
                 blur_sigma: float = 1.5,
                 gamma: float = 10.0,
                 eps: float = 1e-3,
                 from_logits: bool = False):
        super().__init__()
        self.num_regions = num_regions
        self.target_classes = target_classes or list(range(1, num_regions))
        
        self.r_d = r_d
        self.blur_sigma = blur_sigma
        self.gamma = gamma
        self.eps = eps
        self.from_logits = from_logits

    def forward(self,
                preds: Union[torch.Tensor, List[torch.Tensor]], Y: torch.Tensor) -> torch.Tensor:
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)
        Y = Y.float()
        
        stage_losses = []
        for p in preds:
            region_losses = []
            for i in range(1, self.num_regions):
                assert i in self.target_classes
                
                M_i = p[f"M_{i}"]
                M_i = torch.sigmoid(M_i) if self.from_logits else M_i.float()
                
                hw = M_i.shape[-2:]
                Y_i = (Y == i).float()
                Y_i = make_soft_target(Y, r_d=self.r_d, blur_sigma=self.blur_sigma)
                Y_i = F.interpolate(Y_i, size=hw, mode="bilinear", align_corners=False)
                
                BoundWeight = make_boundary_weight(Y, gamma=self.gamma, eps=self.eps)
                BoundWeight = F.interpolate(BoundWeight, size=hw, mode="bilinear", align_corners=False)
                
                diff = (M_i - Y_i).abs()
                num_pix = float(hw[0] * hw[1])
                l_i = (BoundWeight * diff).sum(dim=(-2, -1)).mean() / num_pix
                region_losses.append(l_i)
            
            stage_losses.append(torch.stack(region_losses).mean())
        return torch.stack(stage_losses).mean()