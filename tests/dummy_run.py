"""Dummy-input smoke test for the DyRA-Block target-conditioned kernels
(Sec. III-A.2) and the kernel decorrelation loss (Sec. III-D.3).

No dataset is required: every input is a randomly generated tensor matching the
shapes DReAM-SAM expects.

Usage
-----
    # module-level check (DyRA-Block + ACNN branch + L_kd), runs on CPU
    python tests/dummy_run.py --part dyra

    # full DReAM-SAM forward/backward with the complete objective
    python tests/dummy_run.py --part full --device cuda

    # both
    python tests/dummy_run.py --part all --device cuda
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything.modeling.acnn_branch import ACNN_Branch, DyRABlock  # noqa: E402
from utils.losses import (  # noqa: E402
    DiceCELoss,
    KernelDecorrelationLoss,
    RegionDecouplingLoss,
)
from utils.utils import get_class_vector  # noqa: E402


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Part 1: DyRA-Block / ACNN branch / L_kd on random tensors
# --------------------------------------------------------------------------- #
def run_dyra(args):
    device = torch.device(args.device)
    B, C, H, W = args.batch_size, 64, 56, 56
    N, k = args.num_regions, 3

    banner("[1] DyRA-Block with target-conditioned kernels")
    block = DyRABlock(C, C, num_regions=N, k=k, Kdescriptor=7).to(device).train()

    F_c = torch.randn(B, C, H, W, device=device)
    out, dec = block(F_c)

    print(f"input  F_c      : {tuple(F_c.shape)}")
    print(f"output F_hat_c  : {tuple(out.shape)}")
    print(f"M_0             : {tuple(dec['M_0'].shape)}")
    print(f"K_spec (B,N,C,k,k): {tuple(dec['K_spec'].shape)}")
    print(f"zeta   (B,N)    : {tuple(dec['zeta'].shape)}  values={dec['zeta'][0].tolist()}")
    print(f"prototypes (N,C): {tuple(dec['prototypes'].shape)}")
    assert out.shape == (B, C, H, W)
    assert dec["K_spec"].shape == (B, N, C, k, k)
    assert dec["zeta"].shape == (B, N)
    assert dec["prototypes"].shape == (N, C)

    # The shared basis is identical for every target, the specific residual is not.
    banner("[2] Shared vs. target-specific decomposition")
    with torch.no_grad():
        d = block.akg.desc(F_c)
        # K_shared depends on d only -> identical for every target by construction
        K_shared = block.akg.head_shared(d).view(B, C, k, k)
        finals, specs = zip(*[(lambda r: (r[0], r[1]))(block.akg(F_c, target_idx=n))
                              for n in range(N)])

    print(f"K_shared (target-agnostic) : {tuple(K_shared.shape)}, "
          f"mean|K_shared| = {K_shared.abs().mean().item():.5f}")
    for n in range(1, N):
        print(f"  t0 vs t{n}: mean|K_spec diff| = {(specs[0] - specs[n]).abs().mean().item():.5f}"
              f" | mean|K diff| = {(finals[0] - finals[n]).abs().mean().item():.5f}")
    assert (specs[0] - specs[1]).abs().mean() > 0, "target conditioning has no effect"

    # --------------------------------------------------------------------- #
    banner("[3] Kernel decorrelation loss on a 4-stage ACNN branch")
    acnn = ACNN_Branch(checkpoint=args.pretrained, cls=args.num_regions - 1,
                       num_regions=N).to(device).train()
    img = torch.randn(B, 3, args.acnn_size, args.acnn_size, device=device)
    feat, pred_cls, decoupled_list = acnn(img)
    print(f"ACNN feature    : {tuple(feat.shape)}")
    print(f"TEC logits      : {tuple(pred_cls.shape)}")
    print(f"stages returned : {len(decoupled_list)}")
    for i, dec_l in enumerate(decoupled_list):
        print(f"  stage {i + 1}: K_spec={tuple(dec_l['K_spec'].shape)}, "
              f"prototypes={tuple(dec_l['prototypes'].shape)}")

    kd_loss = KernelDecorrelationLoss(tau_c=args.tau_c, detach_affinity=True)
    l_kd = kd_loss(decoupled_list)
    print(f"L_kd            : {l_kd.item():.6f}")

    l_kd.backward(retain_graph=True)
    proto_grad = acnn.dyra1.akg.prototypes.grad
    spec_grad = acnn.dyra1.akg.head_spec.weight.grad
    shared_grad = acnn.dyra1.akg.head_shared.weight.grad
    print(f"grad |prototypes|      (stage1): {proto_grad.abs().mean().item():.3e}")
    print(f"grad |Conv^spec.weight|(stage1): {spec_grad.abs().mean().item():.3e}")
    # The shared head is not in Eq. (11) directly, but stage-1 kernels shape the
    # features consumed by stages 2..4, so it receives an indirect gradient.
    shared_txt = "None" if shared_grad is None else f"{shared_grad.abs().mean().item():.3e} (indirect, via later stages)"
    print(f"grad |Conv^shared.weight|(st.1): {shared_txt}")
    assert proto_grad is not None and proto_grad.abs().sum() > 0
    assert spec_grad is not None and spec_grad.abs().sum() > 0

    # --------------------------------------------------------------------- #
    if args.optimize_steps > 0:
        banner(f"[4] Minimizing L_kd for {args.optimize_steps} steps (gradient sanity)")
        params = [p for n, p in acnn.named_parameters() if "akg" in n]
        opt = torch.optim.Adam(params, lr=1e-3)
        for step in range(args.optimize_steps):
            opt.zero_grad(set_to_none=True)
            _, _, dec_list = acnn(img)
            loss = kd_loss(dec_list)
            loss.backward()
            opt.step()
            if step % max(1, args.optimize_steps // 5) == 0 or step == args.optimize_steps - 1:
                print(f"  step {step:3d}: L_kd = {loss.item():.6f}")


# --------------------------------------------------------------------------- #
# Part 2: full DReAM-SAM forward/backward with the complete objective
# --------------------------------------------------------------------------- #
def make_dummy_batch(batch_size, num_classes, img_size, device):
    """Mimics one `Processor` output (see utils/dataloader.py)."""
    batch = []
    for _ in range(batch_size):
        image = torch.rand(3, img_size, img_size, device=device) * 255.0
        gt = torch.randint(0, num_classes + 1, (1, img_size, img_size),
                           device=device, dtype=torch.float)
        batch.append({
            "image": image,
            "gt_mask": gt,
            "original_size": (img_size, img_size),
        })
    return batch


def run_full(args):
    from segment_anything import sam_model_registry
    from utils.LoRA import LoRA_Sam

    device = torch.device(args.device)
    cls = args.num_regions - 1
    num_regions = args.num_regions

    banner("[5] Full DReAM-SAM forward/backward on dummy tensors")
    sam = sam_model_registry["DReAM-SAM"](cls=cls, num_regions=num_regions, checkpoint=None)
    sam.prompt_generator.point_embeddings = sam.prompt_encoder.point_embeddings
    sam.prompt_generator.not_a_point_embed = sam.prompt_encoder.not_a_point_embed

    model = LoRA_Sam(sam, r=4) if args.lora else sam
    model.to(device).train()

    img_size = sam.image_encoder.img_size
    batch = make_dummy_batch(args.batch_size, cls, img_size, device)
    print(f"batch           : {len(batch)} x image{tuple(batch[0]['image'].shape)}, "
          f"gt_mask{tuple(batch[0]['gt_mask'].shape)}, cls={cls}, num_regions={num_regions}")

    outputs, pred_cls, decoupled_res = model(batched_input=batch, multimask_output=False)
    print(f"low_res_logits  : {tuple(outputs[0]['low_res_logits'].shape)}")
    print(f"TEC probs       : {tuple(pred_cls.shape)}")

    stk_gt = torch.stack([b["gt_mask"].squeeze(0) for b in batch], dim=0).long()
    stk_out = torch.stack([o["low_res_logits"].squeeze(1) for o in outputs], dim=0)
    print(f"stk_out / stk_gt: {tuple(stk_out.shape)} / {tuple(stk_gt.shape)}")

    criterion = DiceCELoss()
    classification_loss = nn.BCELoss()
    decouple_loss = RegionDecouplingLoss(num_regions=num_regions, gamma=10.0, eps=1e-3,
                                         r_d=2, blur_sigma=1.5, from_logits=False)
    kernel_decorr_loss = KernelDecorrelationLoss(tau_c=args.tau_c, detach_affinity=True)

    loss, dice_loss, ce_loss = criterion(stk_out, stk_gt)
    stk_cls_lbl = torch.stack([get_class_vector(b["gt_mask"], cls) for b in batch]).to(device)
    l_cls = classification_loss(pred_cls, stk_cls_lbl)
    l_kd = kernel_decorr_loss(decoupled_res)
    l_rd = decouple_loss(decoupled_res, stk_gt)

    total = loss + l_cls + args.kd_weight * l_kd + args.lambda_rd * l_rd
    print(f"L_seg (dice/ce) : {loss.item():.4f} ({dice_loss.item():.4f} / {ce_loss.item():.4f})")
    print(f"L_cls           : {l_cls.item():.4f}")
    print(f"L_kd            : {l_kd.item():.6f}")
    print(f"L_rd            : {l_rd.item():.4f}")
    print(f"L_total         : {total.item():.4f}")

    total.backward()
    named = dict(model.named_parameters())
    key_proto = [n for n in named if n.endswith("akg.prototypes")]
    print(f"prototype tensors found: {len(key_proto)} ({key_proto[0]} ...)")
    for n in key_proto:
        g = named[n].grad
        assert g is not None and torch.isfinite(g).all(), f"bad grad for {n}"
    print("all prototype grads are finite and populated -> backward OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["dyra", "full", "all"], default="dyra")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-regions", type=int, default=4, help="cls + 1 (background)")
    p.add_argument("--acnn-size", type=int, default=256, help="input size for the ACNN-only check")
    p.add_argument("--tau-c", type=float, default=0.2)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--lambda-rd", type=float, default=0.5)
    p.add_argument("--optimize-steps", type=int, default=20)
    p.add_argument("--pretrained", action="store_true", help="load ImageNet ResNet18 weights")
    p.add_argument("--lora", action="store_true", help="wrap the model with LoRA_Sam as in train.py")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"device: {args.device}")

    if args.part in ("dyra", "all"):
        run_dyra(args)
    if args.part in ("full", "all"):
        if args.part == "full":
            args.batch_size = min(args.batch_size, 2)
        run_full(args)
    print("\nAll dummy checks passed.")


if __name__ == "__main__":
    main()
