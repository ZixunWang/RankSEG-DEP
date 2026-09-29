import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from convcrf import convcrf


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    Invert (x - mean) / std normalization back to [0, 255].
    Supports [C,H,W] or [B,C,H,W]. Returns uint8 tensor on same device.
    """
    MEAN = torch.tensor([123.68, 116.28, 103.53])
    STD  = torch.tensor([58.40, 57.12, 57.38])

    device = img.device
    mean = MEAN.to(device)
    std  = STD.to(device)

    if img.dim() == 3:           # [C, H, W]
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)
    elif img.dim() == 4:         # [B, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std  = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.dim()}D")

    img = img * std + mean
    return img.clamp(0, 255).to(torch.uint8)


def fix_image_shape(image: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """
    Accepts greyscale or RGB in any of the common layouts:
        (H, W)           grey, single image
        (B, H, W)        grey, batched
        (1, H, W)        grey, single image, channel-first
        (B, 1, H, W)     grey, batched, channel-first
        (3, H, W)        RGB, single image
        (B, 3, H, W)     RGB, batched
    Returns a tensor of shape (B, 3, H, W). Greyscale is replicated to 3 channels.
    """
    # --- add channel dim if missing ---
    if image.dim() == 2:                       # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    elif image.dim() == 3:
        if image.shape[0] in (1, 3) and image.shape[-2:] == (H, W):
            image = image.unsqueeze(0)                   # (1,C,H,W)
        else:                                            # (B,H,W) grey batched
            image = image.unsqueeze(1)                   # (B,1,H,W)
    elif image.dim() == 4:
        pass                                             # (B,C,H,W)
    else:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    # --- replicate grey -> 3 channels ---
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    elif image.shape[1] != 3:
        raise ValueError(
            f"Image must have 1 or 3 channels, got {image.shape[1]}"
        )

    # --- sanity checks ---
    assert image.shape[0] == B, f"batch mismatch: image {image.shape[0]} vs prob {B}"
    assert image.shape[-2:] == (H, W), \
        f"spatial mismatch: image {tuple(image.shape[-2:])} vs prob {(H, W)}"
    return image.contiguous()


@torch.no_grad()
def densecrf_infer(
    image: torch.Tensor,
    prob: torch.Tensor,
    *,
    is_logits: bool = False,
    n_iters: int = 10,
    sxy_gaussian: float = 1.0,
    compat_gaussian: float = 3.0,
    sxy_bilateral: float = 70.0,
    srgb_bilateral: float = 3.0,
    compat_bilateral: float = 4.0,
) -> torch.Tensor:
    """
    DenseCRF post-processing.

    Args:
        image: grey or RGB tensor; see _normalize_image for accepted shapes.
               uint8 or float (any range; auto-scaled to uint8 [0,255]).
        prob:  (B,C,H,W) or (C,H,W). softmax probs (or logits if is_logits=True).
    Returns:
        (B,H,W) torch.long label map on prob.device.
    """
    device = prob.device

    if prob.dim() == 3:
        prob = prob.unsqueeze(0)
    if is_logits:
        prob = torch.softmax(prob, dim=1)
    B, C, H, W = prob.shape

    image = denormalize(image)
    image = fix_image_shape(image, B, H, W)           # (B,3,H,W)

    # ---- image -> uint8 HxWx3 on CPU ----
    img_cpu = image.detach().cpu()
    if img_cpu.dtype != torch.uint8:
        if img_cpu.max() <= 1.0 + 1e-6:
            img_cpu = img_cpu * 255.0
        img_cpu = img_cpu.clamp(0, 255).to(torch.uint8)
    img_np = img_cpu.permute(0, 2, 3, 1).contiguous().numpy()   # (B,H,W,3)

    prob_np = prob.detach().cpu().numpy()                        # (B,C,H,W)

    out = np.empty((B, H, W), dtype=np.int64)
    for b in range(B):
        d = dcrf.DenseCRF2D(W, H, C)
        unary = np.ascontiguousarray(unary_from_softmax(prob_np[b]))
        d.setUnaryEnergy(unary)

        d.addPairwiseGaussian(
            sxy=sxy_gaussian,
            compat=compat_gaussian,
        )
        d.addPairwiseBilateral(
            sxy=sxy_bilateral,
            srgb=srgb_bilateral,
            rgbim=np.ascontiguousarray(img_np[b]),
            compat=compat_bilateral,
        )

        Q = d.inference(n_iters)                                 # (C, H*W)
        out[b] = np.argmax(Q, axis=0).reshape(H, W)

    return torch.from_numpy(out).to(device=device, dtype=torch.long)


def _build_convcrf(shape, nclasses, device, conf=None):
    if conf is None:
        conf = convcrf.default_conf.copy()
        conf["pyinn"] = False
    use_gpu = device.type == "cuda"
    crf = convcrf.GaussCRF(conf=conf, shape=shape, nclasses=nclasses, use_gpu=use_gpu)
    return crf.to(device)


@torch.no_grad()
def convcrf_infer(
    image: torch.Tensor,
    prob: torch.Tensor,
    *,
    is_logits: bool = False,
    n_iters: int = 5,
    conf: dict | None = None,
) -> torch.Tensor:
    """
    ConvCRF post-processing (GPU-native).

    Args:
        image: grey or RGB tensor; see _normalize_image for accepted shapes.
               float or uint8; auto-scaled to ~[0,255].
        prob:  (B,C,H,W) or (C,H,W). softmax probs (or logits if is_logits=True).
    Returns:
        (B,H,W) torch.long label map on prob.device.
    """
    device = prob.device

    if prob.dim() == 3:
        prob = prob.unsqueeze(0)
    if is_logits:
        prob = torch.softmax(prob, dim=1)
    B, C, H, W = prob.shape

    image = denormalize(image)
    image = fix_image_shape(image, B, H, W).to(device=device, dtype=torch.float32)
    if image.max() <= 1.0 + 1e-6:
        image = image * 255.0

    crf = _build_convcrf((H, W), nclasses=C, device=device, conf=conf)
    try:
        refined = crf(unary=prob.to(device), img=image, num_iter=n_iters)
    except TypeError:
        refined = crf(unary=prob.to(device), img=image)

    return refined.argmax(dim=1).to(torch.long)
