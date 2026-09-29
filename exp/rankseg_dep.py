import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


TRUNCATE_PROB = 0.5
logger = logging.getLogger(__name__)


def convert_to_nonoverlap_increment(overlap_predict, prob, sorted_prob, vol_mean, num_steps, min_num_per_step, **kwargs):
    num_class = overlap_predict.size(0)
    nonoverlap_predict = torch.ones_like(overlap_predict[0], dtype=torch.uint8) * num_class
    assert num_class < 256, 'num_class should be less than 256, when using uint8'
    overlap_mask = overlap_predict.sum(0) > 1
    
    for c in range(num_class):
        if sorted_prob[c][0] <= TRUNCATE_PROB:
            continue
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c

    not_resolved_mask = nonoverlap_predict == num_class
    num_to_resolve = not_resolved_mask.sum().item()

    num_steps = max(1, min(num_steps, (num_to_resolve+min_num_per_step-1) // min_num_per_step))
    num_overlaps_to_resolve_per_step: list = [num_to_resolve // num_steps] * num_steps
    num_overlaps_to_resolve_per_step[-1] += num_to_resolve % num_steps


    for s, num_overlaps_to_resolve_this_step in enumerate(num_overlaps_to_resolve_per_step):
        increment_score = torch.zeros_like(prob, dtype=torch.float32)
        for c in range(num_class):
            if sorted_prob[c][0] <= TRUNCATE_PROB:  # TODO: review this prune
                continue
            predict_c_mask = nonoverlap_predict == c
            nonoverlap_tau = predict_c_mask.sum()

            denominator = nonoverlap_tau + vol_mean[c] + 1

            nonoverlap_discount = 1 / denominator 
            nonoverlap_s = 2 * prob[c] * nonoverlap_discount
            nonoverlap_score = nonoverlap_s[predict_c_mask].sum()

            add_one_discount = 1 / (denominator + 1)
            add_one_s = 2 * prob[c] * add_one_discount
            add_one_score = add_one_s[predict_c_mask].sum()

            diff = add_one_score - nonoverlap_score

            increment_score[c] = add_one_s + diff
    
        increment_max, increment_argmax_mask = torch.max(increment_score, dim=0)
        _, topk_indices = torch.topk(increment_max[not_resolved_mask], k=num_overlaps_to_resolve_this_step)
        not_resolved_mask_indices = not_resolved_mask.nonzero(as_tuple=True)[0]
        indices_to_resolve_this_step = not_resolved_mask_indices[topk_indices]
        nonoverlap_predict[indices_to_resolve_this_step] = increment_argmax_mask[indices_to_resolve_this_step].type(torch.uint8)
        assert not_resolved_mask[indices_to_resolve_this_step].all()
        not_resolved_mask[indices_to_resolve_this_step] = False

    assert not not_resolved_mask.any()
    assert (nonoverlap_predict < num_class).all(), "nonoverlap_predict has values >= num_class"
    return nonoverlap_predict


def get_kernel(size, theta, amplitude, device='cpu'):
    """
    Creates a fixed 2D Gaussian kernel with given parameters.
    size: The spatial extent of the kernel, an int (e.g., 31 for 31x31) or a tuple (h, w).
    theta: Controls the spread of dependency.
    amplitude: Controls the strength of the relaxation.
    """
    h, w = (size, size) if isinstance(size, int) else size
    assert h % 2 == 1 and w % 2 == 1, "Kernel size should be odd."

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    dist = torch.sqrt((grid_x - w // 2) ** 2 + (grid_y - h // 2) ** 2)
    kernel = amplitude * torch.exp(-dist**2 / (2 * theta**2))

    return kernel
    

def estimate_mu_var_by_kernel(
    prob: torch.Tensor,
    kernel: torch.Tensor,
    use_fft: bool = True,
    max_fft_numel: int = 2 ** 27,
):
    """
    Estimates mu and sigma_sq for a batch of images.
    prob: Tensor of shape (B, H, W)
    kernel: Tensor of shape (h, w); h, w are smaller than H, W
    return_var: Bool; if true, compute and return var
    use_fft: Bool; if true, use fft for convolution; otherwise, use F.conv2d (maybe faster if dimension is small)
    max_fft_numel: Int; max number of elements of the padded tensor processed by one FFT call
    """
    B, H, W = prob.shape
    h, w = kernel.shape
    device = prob.device
    
    nu = torch.sqrt(prob * (1 - prob) + 1e-10)
    
    if use_fft:
        # Pad to size (H+h//2, W+w//2), which suffices to avoid circular wrap-around within the (H, W) output
        target_h, target_w = H + h // 2, W + w // 2
        k_padded = F.pad(kernel, (0, target_w - w, 0, target_h - h))
        
        # Roll kernel so distance=0 is at (0,0) for FFT
        k_padded = torch.roll(k_padded, shifts=(-(h // 2), -(w // 2)), dims=(-2, -1))
        k_fft = torch.fft.rfft2(k_padded)

        # Process slices in chunks to bound the peak memory of the padded FFTs
        chunk = max(1, max_fft_numel // (target_h * target_w))
        nu_conv_kappa = torch.empty_like(nu)
        for i in range(0, B, chunk):
            nu_padded = F.pad(nu[i:i + chunk], (0, target_w - W, 0, target_h - H))
            conv_result = torch.fft.irfft2(torch.fft.rfft2(nu_padded) * k_fft, s=(target_h, target_w))
            nu_conv_kappa[i:i + chunk] = conv_result[:, :H, :W]
    else:
        k_4d = kernel.view(1, 1, h, w)
        nu_conv_kappa = F.conv2d(nu.view(B, 1, H, W), k_4d, padding=(h//2, w//2))
        nu_conv_kappa = nu_conv_kappa[:, 0, :H, :W]

    q = prob.view(B, -1).sum(dim=-1, keepdim=True).view(B, 1, 1)
    mu = q + (nu / (prob + 1e-8)) * nu_conv_kappa

    return mu


def rankseg_dep(
        prob: Union[torch.Tensor, np.ndarray],
        opt_metric: str='dice',
        allow_overlap: bool=False,
        in_batch: bool=True,
        to_nonverlap_method: str='increment',
        size: int=None,
        theta: float=300.0,
        amplitude: float=1.0
    ) -> Union[torch.Tensor, tuple]:
    is_binary = (prob.shape[1] == 2) and not allow_overlap

    # Check input and convert to tensor if needed
    if isinstance(prob, np.ndarray):
        prob = torch.from_numpy(prob)
    prob = prob.float()

    # check if required memory too large (~3GB), then use cpu
    if prob.nelement() * prob.element_size() > 3e9:  # TODO: review this threshold
        prob = prob.cpu()
        logger.warning(f'Input tensor is too large: {prob.nelement() * prob.element_size() / 1e9:.2f} GB; use CPU instead')

    batch_size = prob.shape[0] if in_batch else 1
    prob = prob.unsqueeze(0) if not in_batch else prob

    if is_binary:
        prob = prob[:, 1:2, ...]
        num_class = 1

    assert opt_metric in ['iou', 'dice'], 'opt_metric should be iou or dice'

    if to_nonverlap_method == 'increment':
        to_nonverlap_fn = convert_to_nonoverlap_increment
    else:
        raise ValueError(f'Invalid to_nonverlap_method: {to_nonverlap_method}')
    
    device = prob.device
    num_class = prob.shape[1]
    img_shape = prob.shape[2:]

    # size=None: the kernel covers all displacements within the image, i.e., no truncation
    kernel_size = size if size is not None else tuple(2 * n - 1 for n in img_shape)
    kernel = get_kernel(size=kernel_size, theta=theta, amplitude=amplitude, device=device)
    vol_mean = estimate_mu_var_by_kernel(prob.view(-1, *img_shape), kernel, use_fft=True)
    vol_mean = vol_mean.view(-1, num_class, *img_shape)

    prob = torch.flatten(prob, start_dim=2, end_dim=-1)
    vol_mean = torch.flatten(vol_mean, start_dim=2, end_dim=-1)
    dim = prob.shape[-1]


    sorted_prob, top_index = torch.sort(prob, dim=-1, descending=True)

    overlap_predict = make_overlap_predict_taylor_moving_avg_fix_point(
        opt_metric=opt_metric,
        prob=prob,
        sorted_prob=sorted_prob,
        sorted_index_by_prob=top_index,
        vol_mean=vol_mean,
    )

    if allow_overlap:
        predict = overlap_predict.reshape(batch_size, num_class, *img_shape) if in_batch else overlap_predict.squeeze(0).reshape(num_class, *img_shape)
    else:
        if is_binary:
            nonoverlap_predict = overlap_predict[:, 0, ...].long()
        else:
            nonoverlap_predict = torch.zeros(batch_size, dim, dtype=torch.uint8, device=device)
            for b in range(batch_size):
                nonoverlap_predict[b] = to_nonverlap_fn(overlap_predict[b], prob=prob[b], sorted_prob=sorted_prob[b], vol_mean=vol_mean[b], num_steps=1, min_num_per_step=1000)
        predict = nonoverlap_predict.reshape(batch_size, *img_shape) if in_batch else nonoverlap_predict.reshape(*img_shape)

    return predict #, intermediate_results

def make_overlap_predict_taylor_moving_avg_fix_point(
        opt_metric: str, 
        prob: torch.Tensor,
        sorted_prob: torch.Tensor,
        sorted_index_by_prob: torch.Tensor,
        vol_mean: torch.Tensor,
        max_iters: int=10,
    ) -> torch.Tensor:
    batch_size, num_class, dim = prob.shape
    range_of_tau = torch.arange(1, dim + 1, device=prob.device)
    overlap_predict = torch.zeros_like(prob, dtype=torch.bool)

    def _fixed_point_optimization(b, c, prob, vol_mean):
        score_for_ranking = prob
        last_best_dice = -1.0
        
        for it in range(1, max_iters+1):
            if it == 1:
                sorted_score, sorted_index = sorted_prob[b, c], sorted_index_by_prob[b, c]
                _sorted_prob = sorted_prob[b, c]
            else:
                sorted_score, sorted_index = torch.sort(score_for_ranking, dim=-1, descending=True)
                _sorted_prob = torch.gather(prob, dim=-1, index=sorted_index)

            # cumsum_sorted_prob = torch.cumsum(_sorted_prob, dim=-1)
            sorted_vol_mean = torch.gather(vol_mean, dim=-1, index=sorted_index)

            cumsum_sorted_vol_mean = torch.cumsum(sorted_vol_mean, dim=-1)
            cummean_sorted_vol_mean = cumsum_sorted_vol_mean / range_of_tau
            # sorted_diff_mean = sorted_vol_mean - cummean_sorted_vol_mean

            Z_0 = torch.cumsum(_sorted_prob, dim=-1)
            Z_1 = torch.cumsum(_sorted_prob * sorted_vol_mean, dim=-1)
            Z_2 = torch.cumsum(_sorted_prob * sorted_vol_mean.pow(2), dim=-1)
            Z_3 = torch.cumsum(_sorted_prob * sorted_vol_mean.pow(3), dim=-1)
    
            # cross_term = torch.cumsum(_sorted_prob * sorted_vol_mean, dim=-1)
            # sq_term = torch.cumsum(_sorted_prob * sorted_vol_mean ** 2, dim=-1)

            discount = 1 / (range_of_tau + cummean_sorted_vol_mean)

            # zero_term = cumsum_sorted_prob * discount
            # first_term = (cross_term - cummean_sorted_vol_mean * cumsum_sorted_prob) * discount ** 2
            # second_term = (sq_term - 2*cummean_sorted_vol_mean*cross_term + (cummean_sorted_vol_mean ** 2) * cumsum_sorted_prob) * discount ** 3
            # dice = zero_term - first_term + second_term

            # cm_z0 = cummean_sorted_vol_mean * Z_0
            cm2 = cummean_sorted_vol_mean.pow(2)
            # cm3 = cummean_sorted_vol_mean.pow(3)
            T_0 = Z_0 * discount
            T_1 = (Z_1 - cummean_sorted_vol_mean * Z_0) * discount.pow(2)
            T_2 = (Z_2 - 2 * cummean_sorted_vol_mean * Z_1 + cm2 * Z_0) * discount.pow(3)
            # T_3 = (Z_3 - 3 * cummean_sorted_vol_mean * Z_2 + 3 * cm2 * Z_1 - cm3 * Z_0) * discount.pow(4)
            dice = T_0 - T_1 + T_2 #+ T_3
    
            opt_tau = torch.argmax(dice, dim=-1)
            best_dice = dice[opt_tau]
            
            score_for_ranking = prob / (vol_mean + opt_tau)

            # print(f"    iter {it}: tau={opt_tau+1}, best_dice={best_dice}")

            if abs(best_dice - last_best_dice) < 1e-6:
                break
            last_best_dice = best_dice

        if it == max_iters:
            print(f"    Not converge in {max_iters} iterations")
        overlap_predict[b, c, sorted_index[:opt_tau.item()+1]] = True

    for b in range(batch_size):
        for c in range(num_class):
            if sorted_prob[b, c, 0] < TRUNCATE_PROB:
                continue
            # logger.info(f"Starting with ranking over probabilities")
            _fixed_point_optimization(b, c, prob[b, c], vol_mean[b, c])
    return overlap_predict
